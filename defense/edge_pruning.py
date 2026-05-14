"""STRUC-GUARD+ topological integrity filter."""
from __future__ import annotations

import numpy as np

from datasets.cora_loader import GraphData
from defense.semantic_reasoning import build_topic_sets, topic_jaccard
from evaluation.metrics import power_law_exponent
from utils.config import DefenseConfig
from utils.graph_utils import check_connectivity, compute_cosine_similarity


def edge_pruning(
    graph: GraphData,
    cfg: DefenseConfig,
    suspicious_edges: set[tuple[int, int]] | None = None,
    suspicious_nodes: set[int] | None = None,
) -> tuple[GraphData, dict]:
    """
    Prune suspicious edges using centrality, cosine, and degree consistency.

    STRUC-GUARD+ combines:
      1. Semantic suspicious-edge flags from ontology reasoning.
      2. Edge betweenness centrality for low-cosine bridge edges.
      3. Bose-Einstein-inspired degree fitness for anomalous nodes.
    """
    adj = (graph.adj > 0).astype(np.float32)
    sim = compute_cosine_similarity(graph.features)
    suspicious_edges = suspicious_edges or set()
    suspicious_nodes = suspicious_nodes or set()

    rows, cols = np.where(np.triu(adj, k=1) > 0)
    orig_edges = len(rows)
    if orig_edges == 0:
        return graph.copy(), {"edges_before": 0, "edges_after": 0, "edges_removed": 0}

    prune: set[tuple[int, int]] = set(suspicious_edges)
    centrality_pruned = _centrality_prune_edges(adj, sim, cfg)
    prune.update(centrality_pruned)

    degree_pruned = _degree_consistency_prune_edges(adj, sim, graph.features, cfg)
    prune.update(degree_pruned)

    preferential_pruned = _preferential_attachment_prune_edges(
        adj, sim, graph.features, cfg
    )
    prune.update(preferential_pruned)

    assortativity_pruned, assortativity_before = _assortativity_prune_edges(adj, sim, cfg)
    prune.update(assortativity_pruned)

    scale_free_pruned = _scale_free_exponent_prune_edges(adj, sim, graph.features, cfg)
    prune.update(scale_free_pruned)

    temporal_pruned = _suspicious_node_isolation_edges(adj, sim, suspicious_nodes, cfg)
    prune.update(temporal_pruned)

    adj_pruned = adj.copy()
    for u, v in prune:
        if adj_pruned[u, v] > 0:
            adj_pruned[u, v] = 0.0
            adj_pruned[v, u] = 0.0

    min_keep = int(orig_edges * cfg.min_edges_ratio)
    if int(np.triu(adj_pruned, k=1).sum()) < min_keep:
        protected = degree_pruned | scale_free_pruned | temporal_pruned
        adj_pruned = _restore_min_edges(adj, adj_pruned, sim, min_keep, protected)

    if adj_pruned.shape[0] > 0 and not check_connectivity(adj_pruned):
        adj_pruned = _restore_connectivity(adj, adj_pruned, sim)

    edges_after = int(np.triu(adj_pruned, k=1).sum())
    removed = orig_edges - edges_after
    stats = {
        "edges_before": orig_edges,
        "edges_after": edges_after,
        "edges_removed": removed,
        "removal_rate": removed / max(orig_edges, 1),
        "semantic_pruned_candidates": len(suspicious_edges),
        "centrality_pruned_candidates": len(centrality_pruned),
        "degree_pruned_candidates": len(degree_pruned),
        "preferential_pruned_candidates": len(preferential_pruned),
        "assortativity_pruned_candidates": len(assortativity_pruned),
        "scale_free_pruned_candidates": len(scale_free_pruned),
        "temporal_node_pruned_candidates": len(temporal_pruned),
        "suspicious_nodes": len(suspicious_nodes),
        "degree_assortativity_before": assortativity_before,
        "gamma_before": power_law_exponent(adj),
        "gamma_after": power_law_exponent(adj_pruned),
    }
    print(f"  [STRUC-GUARD+ Pruning] removed {removed}/{orig_edges} edges "
          f"({stats['removal_rate']:.1%})")

    pruned = graph.copy().update_adj(adj_pruned)
    pruned.name = graph.name + "_struc_guard_pruned"
    return pruned, stats


def _centrality_prune_edges(
    adj: np.ndarray,
    sim: np.ndarray,
    cfg: DefenseConfig,
) -> set[tuple[int, int]]:
    import networkx as nx

    G = nx.from_numpy_array(adj)
    if G.number_of_edges() == 0:
        return set()

    # Approximate betweenness keeps the defense usable on Elliptic snapshots.
    k = min(256, G.number_of_nodes())
    centrality = nx.edge_betweenness_centrality(G, k=k, seed=42, normalized=True)
    if not centrality:
        return set()

    threshold = float(np.quantile(list(centrality.values()), cfg.centrality_quantile))
    out: set[tuple[int, int]] = set()
    for (u, v), c in centrality.items():
        edge = (min(u, v), max(u, v))
        if c >= threshold and sim[edge] < cfg.centrality_cosine_threshold:
            out.add(edge)
    return out


def _degree_consistency_prune_edges(
    adj: np.ndarray,
    sim: np.ndarray,
    features: np.ndarray,
    cfg: DefenseConfig,
) -> set[tuple[int, int]]:
    degrees = adj.sum(axis=1)
    mean = float(degrees.mean())
    std = float(degrees.std()) or 1.0
    high_limit = mean + cfg.degree_z_threshold * std
    target = mean + cfg.degree_target_z * std
    anomalous = np.where(degrees > high_limit)[0]
    if anomalous.size == 0:
        return set()

    # Reuse topic extraction on a tiny graph-like shim.
    class _Shim:
        pass

    shim = _Shim()
    shim.features = features
    topics = build_topic_sets(shim, cfg)  # type: ignore[arg-type]

    prune: set[tuple[int, int]] = set()
    current_degree = degrees.copy()
    for node in anomalous:
        neighbors = np.where(adj[node] > 0)[0]
        scored = []
        for nb in neighbors:
            deg_consistency = 1.0 / (1.0 + abs(current_degree[nb] - mean) / std)
            j = topic_jaccard(topics[node], topics[nb])
            # Bose-Einstein-style fitness: low semantic/topological fit is pruned.
            fitness = 0.50 * float(sim[node, nb]) + 0.30 * j + 0.20 * deg_consistency
            scored.append((fitness, int(min(node, nb)), int(max(node, nb))))
        for _, u, v in sorted(scored):
            if current_degree[node] <= target:
                break
            edge = (u, v)
            if edge not in prune:
                prune.add(edge)
                current_degree[node] -= 1
    return prune


def _preferential_attachment_prune_edges(
    adj: np.ndarray,
    sim: np.ndarray,
    features: np.ndarray,
    cfg: DefenseConfig,
) -> set[tuple[int, int]]:
    """Prune semantically unrelated edges with implausible growth probability."""
    degrees = adj.sum(axis=1)
    if degrees.size == 0:
        return set()
    low_degree_cut = float(np.quantile(degrees, cfg.preferential_low_degree_quantile))
    denom = max(float(degrees.sum()) ** 2, 1.0)

    class _Shim:
        pass

    shim = _Shim()
    shim.features = features
    topics = build_topic_sets(shim, cfg)  # type: ignore[arg-type]

    rows, cols = np.where(np.triu(adj, k=1) > 0)
    prune: set[tuple[int, int]] = set()
    for u, v in zip(rows, cols):
        growth_probability = float((degrees[u] + 1.0) * (degrees[v] + 1.0) / denom)
        semantically_unrelated = topic_jaccard(topics[u], topics[v]) < cfg.topic_jaccard_threshold
        low_low = degrees[u] <= low_degree_cut and degrees[v] <= low_degree_cut
        if low_low and semantically_unrelated and growth_probability < cfg.preferential_attachment_threshold:
            prune.add((int(u), int(v)))
    return prune


def _assortativity_prune_edges(
    adj: np.ndarray,
    sim: np.ndarray,
    cfg: DefenseConfig,
) -> tuple[set[tuple[int, int]], float]:
    """Prune hub-outlier edges that increase disassortativity and have low cosine."""
    import networkx as nx

    G = nx.from_numpy_array(adj)
    if G.number_of_edges() == 0:
        return set(), 0.0
    assortativity = nx.degree_assortativity_coefficient(G)
    assortativity = 0.0 if not np.isfinite(assortativity) else float(assortativity)
    degrees = adj.sum(axis=1)
    mean = float(degrees.mean())
    std = float(degrees.std()) or 1.0
    prune: set[tuple[int, int]] = set()
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    for u, v in zip(rows, cols):
        degree_gap_z = abs(float(degrees[u] - degrees[v])) / std
        hub_outlier = degree_gap_z >= cfg.assortativity_degree_gap_z
        if hub_outlier and sim[u, v] < cfg.assortativity_cosine_threshold:
            prune.add((int(u), int(v)))
    return prune, assortativity


def _scale_free_exponent_prune_edges(
    adj: np.ndarray,
    sim: np.ndarray,
    features: np.ndarray,
    cfg: DefenseConfig,
) -> set[tuple[int, int]]:
    """Prune low-fitness eigenvector-central edges that worsen gamma consistency."""
    import networkx as nx

    G = nx.from_numpy_array(adj)
    if G.number_of_edges() == 0:
        return set()
    try:
        eig = nx.eigenvector_centrality_numpy(G)
    except Exception:
        eig = nx.degree_centrality(G)
    eig_values = np.array(list(eig.values()), dtype=np.float64)
    threshold = float(np.quantile(eig_values, cfg.eigenvector_top_quantile))
    central_nodes = {int(node) for node, val in eig.items() if val >= threshold}
    if not central_nodes:
        return set()

    base_gamma = power_law_exponent(adj)
    degrees = adj.sum(axis=1)
    mean = float(degrees.mean())
    std = float(degrees.std()) or 1.0
    prune: set[tuple[int, int]] = set()

    for node in central_nodes:
        neighbors = np.where(adj[node] > 0)[0]
        scored = []
        for nb in neighbors:
            deg_consistency = 1.0 / (1.0 + abs(degrees[nb] - mean) / std)
            fitness = 0.70 * float(sim[node, nb]) + 0.30 * deg_consistency
            scored.append((fitness, int(min(node, nb)), int(max(node, nb))))
        for fitness, u, v in sorted(scored):
            if fitness > cfg.bose_einstein_fitness_threshold:
                continue
            trial_degrees = degrees.copy()
            trial_degrees[u] -= 1
            trial_degrees[v] -= 1
            trial_gamma = power_law_exponent(trial_degrees)
            if base_gamma == 0.0 or abs(trial_gamma - base_gamma) <= cfg.gamma_tolerance:
                prune.add((u, v))
                break
    return prune


def _suspicious_node_isolation_edges(
    adj: np.ndarray,
    sim: np.ndarray,
    suspicious_nodes: set[int],
    cfg: DefenseConfig,
) -> set[tuple[int, int]]:
    """Isolate temporal-drift nodes by keeping only their best-fit incident edges."""
    prune: set[tuple[int, int]] = set()
    for node in suspicious_nodes:
        if node < 0 or node >= adj.shape[0]:
            continue
        neighbors = np.where(adj[node] > 0)[0]
        if neighbors.size == 0:
            continue
        keep = max(1, int(np.ceil(neighbors.size * cfg.suspicious_node_edge_keep_ratio)))
        order = neighbors[np.argsort(-sim[node, neighbors])]
        keep_set = set(order[:keep].astype(int).tolist())
        for nb in neighbors:
            if int(nb) not in keep_set:
                prune.add((int(min(node, nb)), int(max(node, nb))))
    return prune


def _restore_min_edges(
    adj_orig: np.ndarray,
    adj_pruned: np.ndarray,
    sim: np.ndarray,
    min_keep: int,
    forbidden_edges: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Re-add highest-similarity removed edges until edge count reaches min_keep."""
    adj_new = adj_pruned.copy()
    forbidden_edges = forbidden_edges or set()
    rows, cols = np.where(np.triu(adj_orig - adj_pruned, k=1) > 0)
    if len(rows) == 0:
        return adj_new
    sims = sim[rows, cols]
    order = np.argsort(-sims)
    current = int(np.triu(adj_new, k=1).sum())
    for idx in order:
        if current >= min_keep:
            break
        i, j = rows[idx], cols[idx]
        edge = (int(min(i, j)), int(max(i, j)))
        if edge in forbidden_edges:
            continue
        adj_new[i, j] = 1.0
        adj_new[j, i] = 1.0
        current += 1
    return adj_new


def _restore_connectivity(
    adj_orig: np.ndarray,
    adj_pruned: np.ndarray,
    sim: np.ndarray,
) -> np.ndarray:
    """Add back high-similarity original edges until connected where possible."""
    import networkx as nx

    adj_new = adj_pruned.copy()
    G = nx.from_numpy_array(adj_new)
    components = list(nx.connected_components(G))

    while len(components) > 1:
        best_sim, best_i, best_j = -1.0, -1, -1
        for c_idx in range(len(components)):
            for d_idx in range(c_idx + 1, len(components)):
                for u in components[c_idx]:
                    for v in components[d_idx]:
                        if adj_orig[u, v] > 0 and sim[u, v] > best_sim:
                            best_sim, best_i, best_j = sim[u, v], u, v
        if best_i < 0:
            break
        adj_new[best_i, best_j] = 1.0
        adj_new[best_j, best_i] = 1.0
        G = nx.from_numpy_array(adj_new)
        components = list(nx.connected_components(G))

    return adj_new
