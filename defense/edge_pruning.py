"""STRUC-GUARD+ topological integrity filter."""
from __future__ import annotations

import numpy as np

from datasets.cora_loader import GraphData
from defense.semantic_reasoning import build_topic_sets, topic_jaccard
from utils.config import DefenseConfig
from utils.graph_utils import check_connectivity, compute_cosine_similarity


def edge_pruning(
    graph: GraphData,
    cfg: DefenseConfig,
    suspicious_edges: set[tuple[int, int]] | None = None,
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

    rows, cols = np.where(np.triu(adj, k=1) > 0)
    orig_edges = len(rows)
    if orig_edges == 0:
        return graph.copy(), {"edges_before": 0, "edges_after": 0, "edges_removed": 0}

    prune: set[tuple[int, int]] = set(suspicious_edges)
    centrality_pruned = _centrality_prune_edges(adj, sim, cfg)
    prune.update(centrality_pruned)

    degree_pruned = _degree_consistency_prune_edges(adj, sim, graph.features, cfg)
    prune.update(degree_pruned)

    adj_pruned = adj.copy()
    for u, v in prune:
        if adj_pruned[u, v] > 0:
            adj_pruned[u, v] = 0.0
            adj_pruned[v, u] = 0.0

    min_keep = int(orig_edges * cfg.min_edges_ratio)
    if int(np.triu(adj_pruned, k=1).sum()) < min_keep:
        adj_pruned = _restore_min_edges(adj, adj_pruned, sim, min_keep)

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


def _restore_min_edges(
    adj_orig: np.ndarray,
    adj_pruned: np.ndarray,
    sim: np.ndarray,
    min_keep: int,
) -> np.ndarray:
    """Re-add highest-similarity removed edges until edge count reaches min_keep."""
    adj_new = adj_pruned.copy()
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
