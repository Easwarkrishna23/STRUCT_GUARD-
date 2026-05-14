"""Small-world kNN graph reconstruction for STRUC-GUARD+."""
from __future__ import annotations

import numpy as np

from datasets.cora_loader import GraphData
from utils.config import DefenseConfig
from utils.graph_utils import compute_cosine_similarity


def graph_reconstruction(
    graph: GraphData,
    cfg: DefenseConfig,
) -> tuple[GraphData, dict]:
    """
    Add high-confidence kNN edges only when they improve local clustering.

    Average path length is monitored on the largest connected component. Edge
    additions cannot increase shortest-path distances, but the tolerance guard
    keeps the small-world criterion explicit and testable.
    """
    import networkx as nx

    adj = (graph.adj > 0).astype(np.float32)
    sim = compute_cosine_similarity(graph.features)
    np.fill_diagonal(sim, -np.inf)

    G = nx.from_numpy_array(adj)
    base_apl = _sampled_average_path_length(G)
    base_hier_loss = _hierarchical_clustering_loss(G, cfg)
    accepted = 0
    rejected_clustering = 0
    rejected_path = 0
    rejected_hierarchy = 0

    n = adj.shape[0]
    path_checked = 0
    for u in range(n):
        candidates = np.argsort(sim[u])[-cfg.knn_k * 3 :][::-1]
        added_for_u = 0
        for v in candidates:
            v = int(v)
            if u == v or adj[u, v] > 0:
                continue
            before = _endpoint_clustering(G, u, v)
            G.add_edge(u, v)
            after = _endpoint_clustering(G, u, v)
            hierarchy_loss = _hierarchical_clustering_loss(G, cfg)
            apl = _sampled_average_path_length(G) if path_checked % 25 == 0 else base_apl
            path_checked += 1
            path_reduction = base_apl - apl
            max_apl = base_apl * (1.0 + cfg.small_world_apl_tolerance)
            hierarchy_ok = (
                hierarchy_loss <= cfg.hierarchical_slope_tolerance
                or hierarchy_loss <= base_hier_loss
            )
            path_ok = apl <= max_apl and path_reduction <= cfg.path_length_reduction_epsilon

            if after >= before and hierarchy_ok and path_ok:
                adj[u, v] = 1.0
                adj[v, u] = 1.0
                accepted += 1
                added_for_u += 1
            else:
                G.remove_edge(u, v)
                if after < before:
                    rejected_clustering += 1
                elif not hierarchy_ok:
                    rejected_hierarchy += 1
                else:
                    rejected_path += 1

            if added_for_u >= cfg.knn_k:
                break

    edges_before = int(np.triu(graph.adj, k=1).sum())
    edges_after = int(np.triu(adj, k=1).sum())
    stats = {
        "edges_before_reconstruction": edges_before,
        "edges_after_reconstruction": edges_after,
        "edges_added_by_knn": edges_after - edges_before,
        "accepted_candidates": accepted,
        "rejected_clustering": rejected_clustering,
        "rejected_hierarchy": rejected_hierarchy,
        "rejected_path_length": rejected_path,
        "base_average_path_length": base_apl,
        "final_average_path_length": _sampled_average_path_length(G),
        "base_hierarchical_clustering_loss": base_hier_loss,
        "final_hierarchical_clustering_loss": _hierarchical_clustering_loss(G, cfg),
        "knn_k": cfg.knn_k,
    }
    print(f"  [Small-World KNN] edges: {edges_before} -> {edges_after} "
          f"(accepted={accepted}, rejected={rejected_clustering + rejected_path})")

    reconstructed = graph.copy().update_adj(adj)
    reconstructed.name = graph.name.replace("_smoothed", "") + "_defended"
    return reconstructed, stats


def _endpoint_clustering(G, u: int, v: int) -> float:
    import networkx as nx

    values = nx.clustering(G, nodes=[u, v])
    return float((values.get(u, 0.0) + values.get(v, 0.0)) / 2.0)


def _safe_average_path_length(G) -> float:
    import networkx as nx

    if G.number_of_nodes() <= 1:
        return 0.0
    if nx.is_connected(G):
        H = G
    else:
        largest = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest).copy()
    if H.number_of_nodes() <= 1:
        return 0.0
    return float(nx.average_shortest_path_length(H))


def _sampled_average_path_length(G, samples: int = 64, seed: int = 42) -> float:
    import networkx as nx

    if G.number_of_nodes() <= 1:
        return 0.0
    if nx.is_connected(G):
        H = G
    else:
        largest = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest).copy()
    nodes = list(H.nodes())
    if len(nodes) <= 1:
        return 0.0
    rng = np.random.default_rng(seed)
    src = nodes if len(nodes) <= samples else rng.choice(nodes, size=samples, replace=False)
    distances = []
    for node in src:
        lengths = nx.single_source_shortest_path_length(H, int(node))
        distances.extend([d for target, d in lengths.items() if target != int(node)])
    return float(np.mean(distances)) if distances else 0.0


def _hierarchical_clustering_loss(G, cfg: DefenseConfig) -> float:
    """Loss from the scale-free hierarchy target C(k) ~ k^-1."""
    import networkx as nx

    if G.number_of_edges() == 0:
        return float("inf")
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    buckets: dict[int, list[float]] = {}
    for node, deg in degrees.items():
        if deg <= 1:
            continue
        buckets.setdefault(int(deg), []).append(float(clustering.get(node, 0.0)))
    xs, ys = [], []
    for deg, vals in buckets.items():
        c = float(np.mean(vals))
        if c > 0.0:
            xs.append(np.log(float(deg)))
            ys.append(np.log(c))
    if len(xs) < 3:
        return 0.0
    slope = float(np.polyfit(np.asarray(xs), np.asarray(ys), 1)[0])
    return abs(slope - cfg.hierarchical_slope_target)
