import numpy as np

from datasets.cora_loader import GraphData
from defense.edge_pruning import edge_pruning
from defense.graph_reconstruction import graph_reconstruction
from evaluation.metrics import (
    assortativity_coefficient,
    attack_success_rate,
    bose_einstein_fitness,
    clean_label_recovery,
    embedding_drift,
    homophily_drop,
    injected_edge_prune_rate,
    neighborhood_entropy,
    power_law_exponent,
    recovery_rate,
)
from utils.config import DefenseConfig


def _graph(adj, features, labels=None):
    labels = np.asarray(labels if labels is not None else np.zeros(adj.shape[0]), dtype=np.int64)
    n = adj.shape[0]
    mask = np.ones(n, dtype=bool)
    from utils.graph_utils import normalize_adjacency

    return GraphData(
        adj=adj.astype(np.float32),
        adj_norm=normalize_adjacency(adj),
        features=features.astype(np.float32),
        labels=labels,
        train_mask=mask.copy(),
        val_mask=mask.copy(),
        test_mask=mask.copy(),
        num_nodes=n,
        num_features=features.shape[1],
        num_classes=int(labels.max()) + 1 if labels.size else 1,
        name="toy",
    )


def test_robustness_metrics():
    y = np.array([0, 1, 1])
    clean = np.array([0, 1, 0])
    attacked = np.array([1, 1, 0])
    assert attack_success_rate(np.array([0, 1]), y, clean, attacked) == 0.5
    assert recovery_rate(0.8, 0.2, 0.8) == 1.0

    clean_emb = np.array([[1.0, 0.0], [0.0, 2.0]])
    other_emb = np.array([[2.0, 0.0], [0.0, 1.0]])
    assert embedding_drift(clean_emb, other_emb) > 0.0

    clean_adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    attacked_adj = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
    labels = np.array([0, 0, 1])
    assert neighborhood_entropy(attacked_adj, labels) >= 0.0
    assert homophily_drop(clean_adj, attacked_adj, labels) > 0.0
    assert assortativity_coefficient(clean_adj) <= 1.0
    metric_features = np.eye(3, dtype=np.float32)
    assert bose_einstein_fitness(clean_adj, metric_features) >= 0.0
    assert power_law_exponent(clean_adj) >= 0.0
    defended = np.array([0, 1, 1])
    assert clean_label_recovery(y, clean, attacked, defended) == 1.0
    defended_adj = clean_adj.copy()
    assert injected_edge_prune_rate(clean_adj, attacked_adj, defended_adj) == 1.0


def test_centrality_low_cosine_bridge_is_pruned():
    adj = np.zeros((6, 6), dtype=np.float32)
    for u, v in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3), (1, 4)]:
        adj[u, v] = adj[v, u] = 1.0
    features = np.array(
        [[1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [0, 1]],
        dtype=np.float32,
    )
    cfg = DefenseConfig(
        min_edges_ratio=0.5,
        centrality_quantile=0.5,
        centrality_cosine_threshold=0.2,
    )
    pruned, stats = edge_pruning(_graph(adj, features), cfg)
    assert stats["centrality_pruned_candidates"] > 0
    assert pruned.adj[2, 3] == 0.0


def test_degree_anomalous_node_loses_low_fitness_edges():
    n = 8
    adj = np.zeros((n, n), dtype=np.float32)
    for v in range(1, n):
        adj[0, v] = adj[v, 0] = 1.0
    for u, v in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 7)]:
        adj[u, v] = adj[v, u] = 1.0
    features = np.eye(n, dtype=np.float32)
    cfg = DefenseConfig(
        min_edges_ratio=0.5,
        degree_z_threshold=1.0,
        degree_target_z=0.2,
        centrality_quantile=1.0,
    )
    pruned, stats = edge_pruning(_graph(adj, features), cfg)
    assert stats["degree_pruned_candidates"] > 0
    assert pruned.adj[0].sum() < adj[0].sum()


def test_small_world_reconstruction_rejects_cluster_harming_edge():
    adj = np.zeros((6, 6), dtype=np.float32)
    for u, v in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        adj[u, v] = adj[v, u] = 1.0
    features = np.array(
        [[1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1]],
        dtype=np.float32,
    )
    cfg = DefenseConfig(knn_k=1)
    reconstructed, stats = graph_reconstruction(_graph(adj, features), cfg)
    assert reconstructed.adj[0, 3] == 0.0
    assert stats["rejected_clustering"] > 0
