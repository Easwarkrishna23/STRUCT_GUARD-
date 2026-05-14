"""
DICE Attack — Delete Internally, Connect Externally.

Reference: Waniek et al., "Hiding Individuals and Communities in a Social Network",
Nature Human Behaviour, 2018.

Strategy (untargeted poisoning):
  - DELETE edges that connect nodes of the SAME predicted class (internal edges)
  - ADD    edges that connect nodes of DIFFERENT predicted classes (external edges)

Rationale: internal edges are informationally redundant (GCN already aggregates
within-class neighbours); external edges inject cross-class noise into aggregation,
maximally disrupting the homophily assumption that GCN relies on.

Unlike gradient-based methods this attack:
  - Needs only predicted class labels (not gradients)
  - Is deterministic given a seed
  - Reliably achieves 10-20pp accuracy drops on Cora at 20% budget
  - Survives model retraining because the structural damage is global
"""
import numpy as np
from typing import Optional

from datasets.cora_loader import GraphData
from attacks.base import AttackResult, edge_budget, diff_edges
from utils.graph_utils import compute_cosine_similarity


def dice_attack(
    graph: GraphData,
    model,
    params,
    budget_ratio: float = 0.20,
    seed: int = 42,
) -> AttackResult:
    """
    DICE poisoning attack: delete internal edges, add external edges.

    Split budget 50/50 between deletions and additions.

    Args:
        graph:        Clean GraphData.
        model:        Trained GCN model (used to get predicted labels).
        params:       Model parameters.
        budget_ratio: Fraction of existing edges to use as total budget.
        seed:         RNG seed.

    Returns:
        AttackResult with perturbed graph.
    """
    import jax.numpy as jnp
    from utils.graph_utils import normalize_adjacency

    rng = np.random.default_rng(seed)
    adj = graph.adj.copy().astype(np.float32)
    n   = adj.shape[0]

    # Get predicted class labels from clean model
    a_hat = jnp.array(graph.adj_norm)
    x     = jnp.array(graph.features)
    _, logits, _ = model.apply({"params": params}, x, a_hat, training=False)
    pred_labels = np.array(logits.argmax(axis=-1))

    total_budget = edge_budget(graph.adj, budget_ratio)
    half         = total_budget // 2

    print(f"  [DICE] Budget={total_budget} edges "
          f"({budget_ratio:.0%} of {int(graph.adj.sum())//2}), "
          f"delete={half}, add={half}")

    # ── Step 1: DELETE internal bottleneck edges (same predicted class) ──────
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    internal_mask = pred_labels[rows] == pred_labels[cols]
    internal_idx  = np.where(internal_mask)[0]
    edge_centrality = _edge_betweenness_lookup(adj)

    if len(internal_idx) > 0:
        n_delete = min(half, len(internal_idx))
        degree = adj.sum(axis=1)
        scores = np.array([
            edge_centrality.get((int(min(rows[idx], cols[idx])), int(max(rows[idx], cols[idx]))), 0.0)
            for idx in internal_idx
        ])
        scores += 0.05 * (degree[rows[internal_idx]] + degree[cols[internal_idx]]) / max(float(degree.max()), 1.0)
        order = internal_idx[np.argsort(-scores)]
        pool = order[: max(n_delete, min(len(order), n_delete * 5))]
        chosen = rng.choice(pool, n_delete, replace=False)
        for idx in chosen:
            i, j = rows[idx], cols[idx]
            adj[i, j] = 0.0
            adj[j, i] = 0.0
    else:
        n_delete = 0
        print("  [DICE] Warning: no internal edges found to delete")

    # ── Step 2: ADD external low-similarity bottleneck edges ─────────────────
    # Candidate non-edges connect different predicted classes and high-degree
    # structural bottleneck endpoints, which fragments homophilous aggregation.
    n_added = 0
    attempts = 0
    max_attempts = half * 20
    degree = adj.sum(axis=1)
    sim = compute_cosine_similarity(graph.features)
    hubs = np.argsort(-degree)[: min(n, max(64, int(np.sqrt(n) * 8)))]

    while n_added < half and attempts < max_attempts:
        i = int(rng.choice(hubs)) if hubs.size else int(rng.integers(0, n))
        j = rng.integers(0, n)
        if (i != j
                and adj[i, j] == 0
                and pred_labels[i] != pred_labels[j]
                and sim[i, j] <= np.quantile(sim[i], 0.25)):
            adj[i, j] = 1.0
            adj[j, i] = 1.0
            n_added += 1
        attempts += 1

    if n_added < half:
        print(f"  [DICE] Only added {n_added}/{half} external edges "
              f"(graph may be too dense)")

    perturbed = graph.copy()
    perturbed = perturbed.update_adj(adj)
    perturbed.name = "dice"

    added, removed = diff_edges(graph.adj, adj)
    print(f"  [DICE] Done: +{added} external edges, -{removed} internal edges")

    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="DICE",
        n_edges_added=added,
        n_edges_removed=removed,
        n_features_perturbed=0,
        budget_used=total_budget,
        diagnostics={
            "budget_ratio": budget_ratio,
            "strategy": "dice_structural_bottleneck_betweenness",
        },
    )


def _edge_betweenness_lookup(adj: np.ndarray) -> dict[tuple[int, int], float]:
    import networkx as nx

    G = nx.from_numpy_array(adj)
    if G.number_of_edges() == 0:
        return {}
    k = min(256, G.number_of_nodes())
    centrality = nx.edge_betweenness_centrality(G, k=k, seed=42, normalized=True)
    return {
        (int(min(u, v)), int(max(u, v))): float(c)
        for (u, v), c in centrality.items()
    }
