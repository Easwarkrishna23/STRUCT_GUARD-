"""
Random Structure Attack — baseline poisoning attack.

Randomly adds and removes edges with equal probability up to a budget.
Serves as the lower-bound baseline: if the model is robust to random noise,
it should certainly be robust to smarter attacks (and vice versa).
"""
import numpy as np
from datasets.cora_loader import GraphData
from attacks.base import AttackResult, edge_budget, diff_edges
from utils.graph_utils import compute_cosine_similarity


def random_structure_attack(
    graph: GraphData,
    budget_ratio: float = 0.05,
    seed: int = 42,
) -> AttackResult:
    """
    Randomly flip edges up to budget (50% add, 50% remove).

    Args:
        graph:        Clean GraphData.
        budget_ratio: Fraction of existing edges to perturb.
        seed:         RNG seed for reproducibility.

    Returns:
        AttackResult with randomly perturbed adjacency.
    """
    rng    = np.random.default_rng(seed)
    adj    = graph.adj.copy()
    n      = adj.shape[0]
    budget = edge_budget(adj, budget_ratio)
    half   = budget // 2

    print(f"  [Random Structure] Budget={budget} "
          f"(+{half} add / -{budget-half} remove)")

    sim = compute_cosine_similarity(graph.features)

    # Remove high-similarity existing edges to disrupt homophilous aggregation.
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    if len(rows) >= budget - half:
        order = np.argsort(-sim[rows, cols])
        top = order[: max(budget - half, min(len(order), (budget - half) * 5))]
        chosen = rng.choice(top, size=budget - half, replace=False)
        for idx in chosen:
            adj[rows[idx], cols[idx]] = 0.0
            adj[cols[idx], rows[idx]] = 0.0

    # Add low-similarity non-edges to create cross-topic shortcuts.
    added = 0
    attempts = 0
    while added < half and attempts < half * 100:
        sources = rng.integers(0, n, size=min(64, n))
        best = None
        best_sim = np.inf
        for i in sources:
            candidates = np.where(adj[i] == 0)[0]
            candidates = candidates[candidates != i]
            if candidates.size == 0:
                continue
            sample = rng.choice(candidates, size=min(64, candidates.size), replace=False)
            idx = int(np.argmin(sim[i, sample]))
            if sim[i, sample[idx]] < best_sim:
                best_sim = sim[i, sample[idx]]
                best = (i, int(sample[idx]))
        if best is not None:
            i, j = best
            adj[i, j] = 1.0
            adj[j, i] = 1.0
            added += 1
        attempts += 1

    perturbed = graph.copy()
    perturbed = perturbed.update_adj(adj)
    perturbed.name = "random_structure"

    a, r = diff_edges(graph.adj, adj)
    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="Random Structure",
        n_edges_added=a,
        n_edges_removed=r,
        n_features_perturbed=0,
        budget_used=budget,
        diagnostics={"budget_ratio": budget_ratio, "strategy": "homophily_break"},
    )
