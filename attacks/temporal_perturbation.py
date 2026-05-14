"""Elliptic temporal consistency attack."""
from __future__ import annotations

import numpy as np

from attacks.base import AttackResult
from datasets.cora_loader import GraphData


def temporal_perturbation_attack(
    graph: GraphData,
    previous_graph: GraphData | None = None,
    epsilon: float = 1.0,
    clip_quantiles: tuple[float, float] | None = None,
    attack_mask: np.ndarray | None = None,
    seed: int = 42,
) -> AttackResult:
    """
    Create severe feature discontinuity relative to the previous timestep.

    Elliptic transaction nodes are not persistent identities across all
    snapshots, so this attack uses the previous snapshot's robust feature
    distribution as the temporal consistency reference.
    """
    rng = np.random.default_rng(seed)
    feats = graph.features.copy()
    mask = np.ones(graph.num_nodes, dtype=bool) if attack_mask is None else np.asarray(attack_mask, dtype=bool)
    ref = previous_graph.features if previous_graph is not None else graph.features
    ref_mean = ref.mean(axis=0)
    ref_std = np.where(ref.std(axis=0) == 0.0, 1.0, ref.std(axis=0))
    direction = np.sign(feats[mask] - ref_mean)
    random_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=direction.shape)
    direction = np.where(direction == 0.0, random_sign, direction)
    feats[mask] = feats[mask] + epsilon * direction * ref_std

    if clip_quantiles is not None:
        lo = np.quantile(graph.features, clip_quantiles[0], axis=0)
        hi = np.quantile(graph.features, clip_quantiles[1], axis=0)
        feats[mask] = np.clip(feats[mask], lo, hi)

    perturbed = graph.copy().update_features(feats)
    perturbed.name = "temporal_perturbation"
    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="Temporal Perturbation",
        n_edges_added=0,
        n_edges_removed=0,
        n_features_perturbed=int(mask.sum()),
        budget_used=int(mask.sum()),
        target_nodes=np.where(mask)[0],
        diagnostics={
            "epsilon": epsilon,
            "clip_quantiles": clip_quantiles,
            "previous_snapshot": getattr(previous_graph, "name", None),
            "strategy": "temporal_distribution_discontinuity",
        },
    )
