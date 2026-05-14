"""
Feature Perturbation Attack — evasion attack at test time.

Adds bounded noise to node features. Applies to ALL nodes (untargeted)
or only test nodes depending on `targeted_test_only`.

Two noise modes:
  'uniform' — add uniform noise in [-ε, +ε]  (default)
  'gaussian' — add Gaussian noise with std=ε
"""
import numpy as np
from datasets.cora_loader import GraphData
from attacks.base import AttackResult


def feature_perturbation_attack(
    graph: GraphData,
    epsilon: float = 0.1,
    noise_mode: str = "uniform",
    test_only: bool = False,
    flip_fraction: float = 0.5,
    clip_quantiles: tuple[float, float] | None = None,
    seed: int = 42,
) -> AttackResult:
    """
    Add bounded noise to node features (test-time evasion).

    Args:
        graph:      Clean GraphData.
        epsilon:    Noise magnitude bound.
        noise_mode: 'uniform' or 'gaussian'.
        test_only:  If True, only perturb test node features.
        seed:       RNG seed.

    Returns:
        AttackResult with perturbed features; adjacency unchanged.
    """
    rng   = np.random.default_rng(seed)
    feats = graph.features.copy()
    n, f  = feats.shape

    node_mask = graph.test_mask if test_only else np.ones(n, dtype=bool)
    n_perturbed = int(node_mask.sum())

    if noise_mode == "binary_flip":
        changed = 0
        for node in np.where(node_mask)[0]:
            active = np.where(feats[node] > 0)[0]
            inactive = np.where(feats[node] <= 0)[0]
            if active.size == 0:
                continue
            n_flip = max(1, int(active.size * flip_fraction))
            n_flip = min(n_flip, active.size)
            off = rng.choice(active, size=n_flip, replace=False)
            feats[node, off] = 0.0
            changed += n_flip
            # Add the same number of false positives when space exists.
            if inactive.size:
                on = rng.choice(inactive, size=min(n_flip, inactive.size), replace=False)
                feats[node, on] = 1.0
                changed += len(on)
        print(f"  [Feature Perturbation] binary_flip={flip_fraction:.0%}, "
              f"perturbed {n_perturbed} nodes, flipped {changed} entries")
    elif noise_mode == "uniform":
        noise = rng.uniform(-epsilon, epsilon, size=(n_perturbed, f)).astype(np.float32)
        feats[node_mask] = feats[node_mask] + noise
    elif noise_mode == "gaussian":
        noise = rng.normal(0, epsilon, size=(n_perturbed, f)).astype(np.float32)
        feats[node_mask] = feats[node_mask] + noise
    elif noise_mode == "centroid_shift":
        labels = graph.labels
        valid = labels >= 0
        class_means = {}
        for cls in np.unique(labels[valid]):
            class_means[int(cls)] = feats[labels == cls].mean(axis=0)
        global_mean = feats[valid].mean(axis=0) if valid.any() else feats.mean(axis=0)
        for node in np.where(node_mask)[0]:
            cls = int(labels[node]) if labels[node] >= 0 else None
            current = class_means.get(cls, feats[node])
            alternatives = [v for k, v in class_means.items() if k != cls]
            target = alternatives[int(rng.integers(0, len(alternatives)))] if alternatives else global_mean
            direction = target - current
            norm = np.linalg.norm(direction)
            if norm > 0:
                feats[node] = feats[node] + epsilon * direction / norm
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")

    if noise_mode != "binary_flip":
        if clip_quantiles is None:
            lo, hi = 0.0, 1.0
        else:
            lo = np.quantile(graph.features, clip_quantiles[0], axis=0)
            hi = np.quantile(graph.features, clip_quantiles[1], axis=0)
        feats[node_mask] = np.clip(feats[node_mask], lo, hi)

        print(f"  [Feature Perturbation] ε={epsilon}, mode={noise_mode}, "
              f"perturbed {n_perturbed} nodes")

    perturbed = graph.copy()
    perturbed = perturbed.update_features(feats)
    perturbed.name = "feature_perturbation"

    return AttackResult(
        perturbed_graph=perturbed,
        attack_name="Feature Perturbation",
        n_edges_added=0,
        n_edges_removed=0,
        n_features_perturbed=n_perturbed,
        budget_used=n_perturbed,
        target_nodes=np.where(node_mask)[0],
        diagnostics={
            "epsilon": epsilon,
            "noise_mode": noise_mode,
            "flip_fraction": flip_fraction,
        },
    )
