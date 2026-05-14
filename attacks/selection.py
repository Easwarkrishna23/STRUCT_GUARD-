"""Target selection helpers shared by adaptive attacks."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from datasets.cora_loader import GraphData


def low_margin_high_degree_targets(
    graph: GraphData,
    model: nn.Module,
    params,
    mask: np.ndarray | None = None,
    max_targets: int = 80,
) -> np.ndarray:
    """Select correctly classified low-margin, high-degree nodes."""
    if mask is None:
        mask = graph.test_mask

    _, logits, _ = model.apply(
        {"params": params},
        jnp.array(graph.features),
        jnp.array(graph.adj_norm),
        training=False,
    )
    logits = np.asarray(logits)
    preds = logits.argmax(axis=1)
    labels = graph.labels
    valid = np.asarray(mask, dtype=bool) & (labels >= 0) & (preds == labels)
    candidates = np.where(valid)[0]
    if candidates.size == 0:
        candidates = np.where(np.asarray(mask, dtype=bool) & (labels >= 0))[0]
    if candidates.size == 0:
        return np.array([], dtype=int)

    true_scores = logits[candidates, labels[candidates]]
    masked = logits[candidates].copy()
    masked[np.arange(candidates.size), labels[candidates]] = -np.inf
    margins = true_scores - masked.max(axis=1)
    degrees = graph.adj.sum(axis=1)[candidates]
    degree_score = degrees / max(float(degrees.max()), 1.0)
    margin_score = 1.0 / (1.0 + np.exp(np.clip(margins, -20.0, 20.0)))
    score = margin_score + 0.25 * degree_score
    order = np.argsort(-score)
    return candidates[order[:max_targets]].astype(int)
