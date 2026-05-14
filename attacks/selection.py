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


def high_confidence_high_degree_targets(
    graph: GraphData,
    model: nn.Module,
    params,
    mask: np.ndarray | None = None,
    max_targets: int = 80,
    confidence_quantile: float = 0.70,
) -> np.ndarray:
    """Select correctly classified high-confidence, high-degree target nodes."""
    if mask is None:
        mask = graph.test_mask

    _, logits, probs = model.apply(
        {"params": params},
        jnp.array(graph.features),
        jnp.array(graph.adj_norm),
        training=False,
    )
    logits = np.asarray(logits)
    probs = np.asarray(probs)
    preds = logits.argmax(axis=1)
    labels = graph.labels
    valid = np.asarray(mask, dtype=bool) & (labels >= 0) & (preds == labels)
    candidates = np.where(valid)[0]
    if candidates.size == 0:
        return low_margin_high_degree_targets(
            graph, model, params, mask=mask, max_targets=max_targets
        )

    confidence = probs[candidates, preds[candidates]]
    threshold = float(np.quantile(confidence, confidence_quantile))
    high_conf = candidates[confidence >= threshold]
    if high_conf.size == 0:
        high_conf = candidates

    degrees = graph.adj.sum(axis=1)[high_conf]
    high_confidence = probs[high_conf, preds[high_conf]]
    degree_score = degrees / max(float(degrees.max()), 1.0)
    confidence_score = high_confidence / max(float(high_confidence.max()), 1e-8)
    score = confidence_score + 0.35 * degree_score
    order = np.argsort(-score)
    return high_conf[order[:max_targets]].astype(int)
