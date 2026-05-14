"""Evaluation metrics for classification, robustness, and graph damage."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _masked(y: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return y
    return y[np.asarray(mask, dtype=bool)]


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """Compute accuracy, macro precision/recall/F1 on optionally masked labels."""
    y_true = _masked(np.asarray(y_true), mask)
    y_pred = _masked(np.asarray(y_pred), mask)
    valid = y_true >= 0
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if y_true.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def accuracy_drop(baseline_acc: float, attacked_acc: float) -> float:
    """Accuracy drop in points: baseline minus attacked accuracy."""
    return float(baseline_acc - attacked_acc)


def recovery_rate(
    baseline_acc: float,
    attacked_acc: float,
    defended_acc: float,
    min_damage: float = 1e-8,
) -> Optional[float]:
    """Fraction of attack damage recovered by defense."""
    denom = baseline_acc - attacked_acc
    if abs(denom) <= min_damage:
        return None
    return float((defended_acc - attacked_acc) / denom)


def attack_success_rate(
    target_nodes: np.ndarray,
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_attacked: np.ndarray,
) -> float:
    """Fraction of originally correct target nodes that become misclassified."""
    target_nodes = np.asarray(target_nodes, dtype=int)
    if target_nodes.size == 0:
        return 0.0

    y_true = np.asarray(y_true)
    clean = np.asarray(y_pred_clean)
    attacked = np.asarray(y_pred_attacked)
    valid = y_true[target_nodes] >= 0
    correct_before = clean[target_nodes] == y_true[target_nodes]
    eligible = valid & correct_before
    if eligible.sum() == 0:
        return 0.0
    nodes = target_nodes[eligible]
    return float((attacked[nodes] != y_true[nodes]).mean())


def embedding_drift(
    clean_embeddings: np.ndarray,
    other_embeddings: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> float:
    """Mean normalized L2 drift between clean and compared embeddings."""
    clean = _masked(np.asarray(clean_embeddings, dtype=np.float64), mask)
    other = _masked(np.asarray(other_embeddings, dtype=np.float64), mask)
    if clean.size == 0:
        return 0.0
    diff = np.linalg.norm(other - clean, axis=1)
    base = np.linalg.norm(clean, axis=1)
    return float(np.mean(diff / np.maximum(base, eps)))


def graph_homophily(
    adj: np.ndarray,
    labels: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Fraction of known-label edges whose endpoints have the same label."""
    adj = np.asarray(adj)
    labels = np.asarray(labels)
    rows, cols = np.where(np.triu(adj > 0, k=1))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        keep = m[rows] | m[cols]
        rows, cols = rows[keep], cols[keep]
    known = (labels[rows] >= 0) & (labels[cols] >= 0)
    rows, cols = rows[known], cols[known]
    if rows.size == 0:
        return 0.0
    return float((labels[rows] == labels[cols]).mean())


def homophily_drop(
    clean_adj: np.ndarray,
    attacked_adj: np.ndarray,
    labels: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Drop in edge-label homophily from clean to attacked graph."""
    return float(
        graph_homophily(clean_adj, labels, mask)
        - graph_homophily(attacked_adj, labels, mask)
    )


def assortativity_coefficient(adj: np.ndarray) -> float:
    """Degree assortativity coefficient, with NaN mapped to 0 for reporting."""
    import networkx as nx

    G = nx.from_numpy_array((np.asarray(adj) > 0).astype(np.float32))
    if G.number_of_edges() == 0:
        return 0.0
    value = nx.degree_assortativity_coefficient(G)
    return 0.0 if not np.isfinite(value) else float(value)


def power_law_exponent(adj_or_degrees: np.ndarray, k_min: int = 1) -> float:
    """Estimate scale-free degree exponent gamma by a discrete MLE surrogate."""
    arr = np.asarray(adj_or_degrees)
    degrees = arr.sum(axis=1) if arr.ndim == 2 else arr
    degrees = np.asarray(degrees, dtype=np.float64)
    tail = degrees[degrees >= max(k_min, 1)]
    if tail.size < 2:
        return 0.0
    denom = np.log(np.maximum(tail, k_min) / max(k_min - 0.5, 0.5)).sum()
    if denom <= 0:
        return 0.0
    return float(1.0 + tail.size / denom)


def bose_einstein_fitness(
    adj: np.ndarray,
    features: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Mean edge fitness combining semantic similarity and degree consistency.

    Higher values indicate that edges follow a natural scale-free attachment
    pattern; low values indicate unlikely growth that a similarity-only defense
    can miss.
    """
    adj = (np.asarray(adj) > 0).astype(np.float32)
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    if rows.size == 0:
        return 0.0
    degrees = adj.sum(axis=1).astype(np.float64)
    deg_norm = np.log1p(degrees)
    temp = float(np.std(deg_norm) + eps)

    x = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = x / np.maximum(norms, eps)
    cosine = (x_norm[rows] * x_norm[cols]).sum(axis=1)
    cosine = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)

    degree_gap = np.abs(deg_norm[rows] - deg_norm[cols])
    degree_fit = np.exp(-degree_gap / temp)
    return float(np.mean(cosine * degree_fit))


def clean_label_recovery(
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_attacked: np.ndarray,
    y_pred_defended: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Fraction of clean-correct, attack-broken nodes restored by defense."""
    y_true = _masked(np.asarray(y_true), mask)
    clean = _masked(np.asarray(y_pred_clean), mask)
    attacked = _masked(np.asarray(y_pred_attacked), mask)
    defended = _masked(np.asarray(y_pred_defended), mask)
    valid = y_true >= 0
    damaged = valid & (clean == y_true) & (attacked != y_true)
    if damaged.sum() == 0:
        return 0.0
    return float((defended[damaged] == y_true[damaged]).mean())


def injected_edge_prune_rate(
    clean_adj: np.ndarray,
    attacked_adj: np.ndarray,
    defended_adj: np.ndarray,
) -> float:
    """Fraction of attack-injected edges removed by the defense."""
    clean = np.asarray(clean_adj) > 0
    attacked = np.asarray(attacked_adj) > 0
    defended = np.asarray(defended_adj) > 0
    injected = np.triu(attacked & ~clean, k=1)
    total = int(injected.sum())
    if total == 0:
        return 1.0
    removed = int((injected & ~np.triu(defended, k=1)).sum())
    return float(removed / total)


def neighborhood_entropy(
    adj: np.ndarray,
    labels: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> float:
    """Mean entropy of each node's known-label neighbor distribution."""
    adj = np.asarray(adj)
    labels = np.asarray(labels)
    nodes = np.arange(adj.shape[0]) if mask is None else np.where(mask)[0]
    entropies: list[float] = []

    for node in nodes:
        nbrs = np.where(adj[node] > 0)[0]
        nbr_labels = labels[nbrs]
        nbr_labels = nbr_labels[nbr_labels >= 0]
        if nbr_labels.size == 0:
            continue
        _, counts = np.unique(nbr_labels, return_counts=True)
        probs = counts.astype(np.float64) / counts.sum()
        entropy = float(-(probs * np.log2(np.maximum(probs, eps))).sum())
        entropies.append(max(0.0, entropy))

    return float(np.mean(entropies)) if entropies else 0.0


def robustness_summary(baseline: dict, attacked: dict, defended: dict) -> dict:
    """Build a compact summary dict from three classification metric dicts."""
    rr = recovery_rate(
        baseline["accuracy"], attacked["accuracy"], defended["accuracy"]
    )
    return {
        "baseline_acc": baseline["accuracy"],
        "attacked_acc": attacked["accuracy"],
        "defended_acc": defended["accuracy"],
        "accuracy_drop": accuracy_drop(baseline["accuracy"], attacked["accuracy"]),
        "recovery_rate": rr,
        "baseline_f1": baseline["f1"],
        "attacked_f1": attacked["f1"],
        "defended_f1": defended["f1"],
    }


def format_metrics_table(
    results: dict[str, dict],
    metric_keys: list[str] | None = None,
) -> str:
    """Format a metrics mapping as a Markdown table."""
    metric_keys = metric_keys or ["accuracy", "precision", "recall", "f1"]
    header = "| Attack | " + " | ".join(k.capitalize() for k in metric_keys) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(metric_keys)) + " |"
    rows = []
    for attack_name, m in results.items():
        vals = " | ".join(f"{float(m.get(k, 0.0)):.4f}" for k in metric_keys)
        rows.append(f"| {attack_name} | {vals} |")
    return "\n".join([header, sep] + rows)


def format_defense_table(results: dict[str, dict]) -> str:
    """Format defense performance as a Markdown table."""
    header = "| Attack | After Attack | After Defense | Recovery Rate |"
    sep = "| --- | --- | --- | --- |"
    rows = []
    for attack_name, m in results.items():
        rr = m.get("recovery_rate", None)
        rr_str = f"{rr:.1%}" if rr is not None else "N/A"
        rows.append(
            f"| {attack_name} | {m.get('attacked_acc', 0.0):.4f} "
            f"| {m.get('defended_acc', 0.0):.4f} | {rr_str} |"
        )
    return "\n".join([header, sep] + rows)
