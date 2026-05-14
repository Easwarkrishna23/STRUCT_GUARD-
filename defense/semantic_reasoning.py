"""Ontology-driven suspicious edge detection for self-healing defense."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datasets.cora_loader import GraphData
from utils.config import DefenseConfig
from utils.graph_utils import compute_cosine_similarity


@dataclass(frozen=True)
class SuspiciousEdge:
    u: int
    v: int
    topic_jaccard: float
    cosine: float
    reason: str


@dataclass(frozen=True)
class SuspiciousNode:
    node: int
    drift_score: float
    reason: str


def build_topic_sets(graph: GraphData, cfg: DefenseConfig) -> list[set[int]]:
    """Build lightweight node topic sets from binary or continuous features."""
    x = np.asarray(graph.features)
    is_binary = np.all((x == 0.0) | (x == 1.0))
    topics: list[set[int]] = []

    if is_binary:
        for row in x:
            topics.append(set(np.where(row > 0)[0].astype(int).tolist()))
        return topics

    mean = x.mean(axis=0)
    std = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
    z = np.abs((x - mean) / std)
    k = min(cfg.topic_top_k, x.shape[1])
    for row in z:
        idx = np.argsort(row)[-k:]
        topics.append(set(idx.astype(int).tolist()))
    return topics


def topic_jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def detect_suspicious_edges(
    graph: GraphData,
    cfg: DefenseConfig,
) -> tuple[set[tuple[int, int]], dict]:
    """Flag existing edges with weak topic agreement or topic mismatch."""
    topics = build_topic_sets(graph, cfg)
    sim = compute_cosine_similarity(graph.features)
    rows, cols = np.where(np.triu(graph.adj, k=1) > 0)
    suspicious: set[tuple[int, int]] = set()
    records: list[SuspiciousEdge] = []

    for u, v in zip(rows, cols):
        j = topic_jaccard(topics[u], topics[v])
        c = float(sim[u, v])
        mismatch = j == 0.0 and c < cfg.centrality_cosine_threshold
        if j < cfg.topic_jaccard_threshold or mismatch:
            reason = "topic_jaccard" if j < cfg.topic_jaccard_threshold else "topic_mismatch"
            edge = (int(min(u, v)), int(max(u, v)))
            suspicious.add(edge)
            records.append(SuspiciousEdge(edge[0], edge[1], j, c, reason))

    stats = {
        "suspicious_edges": len(suspicious),
        "topic_jaccard_threshold": cfg.topic_jaccard_threshold,
        "examples": [r.__dict__ for r in records[:10]],
    }
    print(f"  [Semantic Reasoning] flagged {len(suspicious)}/{len(rows)} edges")
    return suspicious, stats


def detect_temporal_drift(
    graph: GraphData,
    previous_graph: GraphData | None,
    cfg: DefenseConfig,
) -> tuple[set[int], dict]:
    """
    Flag Elliptic nodes whose feature vectors violate temporal consistency.

    Transaction identities are snapshot-local in this loader, so drift is
    measured against the robust feature distribution of the previous snapshot.
    """
    if previous_graph is None:
        return set(), {
            "suspicious_nodes": 0,
            "reason": "no_previous_snapshot",
        }

    x = np.asarray(graph.features, dtype=np.float64)
    ref = np.asarray(previous_graph.features, dtype=np.float64)
    median = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - median), axis=0)
    scale = np.where(mad == 0.0, np.std(ref, axis=0), 1.4826 * mad)
    scale = np.where(scale == 0.0, 1.0, scale)
    z = np.abs((x - median) / scale)
    drift_score = np.sqrt((z * z).mean(axis=1))
    threshold = max(
        cfg.temporal_drift_z_threshold,
        float(np.quantile(drift_score, 0.95)),
    )
    nodes = set(np.where(drift_score >= threshold)[0].astype(int).tolist())
    records = [
        SuspiciousNode(int(i), float(drift_score[i]), "temporal_drift").__dict__
        for i in np.argsort(-drift_score)[:10]
    ]
    stats = {
        "suspicious_nodes": len(nodes),
        "temporal_drift_threshold": threshold,
        "previous_snapshot": getattr(previous_graph, "name", None),
        "examples": records,
    }
    print(f"  [Temporal Ontology] flagged {len(nodes)}/{graph.num_nodes} nodes")
    return nodes, stats
