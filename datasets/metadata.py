"""Dataset metadata writers for Cora and Elliptic Bitcoin experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets.cora_loader import GraphData
from datasets.elliptic_loader import EllipticData
from evaluation.metrics import (
    assortativity_coefficient,
    graph_homophily,
    power_law_exponent,
)


def write_dataset_details(
    cora: GraphData,
    elliptic: EllipticData,
    output_dir: str | Path = ".",
) -> tuple[Path, Path]:
    """Write dataset1details.txt and dataset2details.txt after data loading."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    d1 = output_dir / "dataset1details.txt"
    d2 = output_dir / "dataset2details.txt"
    d1.write_text(_describe_cora(cora))
    d2.write_text(_describe_elliptic(elliptic))
    return d1, d2


def _describe_cora(graph: GraphData) -> str:
    stats = graph.stats()
    degrees = graph.adj.sum(axis=1)
    class_dist = _class_distribution(graph.labels)
    lines = [
        "Dataset 1 Details: Cora Citation Network",
        "=" * 48,
        f"Name: {graph.name}",
        f"Nodes: {graph.num_nodes}",
        f"Edges: {stats['num_edges']}",
        f"Feature dimensions: {graph.num_features}",
        f"Classes: {graph.num_classes}",
        f"Train/Val/Test nodes: {int(graph.train_mask.sum())}/"
        f"{int(graph.val_mask.sum())}/{int(graph.test_mask.sum())}",
        f"Class distribution: {class_dist}",
        f"Baseline homophily: {graph_homophily(graph.adj, graph.labels):.6f}",
        f"Degree assortativity: {assortativity_coefficient(graph.adj):.6f}",
        f"Degree exponent gamma: {power_law_exponent(degrees):.6f}",
        f"Average degree: {float(degrees.mean()):.6f}",
        f"Median degree: {float(np.median(degrees)):.6f}",
        f"Minimum degree: {int(degrees.min())}",
        f"Maximum degree: {int(degrees.max())}",
        f"Degree percentiles p10/p25/p75/p90: {_degree_percentiles(degrees)}",
        "Temporal structure: static single-snapshot citation graph",
        "",
    ]
    return "\n".join(lines)


def _describe_elliptic(data: EllipticData) -> str:
    final = data.final_snapshot()
    final_degrees = final.adj.sum(axis=1)
    node_counts = np.array([s.num_nodes for s in data.snapshots])
    edge_counts = np.array([s.stats()["num_edges"] for s in data.snapshots])
    feature_dims = sorted({s.num_features for s in data.snapshots})
    classes = sorted({int(c) for s in data.snapshots for c in np.unique(s.labels) if c >= 0})
    lines = [
        "Dataset 2 Details: Elliptic Bitcoin Transaction Network",
        "=" * 58,
        f"Timesteps: {data.num_timesteps}",
        f"Feature dimensions per snapshot: {feature_dims}",
        f"Known classes after remapping: {classes} (0=licit, 1=illicit, -1=unknown)",
        f"Nodes per timestep min/mean/max: {int(node_counts.min())}/"
        f"{float(node_counts.mean()):.2f}/{int(node_counts.max())}",
        f"Edges per timestep min/mean/max: {int(edge_counts.min())}/"
        f"{float(edge_counts.mean()):.2f}/{int(edge_counts.max())}",
        f"Average labeled ratio: {float(np.mean(data.labeled_ratios)):.6f}",
        f"Average illicit ratio among labeled nodes: {float(np.mean(data.illicit_ratios)):.6f}",
        "",
        "Final Snapshot (t=49)",
        "-" * 22,
        f"Nodes: {final.num_nodes}",
        f"Edges: {final.stats()['num_edges']}",
        f"Feature dimensions: {final.num_features}",
        f"Class distribution: {_class_distribution(final.labels)}",
        f"Train/Val/Test nodes: {int(final.train_mask.sum())}/"
        f"{int(final.val_mask.sum())}/{int(final.test_mask.sum())}",
        f"Baseline homophily: {graph_homophily(final.adj, final.labels):.6f}",
        f"Degree assortativity: {assortativity_coefficient(final.adj):.6f}",
        f"Degree exponent gamma: {power_law_exponent(final_degrees):.6f}",
        f"Average degree: {float(final_degrees.mean()):.6f}",
        f"Median degree: {float(np.median(final_degrees)):.6f}",
        f"Minimum degree: {int(final_degrees.min())}",
        f"Maximum degree: {int(final_degrees.max())}",
        f"Degree percentiles p10/p25/p75/p90: {_degree_percentiles(final_degrees)}",
        "",
        "Temporal Structure Summary",
        "-" * 26,
    ]
    for idx, snap in enumerate(data.snapshots, start=1):
        labels = _class_distribution(snap.labels)
        lines.append(
            f"t={idx:02d}: nodes={snap.num_nodes}, edges={snap.stats()['num_edges']}, "
            f"labeled={data.labeled_ratios[idx-1]:.4f}, "
            f"illicit={data.illicit_ratios[idx-1]:.4f}, classes={labels}"
        )
    lines.append("")
    return "\n".join(lines)


def _class_distribution(labels: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _degree_percentiles(degrees: np.ndarray) -> str:
    p = np.percentile(degrees, [10, 25, 75, 90])
    return ", ".join(f"{float(v):.2f}" for v in p)
