"""Generate one-row-per-attack attack/defense comparison tables.

The earlier "unified" tables used two rows per attack:
one row for "After Attack" and one row for "After Defense". That is useful
for metric alignment, but it is tiring in a presentation. This script produces
the cleaner viva/presentation layout requested by the user:

    Attack | Type | Accuracy (After Attack, After Defense) | ...

It reads the accepted final metrics from results/tables/final_gated_metrics.json
and writes compact CSV, Markdown, HTML, and PNG tables into results 2/.
"""
from __future__ import annotations

import csv
import html
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "tables" / "final_gated_metrics.json"
OUT_DIR = ROOT / "results 2"


CORE_GROUPS = [
    ("Accuracy", ("After Attack", "After Defense"), ("Accuracy After Attack", "Accuracy After Defense")),
    ("F1", ("After Attack", "After Defense"), ("F1 After Attack", "F1 After Defense")),
    (
        "Drop from Baseline",
        ("After Attack", "After Defense"),
        ("Drop From Baseline After Attack", "Drop From Baseline After Defense"),
    ),
    ("ASR / Residual ASR", ("After Attack", "After Defense"), ("ASR After Attack", "Residual ASR After Defense")),
    (
        "Embedding Drift",
        ("After Attack", "After Defense"),
        ("Embedding Drift After Attack", "Embedding Drift After Defense"),
    ),
    (
        "Neighborhood Entropy",
        ("After Attack", "After Defense"),
        ("Neighborhood Entropy After Attack", "Neighborhood Entropy After Defense"),
    ),
]

FULL_FLAT_HEADERS = [
    "Attack",
    "Type",
    "Accuracy After Attack",
    "Accuracy After Defense",
    "F1 After Attack",
    "F1 After Defense",
    "Drop From Baseline After Attack",
    "Drop From Baseline After Defense",
    "Drop % Baseline After Attack",
    "Drop % Baseline After Defense",
    "ASR After Attack",
    "Residual ASR After Defense",
    "Embedding Drift After Attack",
    "Embedding Drift After Defense",
    "Neighborhood Entropy After Attack",
    "Neighborhood Entropy After Defense",
    "Homophily Drop After Attack",
    "Residual Homophily Gap After Defense",
    "Bose-Einstein Fitness After Attack",
    "Bose-Einstein Fitness After Defense",
    "Assortativity After Attack",
    "Assortativity After Defense",
    "Recovery Rate",
    "Clean Label Recovery",
    "Injected Edge Prune",
    "Pass",
]


def _fmt_float(value: float) -> str:
    return f"{value:.4f}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _load_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"{METRICS_PATH} not found. Run evaluation/generate_final_tables.py first."
        )
    return json.loads(METRICS_PATH.read_text())


def _compact_values(row: dict, baseline_acc: float) -> dict[str, str]:
    attack_drop = baseline_acc - row["attacked_acc"]
    defense_gap = baseline_acc - row["defended_acc"]
    attack_drop_frac = attack_drop / max(baseline_acc, 1e-8)
    defense_gap_frac = defense_gap / max(baseline_acc, 1e-8)
    residual_error_fraction = 1.0 - row["clean_label_recovery"]
    residual_asr = row["asr"] * residual_error_fraction
    residual_entropy = row["neighborhood_entropy"] * residual_error_fraction
    residual_homophily_gap = row["homophily_drop"] * (1.0 - row["injected_edge_prune_rate"])

    return {
        "Attack": row["attack"],
        "Type": row["attack_type"],
        "Accuracy After Attack": _fmt_float(row["attacked_acc"]),
        "Accuracy After Defense": _fmt_float(row["defended_acc"]),
        "F1 After Attack": _fmt_float(row["attacked_f1"]),
        "F1 After Defense": _fmt_float(row["defended_f1"]),
        "Drop From Baseline After Attack": _fmt_float(attack_drop),
        "Drop From Baseline After Defense": _fmt_float(defense_gap),
        "Drop % Baseline After Attack": _fmt_pct(attack_drop_frac),
        "Drop % Baseline After Defense": _fmt_pct(defense_gap_frac),
        "ASR After Attack": _fmt_pct(row["asr"]),
        "Residual ASR After Defense": _fmt_pct(residual_asr),
        "Embedding Drift After Attack": _fmt_float(row["embedding_drift"]),
        "Embedding Drift After Defense": _fmt_float(row["recovery_drift"]),
        "Neighborhood Entropy After Attack": _fmt_float(row["neighborhood_entropy"]),
        "Neighborhood Entropy After Defense": _fmt_float(residual_entropy),
        "Homophily Drop After Attack": _fmt_float(row["homophily_drop"]),
        "Residual Homophily Gap After Defense": _fmt_float(residual_homophily_gap),
        "Bose-Einstein Fitness After Attack": _fmt_float(row["attacked_be_fitness"]),
        "Bose-Einstein Fitness After Defense": _fmt_float(row["defended_be_fitness"]),
        "Assortativity After Attack": _fmt_float(row["attacked_assortativity"]),
        "Assortativity After Defense": _fmt_float(row["defended_assortativity"]),
        "Recovery Rate": _fmt_pct(row["recovery_rate"]),
        "Clean Label Recovery": _fmt_pct(row["clean_label_recovery"]),
        "Injected Edge Prune": _fmt_pct(row["injected_edge_prune_rate"]),
        "Pass": "PASS" if row["overall_pass"] else "FAIL",
    }


def _dataset_rows(metrics: dict, dataset: str) -> list[dict[str, str]]:
    baseline_acc = metrics["baselines"][dataset]["accuracy"]
    return [_compact_values(row, baseline_acc) for row in metrics[dataset]]


def _write_csv(path: Path, rows: list[dict[str, str]], include_dataset: bool = False) -> None:
    headers = (["Dataset"] if include_dataset else []) + FULL_FLAT_HEADERS
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]], include_dataset: bool = False) -> None:
    headers = (["Dataset"] if include_dataset else []) + FULL_FLAT_HEADERS
    lines = [
        "# Compact Unified Attack-Defense Metrics",
        "",
        "Each attack is represented once. Paired columns compare the post-attack and post-defense states directly.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n")


def _write_html(path: Path, title: str, rows: list[dict[str, str]], include_dataset: bool = False) -> None:
    left_headers = ["Dataset", "Attack", "Type"] if include_dataset else ["Attack", "Type"]
    top_cells = "".join(f'<th rowspan="2">{html.escape(h)}</th>' for h in left_headers)
    top_cells += "".join(f'<th colspan="2">{html.escape(group)}</th>' for group, _, _ in CORE_GROUPS)
    top_cells += '<th colspan="4">Recovery Integrity</th>'

    sub_cells = "".join(
        f"<th>{html.escape(sub)}</th>"
        for _, subs, _ in CORE_GROUPS
        for sub in subs
    )
    sub_cells += "<th>Recovery Rate</th><th>Clean Label Recovery</th><th>Injected Edge Prune</th><th>Pass</th>"

    body_rows = []
    for row in rows:
        left = [row[h] for h in left_headers]
        paired = []
        for _, _, keys in CORE_GROUPS:
            paired.append(row[keys[0]])
            paired.append(row[keys[1]])
        integrity = [row["Recovery Rate"], row["Clean Label Recovery"], row["Injected Edge Prune"], row["Pass"]]
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in left + paired + integrity)
        body_rows.append(f"<tr>{cells}</tr>")

    path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{
  font-family: Arial, Helvetica, sans-serif;
  margin: 24px;
  background: #ffffff;
  color: #111827;
}}
h1 {{
  font-size: 26px;
  margin-bottom: 16px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;
  font-size: 14px;
}}
th, td {{
  border: 1px solid #111827;
  padding: 8px 9px;
  text-align: center;
  vertical-align: middle;
  word-wrap: break-word;
}}
th {{
  font-weight: 700;
  background: #f8fafc;
}}
td:first-child, td:nth-child(2), td:nth-child(3) {{
  text-align: left;
  font-weight: 600;
}}
tr:nth-child(even) td {{
  background: #f9fafb;
}}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<table>
  <thead>
    <tr>{top_cells}</tr>
    <tr>{sub_cells}</tr>
  </thead>
  <tbody>
    {"".join(body_rows)}
  </tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )


def _cell_text(value: str, width: int = 15) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def _draw_grouped_png(path: Path, title: str, rows: list[dict[str, str]]) -> None:
    # Core presentation image: keep the most visible metrics only.
    columns: list[tuple[str, str | None, str]] = [
        ("Attack", None, "Attack"),
        ("Type", None, "Type"),
    ]
    for group, subs, keys in CORE_GROUPS:
        columns.append((group, subs[0], keys[0]))
        columns.append((group, subs[1], keys[1]))

    widths = [1.9, 1.45] + [1.35] * (len(columns) - 2)
    header_h = 0.62
    subheader_h = 0.78
    row_h = 0.62
    total_w = sum(widths)
    total_h = header_h + subheader_h + row_h * len(rows)

    fig_w = max(16, total_w * 0.85)
    fig_h = max(5, total_h * 0.68)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h + 0.8)
    ax.axis("off")
    ax.text(total_w / 2, total_h + 0.55, title, ha="center", va="center", fontsize=16, fontweight="bold")

    y_top = total_h
    x = 0.0
    # Attack and Type span both header rows.
    for idx, label in enumerate(["Attack", "Type"]):
        w = widths[idx]
        ax.add_patch(Rectangle((x, y_top - header_h - subheader_h), w, header_h + subheader_h,
                               facecolor="#f8fafc", edgecolor="#111827", linewidth=0.8))
        ax.text(x + w / 2, y_top - (header_h + subheader_h) / 2, label,
                ha="center", va="center", fontsize=9.5, fontweight="bold")
        x += w

    col_idx = 2
    for group, subs, _ in CORE_GROUPS:
        group_w = widths[col_idx] + widths[col_idx + 1]
        ax.add_patch(Rectangle((x, y_top - header_h), group_w, header_h,
                               facecolor="#f8fafc", edgecolor="#111827", linewidth=0.8))
        ax.text(x + group_w / 2, y_top - header_h / 2, _cell_text(group, 18),
                ha="center", va="center", fontsize=9.5, fontweight="bold")
        for sub_idx, sub in enumerate(subs):
            w = widths[col_idx + sub_idx]
            sx = x + sum(widths[col_idx:col_idx + sub_idx])
            ax.add_patch(Rectangle((sx, y_top - header_h - subheader_h), w, subheader_h,
                                   facecolor="#ffffff", edgecolor="#111827", linewidth=0.8))
            ax.text(sx + w / 2, y_top - header_h - subheader_h / 2, _cell_text(sub, 12),
                    ha="center", va="center", fontsize=8.7, fontweight="bold")
        x += group_w
        col_idx += 2

    y = y_top - header_h - subheader_h
    for r, row in enumerate(rows):
        y -= row_h
        x = 0.0
        fill = "#ffffff" if r % 2 == 0 else "#f9fafb"
        for c, (_, _, key) in enumerate(columns):
            w = widths[c]
            ax.add_patch(Rectangle((x, y), w, row_h, facecolor=fill, edgecolor="#111827", linewidth=0.55))
            wrap_width = 13 if key == "Attack" else 10
            ax.text(x + w / 2, y + row_h / 2, _cell_text(row[key], wrap_width),
                    ha="center", va="center", fontsize=8.2)
            x += w

    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_dataset_outputs(metrics: dict, dataset: str) -> list[dict[str, str]]:
    rows = _dataset_rows(metrics, dataset)
    stem = f"{dataset}_unified_attack_defense_metrics"
    title = f"{dataset.upper()} Compact Attack-Defense Metrics"
    _write_csv(OUT_DIR / f"{stem}.csv", rows)
    _write_markdown(OUT_DIR / f"{stem}.md", rows)
    _write_html(OUT_DIR / f"{stem}.html", title, rows)
    _draw_grouped_png(OUT_DIR / f"{stem}.png", title, rows)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics()

    combined: list[dict[str, str]] = []
    for dataset in ["cora", "elliptic"]:
        rows = _write_dataset_outputs(metrics, dataset)
        for row in rows:
            combined.append({"Dataset": dataset, **row})

    _write_csv(OUT_DIR / "combined_unified_attack_defense_metrics.csv", combined, include_dataset=True)
    _write_markdown(OUT_DIR / "combined_unified_attack_defense_metrics.md", combined, include_dataset=True)
    _write_html(
        OUT_DIR / "combined_unified_attack_defense_metrics.html",
        "Combined Compact Attack-Defense Metrics",
        combined,
        include_dataset=True,
    )

    notes = [
        "# Metric Alignment Notes",
        "",
        "The compact unified tables now use one row per attack. Each metric has paired",
        "`After Attack` and `After Defense` columns, matching the requested syntax-style layout.",
        "",
        "Defense-side ASR is reported as residual ASR:",
        "",
        "`Residual ASR = ASR * (1 - Clean Label Recovery)`",
        "",
        "Defense-side entropy is a residual diagnostic derived from attack entropy and remaining unrecovered attacked nodes:",
        "",
        "`Defense Entropy = Attack Entropy * (1 - Clean Label Recovery)`",
        "",
        "Negative defense-side drop means the defended accuracy is above the clean baseline.",
    ]
    (OUT_DIR / "metric_alignment_notes.md").write_text("\n".join(notes) + "\n")

    print(f"Wrote compact unified tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
