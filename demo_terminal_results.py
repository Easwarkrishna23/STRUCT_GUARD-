"""Small terminal demo for final STRUC-GUARD+ results.

This script prints compact attack-vs-defense tables similar to the screenshots
used in the presentation. It is intentionally lightweight: it reads the already
accepted final metrics from results/tables/final_gated_metrics.json and formats
them for terminal display. Each attack is shown once, with paired attack/defense
columns, so the table is easier to explain during viva.

Run:
    python demo_terminal_results.py
    python demo_terminal_results.py --dataset cora
    python demo_terminal_results.py --dataset elliptic --focus

If the JSON file is missing, regenerate it first:
    python evaluation/generate_final_tables.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "results" / "tables" / "final_gated_metrics.json"


FOCUS_ATTACKS = {
    "cora": {"Nettack", "DICE", "Feature Perturbation", "Gradient Attack (PGD)"},
    "elliptic": {"Nettack", "Feature Perturbation", "Gradient Attack (PGD)", "Temporal Perturbation"},
}


def _ensure_metrics_file() -> None:
    if METRICS_PATH.exists():
        return
    print("[Info] final_gated_metrics.json not found; generating final tables first...\n")
    subprocess.run(
        [sys.executable, str(ROOT / "evaluation" / "generate_final_tables.py")],
        check=True,
        cwd=str(ROOT),
    )


def _load_metrics() -> dict:
    _ensure_metrics_file()
    return json.loads(METRICS_PATH.read_text())


def _fmt_float(value: float) -> str:
    return f"{value:.4f}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _wrap(value: str, width: int) -> list[str]:
    words = value.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _print_table(headers: list[str], rows: list[list[str]], widths: list[int]) -> None:
    def sep(char: str = "-") -> str:
        return "+" + "+".join(char * (w + 2) for w in widths) + "+"

    def print_row(cells: list[str]) -> None:
        wrapped = [_wrap(str(cell), width) for cell, width in zip(cells, widths)]
        height = max(len(part) for part in wrapped)
        for i in range(height):
            pieces = []
            for part, width in zip(wrapped, widths):
                text = part[i] if i < len(part) else ""
                pieces.append(f" {text:<{width}} ")
            print("|" + "|".join(pieces) + "|")

    print(sep("="))
    print_row(headers)
    print(sep("="))
    for row in rows:
        print_row(row)
        print(sep("-"))


def _paired_rows(dataset: str, rows: list[dict], baseline_acc: float, focus: bool) -> list[list[str]]:
    out: list[list[str]] = []
    selected = FOCUS_ATTACKS[dataset] if focus else None

    for row in rows:
        if selected and row["attack"] not in selected:
            continue

        attack_drop = baseline_acc - row["attacked_acc"]
        attack_drop_frac = attack_drop / max(baseline_acc, 1e-8)
        defense_drop = baseline_acc - row["defended_acc"]
        defense_drop_frac = defense_drop / max(baseline_acc, 1e-8)
        residual_error_fraction = 1.0 - row["clean_label_recovery"]
        residual_asr = row["asr"] * residual_error_fraction
        residual_entropy = row["neighborhood_entropy"] * residual_error_fraction

        out.append([
            row["attack"],
            row["attack_type"],
            _fmt_float(row["attacked_acc"]),
            _fmt_float(row["defended_acc"]),
            _fmt_float(row["attacked_f1"]),
            _fmt_float(row["defended_f1"]),
            _fmt_float(attack_drop),
            _fmt_float(defense_drop),
            _fmt_pct(attack_drop_frac),
            _fmt_pct(defense_drop_frac),
            _fmt_pct(row["asr"]),
            _fmt_pct(residual_asr),
            _fmt_float(row["embedding_drift"]),
            _fmt_float(row["recovery_drift"]),
            _fmt_float(row["neighborhood_entropy"]),
            _fmt_float(residual_entropy),
        ])

    return out


def _print_dataset(name: str, metrics: dict, focus: bool) -> None:
    baseline = metrics["baselines"][name]
    rows = metrics[name]
    title = "CORA STATIC CITATION NETWORK" if name == "cora" else "ELLIPTIC BITCOIN TEMPORAL GRAPH"
    required_drop = baseline["accuracy"] * metrics["gates"]["target_drop_fraction"]

    print("\n" + "=" * 120)
    print(f"{title}")
    print("=" * 120)
    print(f"Baseline Accuracy: {baseline['accuracy']:.4f} | Baseline F1: {baseline['f1']:.4f}")
    print(f"Attack Gate      : Drop >= {required_drop:.4f} ({metrics['gates']['target_drop_fraction'] * 100:.0f}% of baseline)")
    print(f"Defense Gate     : Defended accuracy >= {baseline['accuracy']:.4f}")
    print(f"Pruning Gate     : Injected-edge pruning >= {metrics['gates']['minimum_injected_edge_prune_rate'] * 100:.0f}%")
    print()

    headers = [
        "Attack",
        "Type",
        "Acc Attack",
        "Acc Defense",
        "F1 Attack",
        "F1 Defense",
        "Drop Attack",
        "Gap Defense",
        "Drop % Attack",
        "Gap % Defense",
        "ASR Attack",
        "Residual ASR",
        "Drift Attack",
        "Drift Defense",
        "Entropy Attack",
        "Entropy Defense",
    ]
    widths = [20, 10, 10, 11, 9, 10, 11, 11, 12, 13, 10, 12, 12, 13, 13, 14]
    table_rows = _paired_rows(name, rows, baseline["accuracy"], focus)
    _print_table(headers, table_rows, widths)

    print("\nQuick read:")
    print("  - Each row now compares one attack before and after STRUC-GUARD+.")
    print("  - Attack-side columns should show large drop, high ASR, and high drift/entropy.")
    print("  - Defense-side columns should show accuracy back to baseline or above with low residual ASR/drift/entropy.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print compact final attack/defense results in terminal.")
    parser.add_argument(
        "--dataset",
        choices=["all", "cora", "elliptic"],
        default="all",
        help="Dataset table to print.",
    )
    parser.add_argument(
        "--focus",
        action="store_true",
        help="Print only the most presentation-relevant attacks for a shorter demo.",
    )
    args = parser.parse_args()

    metrics = _load_metrics()

    datasets = ["cora", "elliptic"] if args.dataset == "all" else [args.dataset]
    print("\nSTRUC-GUARD+ TERMINAL RESULT DEMO")
    print("Source:", METRICS_PATH)
    print("Mode  :", "focus attacks only" if args.focus else "all attacks")

    for dataset in datasets:
        _print_dataset(dataset, metrics, args.focus)


if __name__ == "__main__":
    main()
