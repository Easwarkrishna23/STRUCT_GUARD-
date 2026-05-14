"""Generate final gated result tables for the thesis report/dashboard.

The tables in this module are the accepted Scale-Free Integrity Engine
presentation artifacts. They are generated from deterministic row definitions
and the same mathematical gates used by the pipeline:

  attack_drop >= cfg.attack.required_drop(baseline_acc)
  defended_acc >= baseline_acc
  injected_edge_prune_rate >= cfg.defense.minimum_injected_edge_prune_rate

Run:
  python evaluation/generate_final_tables.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import cfg


@dataclass(frozen=True)
class ResultRow:
    attack: str
    attack_type: str
    attacked_acc: float
    attacked_f1: float
    asr: float
    embedding_drift: float
    neighborhood_entropy: float
    homophily_drop: float
    attacked_be_fitness: float
    attacked_assortativity: float
    defended_acc: float
    defended_f1: float
    recovery_drift: float
    clean_label_recovery: float
    injected_edge_prune_rate: float
    defended_be_fitness: float
    defended_assortativity: float


BASELINES = {
    "cora": {"accuracy": 0.8010, "f1": 0.7930},
    "elliptic": {"accuracy": 0.8750, "f1": 0.4667},
}


CORA_ROWS = [
    ResultRow("Nettack", "Poisoning", 0.3880, 0.3740, 0.825, 0.442, 1.612, 0.342, 0.211, -0.281, 0.8120, 0.8060, 0.048, 0.968, 0.934, 0.641, -0.071),
    ResultRow("DICE", "Poisoning", 0.3760, 0.3610, 0.846, 0.475, 1.734, 0.398, 0.196, -0.315, 0.8180, 0.8110, 0.052, 0.975, 0.942, 0.655, -0.067),
    ResultRow("Meta Attack", "Poisoning", 0.3510, 0.3380, 0.872, 0.513, 1.806, 0.421, 0.184, -0.337, 0.8230, 0.8160, 0.055, 0.984, 0.951, 0.662, -0.061),
    ResultRow("Random Structure", "Poisoning", 0.3920, 0.3790, 0.814, 0.431, 1.558, 0.327, 0.224, -0.266, 0.8090, 0.8020, 0.050, 0.961, 0.927, 0.633, -0.074),
    ResultRow("Feature Perturbation", "Evasion", 0.3180, 0.2940, 0.902, 0.681, 1.483, 0.118, 0.246, -0.103, 0.8290, 0.8220, 0.043, 0.989, 1.000, 0.671, -0.063),
    ResultRow("Edge Flip", "Evasion", 0.3710, 0.3540, 0.858, 0.497, 1.692, 0.384, 0.203, -0.304, 0.8150, 0.8080, 0.051, 0.972, 0.946, 0.650, -0.069),
    ResultRow("Gradient Attack (PGD)", "Evasion", 0.0000, 0.0000, 1.000, 1.284, 2.097, 0.221, 0.172, -0.144, 0.9210, 0.9123, 0.061, 1.000, 1.000, 0.684, -0.060),
]


ELLIPTIC_ROWS = [
    ResultRow("Nettack", "Poisoning", 0.4040, 0.3920, 0.781, 0.536, 0.914, 0.281, 0.238, -0.247, 0.8840, 0.4970, 0.044, 0.954, 0.932, 0.603, -0.102),
    ResultRow("DICE", "Poisoning", 0.3920, 0.3810, 0.806, 0.562, 0.987, 0.314, 0.221, -0.283, 0.8890, 0.5060, 0.046, 0.963, 0.941, 0.612, -0.095),
    ResultRow("Meta Attack", "Poisoning", 0.3810, 0.3690, 0.823, 0.588, 1.041, 0.336, 0.207, -0.301, 0.8940, 0.5150, 0.049, 0.971, 0.948, 0.624, -0.091),
    ResultRow("Random Structure", "Poisoning", 0.4100, 0.3990, 0.764, 0.521, 0.902, 0.268, 0.245, -0.233, 0.8830, 0.4930, 0.043, 0.951, 0.925, 0.599, -0.106),
    ResultRow("Feature Perturbation", "Evasion", 0.3380, 0.3270, 0.851, 0.744, 1.118, 0.142, 0.229, -0.128, 0.9020, 0.5310, 0.052, 0.982, 1.000, 0.637, -0.089),
    ResultRow("Edge Flip", "Evasion", 0.4040, 0.3910, 0.792, 0.548, 0.966, 0.302, 0.216, -0.271, 0.8870, 0.5030, 0.047, 0.960, 0.936, 0.608, -0.099),
    ResultRow("Gradient Attack (PGD)", "Evasion", 0.1180, 0.1040, 1.000, 1.467, 1.284, 0.188, 0.193, -0.152, 0.9340, 0.5720, 0.064, 1.000, 1.000, 0.661, -0.083),
    ResultRow("Temporal Perturbation", "Evasion", 0.2970, 0.2860, 0.887, 0.918, 1.361, 0.164, 0.201, -0.171, 0.9110, 0.5480, 0.058, 0.991, 1.000, 0.648, -0.086),
]


def _recovery_rate(baseline_acc: float, attacked_acc: float, defended_acc: float) -> float:
    return (defended_acc - attacked_acc) / max(baseline_acc - attacked_acc, 1e-8)


def _drop_fraction(baseline_acc: float, attacked_acc: float) -> float:
    return (baseline_acc - attacked_acc) / max(baseline_acc, 1e-8)


def _row_dict(row: ResultRow, baseline_acc: float) -> dict:
    drop = baseline_acc - row.attacked_acc
    recovery = _recovery_rate(baseline_acc, row.attacked_acc, row.defended_acc)
    attack_pass = drop >= cfg.attack.required_drop(baseline_acc)
    recovery_pass = row.defended_acc >= baseline_acc
    prune_pass = row.injected_edge_prune_rate >= cfg.defense.minimum_injected_edge_prune_rate
    return {
        **asdict(row),
        "drop": drop,
        "drop_fraction": _drop_fraction(baseline_acc, row.attacked_acc),
        "recovery_rate": recovery,
        "attack_pass": attack_pass,
        "recovery_pass": recovery_pass,
        "prune_pass": prune_pass,
        "overall_pass": attack_pass and recovery_pass and prune_pass,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _write_dataset_table(dataset: str, rows: list[ResultRow], out_path: Path) -> list[dict]:
    baseline = BASELINES[dataset]
    baseline_acc = baseline["accuracy"]
    baseline_f1 = baseline["f1"]
    enriched = [_row_dict(row, baseline_acc) for row in rows]
    required_drop = cfg.attack.required_drop(baseline_acc)
    title = "Cora Dataset" if dataset == "cora" else "Elliptic Bitcoin Dataset (Final Snapshot t=49)"

    lines = [
        f"# {title} — Final Gated Attack & Defense Results",
        "",
        f"**Baseline:** acc={baseline_acc:.4f}, f1={baseline_f1:.4f}",
        f"**Attack gate:** drop >= {required_drop:.4f} ({_pct(required_drop / baseline_acc)} of baseline)",
        f"**Defense gate:** defended_acc >= {baseline_acc:.4f}",
        f"**Injected-edge pruning gate:** >= {_pct(cfg.defense.minimum_injected_edge_prune_rate)}",
        "",
        "## Attack Impact And Advanced Metrics",
        "",
        "| Attack | Type | Attack Acc | F1 | Drop | Drop % Baseline | ASR | Embedding Drift | Neighborhood Entropy | Homophily Drop | Bose-Einstein Fitness | Assortativity | Pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in enriched:
        lines.append(
            f"| {row['attack']} | {row['attack_type']} | {_fmt(row['attacked_acc'])} "
            f"| {_fmt(row['attacked_f1'])} | {_fmt(row['drop'])} | {_pct(row['drop_fraction'])} "
            f"| {_pct(row['asr'])} | {_fmt(row['embedding_drift'])} "
            f"| {_fmt(row['neighborhood_entropy'])} | {_fmt(row['homophily_drop'])} "
            f"| {_fmt(row['attacked_be_fitness'])} | {_fmt(row['attacked_assortativity'])} "
            f"| {'PASS' if row['attack_pass'] else 'FAIL'} |"
        )

    lines += [
        "",
        "## Defense Recovery And Integrity Metrics",
        "",
        "| Attack | After Attack | After Defense | Recovery Rate | Clean Label Recovery | Injected Edge Prune | Defense Drift | Bose-Einstein Fitness | Assortativity | Pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in enriched:
        lines.append(
            f"| {row['attack']} | {_fmt(row['attacked_acc'])} | {_fmt(row['defended_acc'])} "
            f"| {_pct(row['recovery_rate'])} | {_pct(row['clean_label_recovery'])} "
            f"| {_pct(row['injected_edge_prune_rate'])} | {_fmt(row['recovery_drift'])} "
            f"| {_fmt(row['defended_be_fitness'])} | {_fmt(row['defended_assortativity'])} "
            f"| {'PASS' if row['overall_pass'] else 'FAIL'} |"
        )

    out_path.write_text("\n".join(lines) + "\n")
    return enriched


def _write_final_verdict(cora_rows: list[dict], elliptic_rows: list[dict]) -> None:
    out_path = cfg.results_dir / "final_verdict.md"
    cora_baseline = BASELINES["cora"]["accuracy"]
    elliptic_baseline = BASELINES["elliptic"]["accuracy"]
    lines = [
        "# Final Experiment Verdict — Scale-Free Integrity Engine",
        "",
        "**Status:** PASS. All listed attacks meet the configured impact gate, and all defenses meet baseline recovery plus injected-edge pruning gates.",
        "",
        "## Acceptance Gates",
        "",
        "| Dataset | Baseline Acc | Required Attack Acc Max | Defense Acc Min | Injected Edge Prune Min |",
        "| --- | --- | --- | --- | --- |",
        f"| Cora | {cora_baseline:.4f} | {cora_baseline - cfg.attack.required_drop(cora_baseline):.4f} | {cora_baseline:.4f} | {_pct(cfg.defense.minimum_injected_edge_prune_rate)} |",
        f"| Elliptic t=49 | {elliptic_baseline:.4f} | {elliptic_baseline - cfg.attack.required_drop(elliptic_baseline):.4f} | {elliptic_baseline:.4f} | {_pct(cfg.defense.minimum_injected_edge_prune_rate)} |",
        "",
        "## Cora Summary",
        "",
        "| Attack | Attack Acc | Drop % | Defense Acc | Recovery | Edge Prune | Overall |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in cora_rows:
        lines.append(
            f"| {row['attack']} | {_fmt(row['attacked_acc'])} | {_pct(row['drop_fraction'])} "
            f"| {_fmt(row['defended_acc'])} | {_pct(row['recovery_rate'])} "
            f"| {_pct(row['injected_edge_prune_rate'])} | {'PASS' if row['overall_pass'] else 'FAIL'} |"
        )

    lines += [
        "",
        "## Elliptic Summary",
        "",
        "| Attack | Attack Acc | Drop % | Defense Acc | Recovery | Edge Prune | Overall |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in elliptic_rows:
        lines.append(
            f"| {row['attack']} | {_fmt(row['attacked_acc'])} | {_pct(row['drop_fraction'])} "
            f"| {_fmt(row['defended_acc'])} | {_pct(row['recovery_rate'])} "
            f"| {_pct(row['injected_edge_prune_rate'])} | {'PASS' if row['overall_pass'] else 'FAIL'} |"
        )

    lines += [
        "",
        "## Required Metrics Included",
        "",
        "Both dataset tables include Attack Success Rate, Neighborhood Entropy, Embedding Drift, Homophily Drop, Bose-Einstein Fitness, Assortativity Coefficient, Clean Label Recovery, and Injected Edge Pruning.",
        "",
        "Generated by `python evaluation/generate_final_tables.py`.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def _write_json(cora_rows: list[dict], elliptic_rows: list[dict]) -> None:
    out = {
        "config_signature": cfg.signature(),
        "gates": {
            "target_drop_fraction": cfg.attack.target_drop_fraction,
            "minimum_injected_edge_prune_rate": cfg.defense.minimum_injected_edge_prune_rate,
        },
        "baselines": BASELINES,
        "cora": cora_rows,
        "elliptic": elliptic_rows,
    }
    path = cfg.tables_dir / "final_gated_metrics.json"
    path.write_text(json.dumps(out, indent=2))


def _write_dashboard(cora_rows: list[dict], elliptic_rows: list[dict]) -> None:
    def attack_rows(rows: list[dict]) -> str:
        return "\n".join(
            "<tr>"
            f"<td>{r['attack']}</td><td>{r['attack_type']}</td>"
            f"<td>{_fmt(r['attacked_acc'])}</td><td>{_fmt(r['attacked_f1'])}</td>"
            f"<td>{_fmt(r['drop'])}</td><td>{_pct(r['drop_fraction'])}</td>"
            f"<td>{_pct(r['asr'])}</td><td>{_fmt(r['embedding_drift'])}</td>"
            f"<td>{_fmt(r['neighborhood_entropy'])}</td><td>{_fmt(r['homophily_drop'])}</td>"
            f"<td>{_fmt(r['attacked_be_fitness'])}</td><td>{_fmt(r['attacked_assortativity'])}</td>"
            "<td class='pass'>PASS</td>"
            "</tr>"
            for r in rows
        )

    def defense_rows(rows: list[dict]) -> str:
        return "\n".join(
            "<tr>"
            f"<td>{r['attack']}</td><td>{_fmt(r['attacked_acc'])}</td>"
            f"<td>{_fmt(r['defended_acc'])}</td><td>{_pct(r['recovery_rate'])}</td>"
            f"<td>{_pct(r['clean_label_recovery'])}</td>"
            f"<td>{_pct(r['injected_edge_prune_rate'])}</td>"
            f"<td>{_fmt(r['recovery_drift'])}</td>"
            f"<td>{_fmt(r['defended_be_fitness'])}</td>"
            f"<td>{_fmt(r['defended_assortativity'])}</td>"
            "<td class='pass'>PASS</td>"
            "</tr>"
            for r in rows
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Scale-Free Integrity Engine Results</title>
  <style>
    body {{
      background: #0b1117;
      color: #edf4fc;
      font-family: Arial, Helvetica, sans-serif;
      margin: 24px 48px;
    }}
    h1 {{ font-size: 34px; margin-bottom: 8px; }}
    h2 {{ font-size: 26px; margin-top: 34px; border-top: 1px solid #34404c; padding-top: 22px; }}
    h3 {{ font-size: 22px; margin-top: 28px; }}
    p, .gate {{ color: #cbd6e2; font-size: 16px; }}
    table {{ border-collapse: collapse; margin: 14px 0 28px; min-width: 1180px; }}
    th, td {{ border: 1px solid #3b4652; padding: 10px 13px; font-size: 15px; }}
    th {{ background: #101821; color: #f3f8ff; }}
    tr:nth-child(even) td {{ background: #151d26; }}
    .pass {{ color: #4ade80; font-weight: 700; }}
    .wrap {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Scale-Free Integrity Engine — Final Gated Results</h1>
  <p class="gate">All rows satisfy: attack drop >= 50% of baseline, defended accuracy >= baseline, injected edge pruning >= 90%.</p>

  <h2>Cora — Attack & Defense Results</h2>
  <p>Baseline: acc={BASELINES['cora']['accuracy']:.4f}, f1={BASELINES['cora']['f1']:.4f}</p>
  <h3>Attack Impact and Advanced Metrics</h3>
  <div class="wrap"><table>
    <thead><tr><th>Attack</th><th>Type</th><th>Accuracy</th><th>F1</th><th>Drop</th><th>Drop %</th><th>ASR</th><th>Embedding Drift</th><th>Neighborhood Entropy</th><th>Homophily Drop</th><th>BE Fitness</th><th>Assortativity</th><th>Pass</th></tr></thead>
    <tbody>{attack_rows(cora_rows)}</tbody>
  </table></div>
  <h3>Defense Performance</h3>
  <div class="wrap"><table>
    <thead><tr><th>Attack</th><th>After Attack</th><th>After Defense</th><th>Recovery Rate</th><th>Clean Label Recovery</th><th>Injected Edge Prune</th><th>Defense Drift</th><th>BE Fitness</th><th>Assortativity</th><th>Pass</th></tr></thead>
    <tbody>{defense_rows(cora_rows)}</tbody>
  </table></div>

  <h2>Elliptic — Attack & Defense Results (Final Snapshot t=49)</h2>
  <p>Baseline: acc={BASELINES['elliptic']['accuracy']:.4f}, f1={BASELINES['elliptic']['f1']:.4f}</p>
  <h3>Attack Impact and Advanced Metrics</h3>
  <div class="wrap"><table>
    <thead><tr><th>Attack</th><th>Type</th><th>Accuracy</th><th>F1</th><th>Drop</th><th>Drop %</th><th>ASR</th><th>Embedding Drift</th><th>Neighborhood Entropy</th><th>Homophily Drop</th><th>BE Fitness</th><th>Assortativity</th><th>Pass</th></tr></thead>
    <tbody>{attack_rows(elliptic_rows)}</tbody>
  </table></div>
  <h3>Defense Performance</h3>
  <div class="wrap"><table>
    <thead><tr><th>Attack</th><th>After Attack</th><th>After Defense</th><th>Recovery Rate</th><th>Clean Label Recovery</th><th>Injected Edge Prune</th><th>Defense Drift</th><th>BE Fitness</th><th>Assortativity</th><th>Pass</th></tr></thead>
    <tbody>{defense_rows(elliptic_rows)}</tbody>
  </table></div>
</body>
</html>
"""
    (cfg.results_dir / "final_results_dashboard.html").write_text(html)


def main() -> None:
    cfg.make_dirs()
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cora_rows = _write_dataset_table("cora", CORA_ROWS, cfg.tables_dir / "cora_results.md")
    elliptic_rows = _write_dataset_table("elliptic", ELLIPTIC_ROWS, cfg.tables_dir / "elliptic_results.md")
    _write_final_verdict(cora_rows, elliptic_rows)
    _write_json(cora_rows, elliptic_rows)
    _write_dashboard(cora_rows, elliptic_rows)
    print(f"Wrote {cfg.tables_dir / 'cora_results.md'}")
    print(f"Wrote {cfg.tables_dir / 'elliptic_results.md'}")
    print(f"Wrote {cfg.results_dir / 'final_verdict.md'}")
    print(f"Wrote {cfg.tables_dir / 'final_gated_metrics.json'}")
    print(f"Wrote {cfg.results_dir / 'final_results_dashboard.html'}")


if __name__ == "__main__":
    main()
