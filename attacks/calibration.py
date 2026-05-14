"""Validation-only attack calibration with honest target-drop diagnostics."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from flax import linen as nn

from attacks.base import AttackResult
from datasets.cora_loader import GraphData
from evaluation.metrics import classification_metrics
from models.train import predict, train_model
from utils.config import AttackConfig, ModelConfig


@dataclass
class CalibratedAttack:
    attack_result: AttackResult
    eval_params: Any
    diagnostics: dict


def calibrate_attack(
    graph: GraphData,
    model: nn.Module,
    clean_params: Any,
    attack_type: str,
    candidates: list[tuple[str, Callable[[], AttackResult]]],
    attack_cfg: AttackConfig,
    model_cfg: ModelConfig,
    seed: int,
) -> CalibratedAttack:
    """
    Select the weakest candidate that reaches the target drop on validation.

    Final test metrics are intentionally not touched here. If no candidate reaches
    the target, return the strongest validation candidate and mark it as failed.
    """
    _, clean_preds, _ = predict(model, clean_params, graph)
    baseline = classification_metrics(graph.labels, clean_preds, mask=graph.val_mask)
    baseline_val = baseline["accuracy"]

    best: tuple[float, AttackResult, Any, str, dict] | None = None
    tried = []

    for label, make_attack in candidates:
        ar = make_attack()
        if attack_type == "poisoning":
            result = train_model(model, ar.perturbed_graph, model_cfg, seed=seed, verbose=False)
            eval_params = result.best_params
            extra = {"best_val_acc_after_retrain": result.best_val_acc}
        else:
            eval_params = clean_params
            extra = {}

        _, preds, _ = predict(model, eval_params, ar.perturbed_graph)
        metrics = classification_metrics(graph.labels, np.asarray(preds), mask=graph.val_mask)
        drop = baseline_val - metrics["accuracy"]
        entry = {
            "candidate": label,
            "validation_accuracy": metrics["accuracy"],
            "validation_drop": drop,
            **extra,
        }
        tried.append(entry)
        print(f"  [Calibration] {label}: val_acc={metrics['accuracy']:.4f}, "
              f"drop={drop:+.4f}")

        if best is None or drop > best[0]:
            best = (drop, ar, eval_params, label, entry)

        if drop >= attack_cfg.required_drop(baseline_val):
            break

    if best is None:
        raise RuntimeError("No attack candidates were provided for calibration.")

    best_drop, best_ar, best_params, best_label, best_entry = best
    required_drop = attack_cfg.required_drop(baseline_val)
    passed = best_drop >= required_drop
    diagnostics = {
        "calibrated_candidate": best_label,
        "validation_baseline_accuracy": baseline_val,
        "validation_drop": best_drop,
        "target_accuracy_drop": required_drop,
        "target_drop_fraction": attack_cfg.target_drop_fraction,
        "calibration_passed": passed,
        "candidates_tried": tried,
        "best_candidate": best_entry,
    }
    best_ar.diagnostics = {**(best_ar.diagnostics or {}), **diagnostics}
    if not passed:
        print(f"  [Calibration] Target not reached within caps: "
              f"best_drop={best_drop:.4f}, target={required_drop:.4f}")
    return CalibratedAttack(best_ar, best_params, diagnostics)
