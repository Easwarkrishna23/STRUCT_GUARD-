"""
Attack runner — applies all 6 attacks one-by-one and collects results.

Rules enforced per PROMPT:
  - ONE attack at a time; each starts from the CLEAN graph
  - Poisoning attacks  → retrain model on poisoned graph, then evaluate
  - Evasion attacks    → keep clean model, evaluate on modified graph
  - Attacked graphs saved separately to disk
  - Results evaluated independently

Poisoning vs Evasion distinction:
  Poisoning (Nettack, Meta, Random Structure):
    corrupt training data → retrain GCN → evaluate retrained model on test nodes
  Evasion (Feature Perturbation, Edge Flip, Gradient Attack):
    keep clean trained model → modify graph at test time → evaluate
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import numpy as np
from flax import linen as nn

from datasets.cora_loader import GraphData
from attacks.base import AttackResult, edge_budget
from attacks.calibration import calibrate_attack
from attacks.nettack import nettack
from attacks.meta_attack import meta_attack
from attacks.random_structure import random_structure_attack
from attacks.feature_perturbation import feature_perturbation_attack
from attacks.edge_flip import edge_flip_attack
from attacks.gradient_attack import gradient_attack
from attacks.dice import dice_attack
from utils.config import AttackConfig, ModelConfig


POISONING_ATTACKS = {"nettack", "meta_attack", "random_structure", "dice"}
EVASION_ATTACKS   = {"feature_perturbation", "edge_flip", "gradient_attack"}
ATTACK_NAMES      = list(POISONING_ATTACKS) + list(EVASION_ATTACKS)


@dataclass
class EvaluatedAttack:
    """Combines AttackResult with its evaluation model and metrics."""
    attack_result: AttackResult
    attack_type: str                      # 'poisoning' or 'evasion'
    eval_params: Any                      # params used for evaluation
    retrained: bool                       # True if model was retrained after attack
    diagnostics: dict | None = None


def run_all_attacks(
    graph: GraphData,
    model: nn.Module,
    clean_params: Any,
    attack_cfg: AttackConfig,
    model_cfg: ModelConfig,
    seed: int = 42,
    save_dir: Optional[Path] = None,
) -> dict[str, EvaluatedAttack]:
    """
    Apply all 6 attacks to a clean graph, one-by-one.

    Poisoning attacks trigger a full model retrain on the poisoned graph.
    Evasion attacks use the pre-trained clean model.

    Returns:
        Dict mapping attack_name → EvaluatedAttack (with correct eval params).
    """
    from models.train import train_model
    from models.gcn import create_gcn

    results: dict[str, EvaluatedAttack] = {}

    print(f"\n{'='*60}")
    print(f"PHASE 4 — Adversarial Attacks on {graph.name.upper()}")
    print(f"{'='*60}")

    def _evaluate(
        name: str,
        attack_type: str,
        candidates: list[tuple[str, Any]],
        fallback_attack,
    ) -> EvaluatedAttack:
        if attack_cfg.enforce_target_drop:
            calibrated = calibrate_attack(
                graph=graph,
                model=model,
                clean_params=clean_params,
                attack_type=attack_type,
                candidates=candidates,
                attack_cfg=attack_cfg,
                model_cfg=model_cfg,
                seed=seed,
            )
            return EvaluatedAttack(
                calibrated.attack_result,
                attack_type,
                calibrated.eval_params,
                retrained=attack_type == "poisoning",
                diagnostics=calibrated.diagnostics,
            )

        r = fallback_attack()
        eval_params = (
            _retrain(r.perturbed_graph, model, model_cfg, seed, name)
            if attack_type == "poisoning"
            else clean_params
        )
        return EvaluatedAttack(
            r, attack_type, eval_params, retrained=attack_type == "poisoning"
        )

    structural_ratios = [
        min(0.10, attack_cfg.structural_max_budget_ratio),
        min(0.20, attack_cfg.structural_max_budget_ratio),
        attack_cfg.structural_max_budget_ratio,
    ]

    # ── Poisoning Attack 1: Nettack (margin-based) ───────────────
    print("\n[Poisoning 1/4] Nettack (margin scoring)")
    nettack_candidates = [
        (
            f"perturbations={p}",
            lambda p=p: nettack(
                graph, model, clean_params,
                n_perturbations=p,
                direct_attack=attack_cfg.nettack_direct,
            ),
        )
        for p in sorted({attack_cfg.nettack_n_perturbations, 30, attack_cfg.nettack_max_perturbations})
        if p <= attack_cfg.nettack_max_perturbations
    ]
    results["nettack"] = _evaluate(
        "Nettack", "poisoning", nettack_candidates, nettack_candidates[0][1]
    )
    print(f"  {results['nettack'].attack_result.summary()}")

    # ── Poisoning Attack 2: DICE ──────────────────────────────────
    print("\n[Poisoning 2/4] DICE Attack")
    dice_candidates = [
        (
            f"budget_ratio={ratio:.2f}",
            lambda ratio=ratio: dice_attack(
                graph, model, clean_params, budget_ratio=ratio, seed=seed
            ),
        )
        for ratio in structural_ratios
    ]
    results["dice"] = _evaluate("DICE", "poisoning", dice_candidates, dice_candidates[0][1])
    print(f"  {results['dice'].attack_result.summary()}")

    # ── Poisoning Attack 3: Meta Attack (with inner-loop retrain) ─
    print("\n[Poisoning 3/4] Meta Attack (bilevel approx.)")
    meta_candidates = [
        (
            f"budget_ratio={ratio:.2f}",
            lambda ratio=ratio: meta_attack(
                graph, model, clean_params,
                budget_ratio=ratio,
                n_steps=min(edge_budget(graph.adj, ratio), attack_cfg.meta_epochs),
            ),
        )
        for ratio in structural_ratios
    ]
    results["meta_attack"] = _evaluate(
        "Meta Attack", "poisoning", meta_candidates, meta_candidates[0][1]
    )
    print(f"  {results['meta_attack'].attack_result.summary()}")

    # ── Poisoning Attack 4: Random Structure ──────────────────────
    print("\n[Poisoning 4/4] Random Structure Attack")
    random_candidates = [
        (
            f"budget_ratio={ratio:.2f}",
            lambda ratio=ratio: random_structure_attack(
                graph, budget_ratio=ratio, seed=seed
            ),
        )
        for ratio in structural_ratios
    ]
    results["random_structure"] = _evaluate(
        "Random Structure", "poisoning", random_candidates, random_candidates[0][1]
    )
    print(f"  {results['random_structure'].attack_result.summary()}")

    # ── Evasion Attack 1: Feature Perturbation ────────────────────
    print("\n[Evasion 1/3] Feature Perturbation Attack")
    feature_candidates = [
        (
            f"binary_flip={frac:.2f}",
            lambda frac=frac: feature_perturbation_attack(
                graph, noise_mode="binary_flip", flip_fraction=frac, seed=seed
            ),
        )
        for frac in [0.25, 0.35, attack_cfg.cora_feature_flip_cap]
    ]
    results["feature_perturbation"] = _evaluate(
        "Feature Perturbation", "evasion", feature_candidates, feature_candidates[0][1]
    )
    print(f"  {results['feature_perturbation'].attack_result.summary()}")

    # ── Evasion Attack 2: Edge Flip ───────────────────────────────
    print("\n[Evasion 2/3] Edge Flip Attack")
    edge_candidates = [
        (
            f"budget_ratio={ratio:.2f}",
            lambda ratio=ratio: edge_flip_attack(
                graph, budget_ratio=ratio, strategy="homophily_break", seed=seed
            ),
        )
        for ratio in structural_ratios
    ]
    results["edge_flip"] = _evaluate(
        "Edge Flip", "evasion", edge_candidates, edge_candidates[0][1]
    )
    print(f"  {results['edge_flip'].attack_result.summary()}")

    # ── Evasion Attack 3: Gradient-Based ─────────────────────────
    print("\n[Evasion 3/3] Gradient-Based Attack")
    grad_candidates = [
        (
            f"epsilon={eps:.2f}",
            lambda eps=eps: gradient_attack(
                graph, model, clean_params,
                epsilon=eps,
                steps=attack_cfg.grad_steps,
                attack_mask=np.ones(graph.num_nodes, dtype=bool),
            ),
        )
        for eps in sorted({attack_cfg.grad_epsilon, 0.30, 0.50})
    ]
    results["gradient_attack"] = _evaluate(
        "Gradient Attack", "evasion", grad_candidates, grad_candidates[0][1]
    )
    print(f"  {results['gradient_attack'].attack_result.summary()}")

    # ── Save attacked graphs ──────────────────────────────────────
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, ea in results.items():
            g = ea.attack_result.perturbed_graph
            np.savez(save_dir / f"{graph.name}_{name}.npz",
                     adj=g.adj, features=g.features, labels=g.labels,
                     train_mask=g.train_mask, val_mask=g.val_mask,
                     test_mask=g.test_mask)
        print(f"\n[Runner] Saved {len(results)} attacked graphs → {save_dir}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Retrain helper (poisoning evaluation)
# ──────────────────────────────────────────────────────────────────────────────

def _retrain(poisoned_graph: GraphData, model: nn.Module,
             model_cfg: ModelConfig, seed: int, attack_name: str) -> Any:
    """Retrain model on poisoned graph and return best params."""
    from models.train import train_model
    print(f"  [Retrain after {attack_name}] Training on poisoned graph...")
    result = train_model(model, poisoned_graph, model_cfg,
                         seed=seed, verbose=False)
    print(f"  [Retrain] Best val acc on poisoned graph: {result.best_val_acc:.4f}")
    return result.best_params
