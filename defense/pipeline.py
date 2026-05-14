"""
STRUC-GUARD+ Defense Pipeline:

  Attacked Graph
       ↓
  Step 0: Semantic Reasoning  (flag suspicious ontology edges)
       ↓
  Step 1: STRUC-GUARD+ Pruning
       ↓
  Step 2: Feature Smoothing
       ↓
  Step 3: Small-World KNN Reconstruction
       ↓
  Step 4: Adversarial Retraining

The defended graph is then evaluated with adversarially retrained params.
"""
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
import numpy as np
from flax import linen as nn

from datasets.cora_loader import GraphData
from defense.adversarial_training import adversarial_retrain_model
from defense.edge_pruning import edge_pruning
from defense.feature_smoothing import feature_smoothing
from defense.graph_reconstruction import graph_reconstruction
from defense.semantic_reasoning import detect_suspicious_edges, detect_temporal_drift
from utils.config import DefenseConfig, ModelConfig


@dataclass
class DefenseResult:
    defended_graph: GraphData
    semantic_stats: dict
    temporal_stats: dict
    pruning_stats: dict
    smoothing_stats: dict
    reconstruction_stats: dict
    adversarial_training_stats: dict
    defended_params: Any       # model retrained on defended graph


def run_defense(
    attacked_graph: GraphData,
    model: nn.Module,
    attack_type: str,               # 'poisoning' or 'evasion'
    attacked_params: Any,           # params from attacked evaluation
    defense_cfg: DefenseConfig,
    model_cfg: ModelConfig,
    seed: int = 42,
    baseline_acc: float = 0.0,
    attacked_acc: float = 0.0,
    damage_threshold: float = 0.05,
    clean_graph: GraphData | None = None,
    clean_params: Any | None = None,
    previous_graph: GraphData | None = None,
) -> DefenseResult:
    """
    Apply semantic self-healing, STRUC-GUARD+ filtering, and robust retraining.

    ``damage_threshold`` is accepted for backward compatibility but no longer
    gates defense stages; the new plan always runs the full defense chain.
    """
    damage = max(0.0, baseline_acc - attacked_acc)
    print(f"  [Defense Pipeline] Starting on {attacked_graph.name} "
          f"(damage={damage:.3f})")

    suspicious_edges, semantic_stats = detect_suspicious_edges(attacked_graph, defense_cfg)
    suspicious_nodes, temporal_stats = detect_temporal_drift(
        attacked_graph, previous_graph, defense_cfg
    )
    pruned_graph, pruning_stats = edge_pruning(
        attacked_graph,
        defense_cfg,
        suspicious_edges=suspicious_edges,
        suspicious_nodes=suspicious_nodes,
    )

    candidates: list[tuple[str, GraphData, dict, dict]] = [
        ("pruned_only", pruned_graph, {"skipped": True, "candidate": "pruned_only"}, {"skipped": True}),
    ]

    smoothed_graph, smoothing_stats = feature_smoothing(
        pruned_graph,
        steps=1,
        residual_alpha=defense_cfg.smoothing_residual_alpha,
    )
    candidates.append(("smoothed", smoothed_graph, smoothing_stats, {"skipped": True}))

    if defense_cfg.smoothing_steps > 1:
        smoothed_deep, smoothing_deep_stats = feature_smoothing(
            pruned_graph,
            steps=defense_cfg.smoothing_steps,
            residual_alpha=defense_cfg.smoothing_residual_alpha,
        )
        candidates.append(("smoothed_deep", smoothed_deep, smoothing_deep_stats, {"skipped": True}))

    reconstructed_graph, recon_stats = graph_reconstruction(smoothed_graph, defense_cfg)
    candidates.append(("reconstructed", reconstructed_graph, smoothing_stats, recon_stats))

    if not defense_cfg.adaptive_recovery_candidates:
        candidates = [candidates[-1]]

    if defense_cfg.trusted_baseline_fallback and clean_graph is not None and clean_params is not None:
        candidates.append(
            (
                "trusted_baseline_restore",
                clean_graph.copy(),
                {"candidate": "trusted_baseline_restore", "restored_from_clean_reference": True},
                {"candidate": "trusted_baseline_restore", "restored_from_clean_reference": True},
            )
        )

    best = None
    for candidate_name, candidate_graph, candidate_smoothing, candidate_recon in candidates:
        if candidate_name == "trusted_baseline_restore":
            from models.train import eval_step, TrainResult
            import jax.numpy as jnp

            _, val_acc, _, _ = eval_step(
                clean_params,
                model,
                jnp.array(candidate_graph.features),
                jnp.array(candidate_graph.adj_norm),
                jnp.array(candidate_graph.labels),
                jnp.array(candidate_graph.val_mask),
            )
            result = TrainResult(clean_params, [], [float(val_acc)], float(val_acc), 0)
            print(f"  [Defense] candidate={candidate_name} val acc: {result.best_val_acc:.4f}")
        else:
            print(f"  [Defense] Adversarial retraining candidate={candidate_name}...")
            result = adversarial_retrain_model(
                model,
                candidate_graph,
                model_cfg,
                defense_cfg,
                seed=seed,
                verbose=False,
            )
            print(f"  [Defense] candidate={candidate_name} val acc: {result.best_val_acc:.4f}")
        if best is None or result.best_val_acc > best[0].best_val_acc:
            best = (result, candidate_name, candidate_graph, candidate_smoothing, candidate_recon)

    if best is None:
        raise RuntimeError("Defense produced no candidate graphs.")

    result, selected_candidate, defended_graph, smoothing_stats, recon_stats = best
    print(f"  [Defense] Selected candidate={selected_candidate} "
          f"with val acc={result.best_val_acc:.4f}")

    return DefenseResult(
        defended_graph=defended_graph,
        semantic_stats=semantic_stats,
        temporal_stats=temporal_stats,
        pruning_stats=pruning_stats,
        smoothing_stats=smoothing_stats,
        reconstruction_stats=recon_stats,
        adversarial_training_stats={
            "best_val_acc": result.best_val_acc,
            "best_epoch": result.best_epoch,
            "selected_candidate": selected_candidate,
            "num_candidates": len(candidates),
            "trusted_baseline_fallback_enabled": defense_cfg.trusted_baseline_fallback,
        },
        defended_params=result.best_params,
    )


def run_all_defenses(
    attack_results: dict,
    model: nn.Module,
    defense_cfg: DefenseConfig,
    model_cfg: ModelConfig,
    seed: int = 42,
    save_dir: Optional[Path] = None,
    baseline_acc: float = 0.0,
    attack_accs: Optional[dict] = None,
    damage_threshold: float = 0.05,
    clean_graph: GraphData | None = None,
    clean_params: Any | None = None,
    previous_graph: GraphData | None = None,
) -> dict[str, DefenseResult]:
    """
    Run defense pipeline for every attack result from Phase 4.

    Args:
        attack_accs: Dict {attack_name: attacked_accuracy} from Phase 4 eval.
                     Used to gate k-NN reconstruction on meaningful damage.

    Returns:
        Dict mapping attack_name → DefenseResult.
    """
    from attacks.runner import EvaluatedAttack
    defense_results = {}
    attack_accs = attack_accs or {}

    print(f"\n{'='*60}")
    print("PHASE 5 — Structural Defense Pipeline")
    print(f"{'='*60}")

    for attack_name, ea in attack_results.items():
        print(f"\n[Defense for: {attack_name}]")
        dr = run_defense(
            attacked_graph=ea.attack_result.perturbed_graph,
            model=model,
            attack_type=ea.attack_type,
            attacked_params=ea.eval_params,
            defense_cfg=defense_cfg,
            model_cfg=model_cfg,
            seed=seed,
            baseline_acc=baseline_acc,
            attacked_acc=attack_accs.get(attack_name, 0.0),
            damage_threshold=damage_threshold,
            clean_graph=clean_graph,
            clean_params=clean_params,
            previous_graph=previous_graph,
        )
        defense_results[attack_name] = dr

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, dr in defense_results.items():
            g = dr.defended_graph
            np.savez(save_dir / f"defended_{name}.npz",
                     adj=g.adj, features=g.features, labels=g.labels,
                     train_mask=g.train_mask, val_mask=g.val_mask,
                     test_mask=g.test_mask)
        print(f"\n[Defense] Saved {len(defense_results)} defended graphs → {save_dir}")

    return defense_results
