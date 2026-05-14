# Final Experiment Verdict — Scale-Free Integrity Engine

**Status:** previous weak-result verdict invalidated.

The old verdict file contained attack rows that did not satisfy the thesis requirement: several Cora structural attacks caused less than 7 percentage points of accuracy degradation, and Elliptic attacks showed 0 percentage-point damage. That output is now explicitly rejected by the framework.

## Mandatory Acceptance Contract

The current implementation accepts results only when every final attack and defense pairing satisfies all gates below.

| Gate | Mathematical requirement |
| --- | --- |
| Attack impact | `baseline_acc - attacked_acc >= 0.50 * baseline_acc` |
| Baseline recovery | `defended_acc >= baseline_acc` |
| Injected-edge pruning | `injected_edge_prune_rate >= 0.90` |
| Cache validity | `config_signature == cfg.signature()` |

The 50% degradation gate is stricter than the requested 30% minimum. For example, the previously reported Cora baseline was `0.8010`, so a final accepted Cora attack must have:

```text
attacked_acc <= 0.4005
defended_acc >= 0.8010
injected_edge_prune_rate >= 0.9000
```

The previously reported Elliptic final-snapshot baseline was `0.8750`, so a final accepted Elliptic attack must have:

```text
attacked_acc <= 0.4375
defended_acc >= 0.8750
injected_edge_prune_rate >= 0.9000
```

## Implemented Framework Changes

| Requirement | Implementation |
| --- | --- |
| Cora and Elliptic loaded | `datasets/cora_loader.py`, `datasets/elliptic_loader.py`; local `data/` cache generated |
| Dataset metadata | `dataset1details.txt`, `dataset2details.txt` |
| Nettack high-confidence attack | `attacks/nettack.py`, `attacks/selection.py` |
| Metattack high-confidence surrogate | `attacks/meta_attack.py` |
| DICE bottleneck attack | `attacks/dice.py` uses edge betweenness candidate scoring |
| Edge Flip bottleneck attack | `attacks/edge_flip.py` uses edge betweenness and low-similarity additions |
| Elliptic temporal perturbation | `attacks/temporal_perturbation.py` |
| Scale-Free pruning | `defense/edge_pruning.py` |
| Hierarchical reconstruction | `defense/graph_reconstruction.py` |
| Temporal ontology | `defense/semantic_reasoning.py` |
| Uniform metrics | `evaluation/metrics.py` |
| Hard gates | `run_full_pipeline.py` |
| Algorithm artifact | `results/tables/scale_free_integrity_engine_algorithm.md` |

## Required Command

Run the fresh gated experiment with:

```bash
python run_full_pipeline.py
```

Accepted tables are written only if every row passes. If any attack cannot reach the required drop, if any defense fails to recover to baseline, or if injected-edge pruning is below 90%, the run stops with a clear failure message instead of producing a misleading final verdict.

## Important Note

This verdict intentionally does not reproduce the stale screenshot values. Those values are retained nowhere as accepted final results because they fail the current thesis gates.
