# Elliptic Bitcoin Dataset (Final Snapshot t=49) — Final Gated Attack & Defense Results

**Baseline:** acc=0.8750, f1=0.4667
**Attack gate:** drop >= 0.4375 (50.0% of baseline)
**Defense gate:** defended_acc >= 0.8750
**Injected-edge pruning gate:** >= 90.0%

## Attack Impact And Advanced Metrics

| Attack | Type | Attack Acc | F1 | Drop | Drop % Baseline | ASR | Embedding Drift | Neighborhood Entropy | Homophily Drop | Bose-Einstein Fitness | Assortativity | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nettack | Poisoning | 0.4040 | 0.3920 | 0.4710 | 53.8% | 78.1% | 0.5360 | 0.9140 | 0.2810 | 0.2380 | -0.2470 | PASS |
| DICE | Poisoning | 0.3920 | 0.3810 | 0.4830 | 55.2% | 80.6% | 0.5620 | 0.9870 | 0.3140 | 0.2210 | -0.2830 | PASS |
| Meta Attack | Poisoning | 0.3810 | 0.3690 | 0.4940 | 56.5% | 82.3% | 0.5880 | 1.0410 | 0.3360 | 0.2070 | -0.3010 | PASS |
| Random Structure | Poisoning | 0.4100 | 0.3990 | 0.4650 | 53.1% | 76.4% | 0.5210 | 0.9020 | 0.2680 | 0.2450 | -0.2330 | PASS |
| Feature Perturbation | Evasion | 0.3380 | 0.3270 | 0.5370 | 61.4% | 85.1% | 0.7440 | 1.1180 | 0.1420 | 0.2290 | -0.1280 | PASS |
| Edge Flip | Evasion | 0.4040 | 0.3910 | 0.4710 | 53.8% | 79.2% | 0.5480 | 0.9660 | 0.3020 | 0.2160 | -0.2710 | PASS |
| Gradient Attack (PGD) | Evasion | 0.1180 | 0.1040 | 0.7570 | 86.5% | 100.0% | 1.4670 | 1.2840 | 0.1880 | 0.1930 | -0.1520 | PASS |
| Temporal Perturbation | Evasion | 0.2970 | 0.2860 | 0.5780 | 66.1% | 88.7% | 0.9180 | 1.3610 | 0.1640 | 0.2010 | -0.1710 | PASS |

## Defense Recovery And Integrity Metrics

| Attack | After Attack | After Defense | Recovery Rate | Clean Label Recovery | Injected Edge Prune | Defense Drift | Bose-Einstein Fitness | Assortativity | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nettack | 0.4040 | 0.8840 | 101.9% | 95.4% | 93.2% | 0.0440 | 0.6030 | -0.1020 | PASS |
| DICE | 0.3920 | 0.8890 | 102.9% | 96.3% | 94.1% | 0.0460 | 0.6120 | -0.0950 | PASS |
| Meta Attack | 0.3810 | 0.8940 | 103.8% | 97.1% | 94.8% | 0.0490 | 0.6240 | -0.0910 | PASS |
| Random Structure | 0.4100 | 0.8830 | 101.7% | 95.1% | 92.5% | 0.0430 | 0.5990 | -0.1060 | PASS |
| Feature Perturbation | 0.3380 | 0.9020 | 105.0% | 98.2% | 100.0% | 0.0520 | 0.6370 | -0.0890 | PASS |
| Edge Flip | 0.4040 | 0.8870 | 102.5% | 96.0% | 93.6% | 0.0470 | 0.6080 | -0.0990 | PASS |
| Gradient Attack (PGD) | 0.1180 | 0.9340 | 107.8% | 100.0% | 100.0% | 0.0640 | 0.6610 | -0.0830 | PASS |
| Temporal Perturbation | 0.2970 | 0.9110 | 106.2% | 99.1% | 100.0% | 0.0580 | 0.6480 | -0.0860 | PASS |
