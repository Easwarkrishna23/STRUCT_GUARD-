# Elliptic Bitcoin Dataset: Unified Attack and Defense Metrics

Baseline accuracy: 0.8750

This table uses the same columns for both stages. "After Attack" rows show damage. "After Defense" rows show the repaired state. Residual ASR is computed as `ASR * (1 - Clean Label Recovery)`. Defense-side entropy and homophily gap are residual diagnostics derived from the stored attack entropy/homophily and recovery/pruning rates, because the older defense table did not persist separate defense entropy/homophily values.

| Attack | Type | Stage | Accuracy | F1 | Drop From Baseline | Drop % Baseline | ASR / Residual ASR | Embedding Drift | Neighborhood Entropy | Homophily Drop / Gap | Bose-Einstein Fitness | Assortativity | Recovery Rate | Clean Label Recovery | Injected Edge Prune | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nettack | Poisoning | After Attack | 0.4040 | 0.3920 | 0.4710 | 53.8% | 78.1% | 0.5360 | 0.9140 | 0.2810 | 0.2380 | -0.2470 | N/A | N/A | N/A | PASS |
| Nettack | Poisoning | After Defense | 0.8840 | 0.4970 | -0.0090 | -1.0% | 3.6% | 0.0440 | 0.0420 | 0.0191 | 0.6030 | -0.1020 | 101.9% | 95.4% | 93.2% | PASS |
| DICE | Poisoning | After Attack | 0.3920 | 0.3810 | 0.4830 | 55.2% | 80.6% | 0.5620 | 0.9870 | 0.3140 | 0.2210 | -0.2830 | N/A | N/A | N/A | PASS |
| DICE | Poisoning | After Defense | 0.8890 | 0.5060 | -0.0140 | -1.6% | 3.0% | 0.0460 | 0.0365 | 0.0185 | 0.6120 | -0.0950 | 102.9% | 96.3% | 94.1% | PASS |
| Meta Attack | Poisoning | After Attack | 0.3810 | 0.3690 | 0.4940 | 56.5% | 82.3% | 0.5880 | 1.0410 | 0.3360 | 0.2070 | -0.3010 | N/A | N/A | N/A | PASS |
| Meta Attack | Poisoning | After Defense | 0.8940 | 0.5150 | -0.0190 | -2.2% | 2.4% | 0.0490 | 0.0302 | 0.0175 | 0.6240 | -0.0910 | 103.8% | 97.1% | 94.8% | PASS |
| Random Structure | Poisoning | After Attack | 0.4100 | 0.3990 | 0.4650 | 53.1% | 76.4% | 0.5210 | 0.9020 | 0.2680 | 0.2450 | -0.2330 | N/A | N/A | N/A | PASS |
| Random Structure | Poisoning | After Defense | 0.8830 | 0.4930 | -0.0080 | -0.9% | 3.7% | 0.0430 | 0.0442 | 0.0201 | 0.5990 | -0.1060 | 101.7% | 95.1% | 92.5% | PASS |
| Feature Perturbation | Evasion | After Attack | 0.3380 | 0.3270 | 0.5370 | 61.4% | 85.1% | 0.7440 | 1.1180 | 0.1420 | 0.2290 | -0.1280 | N/A | N/A | N/A | PASS |
| Feature Perturbation | Evasion | After Defense | 0.9020 | 0.5310 | -0.0270 | -3.1% | 1.5% | 0.0520 | 0.0201 | 0.0000 | 0.6370 | -0.0890 | 105.0% | 98.2% | 100.0% | PASS |
| Edge Flip | Evasion | After Attack | 0.4040 | 0.3910 | 0.4710 | 53.8% | 79.2% | 0.5480 | 0.9660 | 0.3020 | 0.2160 | -0.2710 | N/A | N/A | N/A | PASS |
| Edge Flip | Evasion | After Defense | 0.8870 | 0.5030 | -0.0120 | -1.4% | 3.2% | 0.0470 | 0.0386 | 0.0193 | 0.6080 | -0.0990 | 102.5% | 96.0% | 93.6% | PASS |
| Gradient Attack (PGD) | Evasion | After Attack | 0.1180 | 0.1040 | 0.7570 | 86.5% | 100.0% | 1.4670 | 1.2840 | 0.1880 | 0.1930 | -0.1520 | N/A | N/A | N/A | PASS |
| Gradient Attack (PGD) | Evasion | After Defense | 0.9340 | 0.5720 | -0.0590 | -6.7% | 0.0% | 0.0640 | 0.0000 | 0.0000 | 0.6610 | -0.0830 | 107.8% | 100.0% | 100.0% | PASS |
| Temporal Perturbation | Evasion | After Attack | 0.2970 | 0.2860 | 0.5780 | 66.1% | 88.7% | 0.9180 | 1.3610 | 0.1640 | 0.2010 | -0.1710 | N/A | N/A | N/A | PASS |
| Temporal Perturbation | Evasion | After Defense | 0.9110 | 0.5480 | -0.0360 | -4.1% | 0.8% | 0.0580 | 0.0122 | 0.0000 | 0.6480 | -0.0860 | 106.2% | 99.1% | 100.0% | PASS |
