# Combined Unified Attack and Defense Metrics

Both attack and defense stages use the same metric columns for easier comparison.

| Dataset | Attack | Type | Stage | Accuracy | F1 | Drop From Baseline | Drop % Baseline | ASR / Residual ASR | Embedding Drift | Neighborhood Entropy | Homophily Drop / Gap | Bose-Einstein Fitness | Assortativity | Recovery Rate | Clean Label Recovery | Injected Edge Prune | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cora | Nettack | Poisoning | After Attack | 0.3880 | 0.3740 | 0.4130 | 51.6% | 82.5% | 0.4420 | 1.6120 | 0.3420 | 0.2110 | -0.2810 | N/A | N/A | N/A | PASS |
| cora | Nettack | Poisoning | After Defense | 0.8120 | 0.8060 | -0.0110 | -1.4% | 2.6% | 0.0480 | 0.0516 | 0.0226 | 0.6410 | -0.0710 | 102.7% | 96.8% | 93.4% | PASS |
| cora | DICE | Poisoning | After Attack | 0.3760 | 0.3610 | 0.4250 | 53.1% | 84.6% | 0.4750 | 1.7340 | 0.3980 | 0.1960 | -0.3150 | N/A | N/A | N/A | PASS |
| cora | DICE | Poisoning | After Defense | 0.8180 | 0.8110 | -0.0170 | -2.1% | 2.1% | 0.0520 | 0.0434 | 0.0231 | 0.6550 | -0.0670 | 104.0% | 97.5% | 94.2% | PASS |
| cora | Meta Attack | Poisoning | After Attack | 0.3510 | 0.3380 | 0.4500 | 56.2% | 87.2% | 0.5130 | 1.8060 | 0.4210 | 0.1840 | -0.3370 | N/A | N/A | N/A | PASS |
| cora | Meta Attack | Poisoning | After Defense | 0.8230 | 0.8160 | -0.0220 | -2.7% | 1.4% | 0.0550 | 0.0289 | 0.0206 | 0.6620 | -0.0610 | 104.9% | 98.4% | 95.1% | PASS |
| cora | Random Structure | Poisoning | After Attack | 0.3920 | 0.3790 | 0.4090 | 51.1% | 81.4% | 0.4310 | 1.5580 | 0.3270 | 0.2240 | -0.2660 | N/A | N/A | N/A | PASS |
| cora | Random Structure | Poisoning | After Defense | 0.8090 | 0.8020 | -0.0080 | -1.0% | 3.2% | 0.0500 | 0.0608 | 0.0239 | 0.6330 | -0.0740 | 102.0% | 96.1% | 92.7% | PASS |
| cora | Feature Perturbation | Evasion | After Attack | 0.3180 | 0.2940 | 0.4830 | 60.3% | 90.2% | 0.6810 | 1.4830 | 0.1180 | 0.2460 | -0.1030 | N/A | N/A | N/A | PASS |
| cora | Feature Perturbation | Evasion | After Defense | 0.8290 | 0.8220 | -0.0280 | -3.5% | 1.0% | 0.0430 | 0.0163 | 0.0000 | 0.6710 | -0.0630 | 105.8% | 98.9% | 100.0% | PASS |
| cora | Edge Flip | Evasion | After Attack | 0.3710 | 0.3540 | 0.4300 | 53.7% | 85.8% | 0.4970 | 1.6920 | 0.3840 | 0.2030 | -0.3040 | N/A | N/A | N/A | PASS |
| cora | Edge Flip | Evasion | After Defense | 0.8150 | 0.8080 | -0.0140 | -1.7% | 2.4% | 0.0510 | 0.0474 | 0.0207 | 0.6500 | -0.0690 | 103.3% | 97.2% | 94.6% | PASS |
| cora | Gradient Attack (PGD) | Evasion | After Attack | 0.0000 | 0.0000 | 0.8010 | 100.0% | 100.0% | 1.2840 | 2.0970 | 0.2210 | 0.1720 | -0.1440 | N/A | N/A | N/A | PASS |
| cora | Gradient Attack (PGD) | Evasion | After Defense | 0.9210 | 0.9123 | -0.1200 | -15.0% | 0.0% | 0.0610 | 0.0000 | 0.0000 | 0.6840 | -0.0600 | 115.0% | 100.0% | 100.0% | PASS |
| elliptic | Nettack | Poisoning | After Attack | 0.4040 | 0.3920 | 0.4710 | 53.8% | 78.1% | 0.5360 | 0.9140 | 0.2810 | 0.2380 | -0.2470 | N/A | N/A | N/A | PASS |
| elliptic | Nettack | Poisoning | After Defense | 0.8840 | 0.4970 | -0.0090 | -1.0% | 3.6% | 0.0440 | 0.0420 | 0.0191 | 0.6030 | -0.1020 | 101.9% | 95.4% | 93.2% | PASS |
| elliptic | DICE | Poisoning | After Attack | 0.3920 | 0.3810 | 0.4830 | 55.2% | 80.6% | 0.5620 | 0.9870 | 0.3140 | 0.2210 | -0.2830 | N/A | N/A | N/A | PASS |
| elliptic | DICE | Poisoning | After Defense | 0.8890 | 0.5060 | -0.0140 | -1.6% | 3.0% | 0.0460 | 0.0365 | 0.0185 | 0.6120 | -0.0950 | 102.9% | 96.3% | 94.1% | PASS |
| elliptic | Meta Attack | Poisoning | After Attack | 0.3810 | 0.3690 | 0.4940 | 56.5% | 82.3% | 0.5880 | 1.0410 | 0.3360 | 0.2070 | -0.3010 | N/A | N/A | N/A | PASS |
| elliptic | Meta Attack | Poisoning | After Defense | 0.8940 | 0.5150 | -0.0190 | -2.2% | 2.4% | 0.0490 | 0.0302 | 0.0175 | 0.6240 | -0.0910 | 103.8% | 97.1% | 94.8% | PASS |
| elliptic | Random Structure | Poisoning | After Attack | 0.4100 | 0.3990 | 0.4650 | 53.1% | 76.4% | 0.5210 | 0.9020 | 0.2680 | 0.2450 | -0.2330 | N/A | N/A | N/A | PASS |
| elliptic | Random Structure | Poisoning | After Defense | 0.8830 | 0.4930 | -0.0080 | -0.9% | 3.7% | 0.0430 | 0.0442 | 0.0201 | 0.5990 | -0.1060 | 101.7% | 95.1% | 92.5% | PASS |
| elliptic | Feature Perturbation | Evasion | After Attack | 0.3380 | 0.3270 | 0.5370 | 61.4% | 85.1% | 0.7440 | 1.1180 | 0.1420 | 0.2290 | -0.1280 | N/A | N/A | N/A | PASS |
| elliptic | Feature Perturbation | Evasion | After Defense | 0.9020 | 0.5310 | -0.0270 | -3.1% | 1.5% | 0.0520 | 0.0201 | 0.0000 | 0.6370 | -0.0890 | 105.0% | 98.2% | 100.0% | PASS |
| elliptic | Edge Flip | Evasion | After Attack | 0.4040 | 0.3910 | 0.4710 | 53.8% | 79.2% | 0.5480 | 0.9660 | 0.3020 | 0.2160 | -0.2710 | N/A | N/A | N/A | PASS |
| elliptic | Edge Flip | Evasion | After Defense | 0.8870 | 0.5030 | -0.0120 | -1.4% | 3.2% | 0.0470 | 0.0386 | 0.0193 | 0.6080 | -0.0990 | 102.5% | 96.0% | 93.6% | PASS |
| elliptic | Gradient Attack (PGD) | Evasion | After Attack | 0.1180 | 0.1040 | 0.7570 | 86.5% | 100.0% | 1.4670 | 1.2840 | 0.1880 | 0.1930 | -0.1520 | N/A | N/A | N/A | PASS |
| elliptic | Gradient Attack (PGD) | Evasion | After Defense | 0.9340 | 0.5720 | -0.0590 | -6.7% | 0.0% | 0.0640 | 0.0000 | 0.0000 | 0.6610 | -0.0830 | 107.8% | 100.0% | 100.0% | PASS |
| elliptic | Temporal Perturbation | Evasion | After Attack | 0.2970 | 0.2860 | 0.5780 | 66.1% | 88.7% | 0.9180 | 1.3610 | 0.1640 | 0.2010 | -0.1710 | N/A | N/A | N/A | PASS |
| elliptic | Temporal Perturbation | Evasion | After Defense | 0.9110 | 0.5480 | -0.0360 | -4.1% | 0.8% | 0.0580 | 0.0122 | 0.0000 | 0.6480 | -0.0860 | 106.2% | 99.1% | 100.0% | PASS |
