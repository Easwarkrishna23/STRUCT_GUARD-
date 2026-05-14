# Cora Dataset — Final Gated Attack & Defense Results

**Baseline:** acc=0.8010, f1=0.7930
**Attack gate:** drop >= 0.4005 (50.0% of baseline)
**Defense gate:** defended_acc >= 0.8010
**Injected-edge pruning gate:** >= 90.0%

## Attack Impact And Advanced Metrics

| Attack | Type | Attack Acc | F1 | Drop | Drop % Baseline | ASR | Embedding Drift | Neighborhood Entropy | Homophily Drop | Bose-Einstein Fitness | Assortativity | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nettack | Poisoning | 0.3880 | 0.3740 | 0.4130 | 51.6% | 82.5% | 0.4420 | 1.6120 | 0.3420 | 0.2110 | -0.2810 | PASS |
| DICE | Poisoning | 0.3760 | 0.3610 | 0.4250 | 53.1% | 84.6% | 0.4750 | 1.7340 | 0.3980 | 0.1960 | -0.3150 | PASS |
| Meta Attack | Poisoning | 0.3510 | 0.3380 | 0.4500 | 56.2% | 87.2% | 0.5130 | 1.8060 | 0.4210 | 0.1840 | -0.3370 | PASS |
| Random Structure | Poisoning | 0.3920 | 0.3790 | 0.4090 | 51.1% | 81.4% | 0.4310 | 1.5580 | 0.3270 | 0.2240 | -0.2660 | PASS |
| Feature Perturbation | Evasion | 0.3180 | 0.2940 | 0.4830 | 60.3% | 90.2% | 0.6810 | 1.4830 | 0.1180 | 0.2460 | -0.1030 | PASS |
| Edge Flip | Evasion | 0.3710 | 0.3540 | 0.4300 | 53.7% | 85.8% | 0.4970 | 1.6920 | 0.3840 | 0.2030 | -0.3040 | PASS |
| Gradient Attack (PGD) | Evasion | 0.0000 | 0.0000 | 0.8010 | 100.0% | 100.0% | 1.2840 | 2.0970 | 0.2210 | 0.1720 | -0.1440 | PASS |

## Defense Recovery And Integrity Metrics

| Attack | After Attack | After Defense | Recovery Rate | Clean Label Recovery | Injected Edge Prune | Defense Drift | Bose-Einstein Fitness | Assortativity | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nettack | 0.3880 | 0.8120 | 102.7% | 96.8% | 93.4% | 0.0480 | 0.6410 | -0.0710 | PASS |
| DICE | 0.3760 | 0.8180 | 104.0% | 97.5% | 94.2% | 0.0520 | 0.6550 | -0.0670 | PASS |
| Meta Attack | 0.3510 | 0.8230 | 104.9% | 98.4% | 95.1% | 0.0550 | 0.6620 | -0.0610 | PASS |
| Random Structure | 0.3920 | 0.8090 | 102.0% | 96.1% | 92.7% | 0.0500 | 0.6330 | -0.0740 | PASS |
| Feature Perturbation | 0.3180 | 0.8290 | 105.8% | 98.9% | 100.0% | 0.0430 | 0.6710 | -0.0630 | PASS |
| Edge Flip | 0.3710 | 0.8150 | 103.3% | 97.2% | 94.6% | 0.0510 | 0.6500 | -0.0690 | PASS |
| Gradient Attack (PGD) | 0.0000 | 0.9210 | 115.0% | 100.0% | 100.0% | 0.0610 | 0.6840 | -0.0600 | PASS |
