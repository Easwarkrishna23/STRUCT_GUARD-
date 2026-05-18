# Why the Earlier Attack and Defense Tables Had Different Metrics

The older result display separated two different questions. The attack table answered: how much damage did the attack cause? Therefore it emphasized attack accuracy, F1, drop, drop percentage, ASR, embedding drift, neighborhood entropy, homophily drop, Bose-Einstein fitness, and assortativity.

The defense table answered: how much recovery and graph repair happened after STRUC-GUARD+? Therefore it emphasized after-defense accuracy, recovery rate, clean-label recovery, injected-edge pruning, defense drift, defended Bose-Einstein fitness, and defended assortativity.

The new unified tables keep one common column schema for both stages. For the defense rows, residual ASR is derived as `ASR * (1 - Clean Label Recovery)`. Defense-side entropy and homophily gap are shown as residual diagnostics derived from stored attack entropy/homophily and recovery/pruning rates because the original defense table did not persist separately recomputed defense entropy/homophily values.
