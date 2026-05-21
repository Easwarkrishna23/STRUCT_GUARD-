# Metric Alignment Notes

The compact unified tables now use one row per attack. Each metric has paired
`After Attack` and `After Defense` columns, matching the requested syntax-style layout.

Defense-side ASR is reported as residual ASR:

`Residual ASR = ASR * (1 - Clean Label Recovery)`

Defense-side entropy is a residual diagnostic derived from attack entropy and remaining unrecovered attacked nodes:

`Defense Entropy = Attack Entropy * (1 - Clean Label Recovery)`

Negative defense-side drop means the defended accuracy is above the clean baseline.
