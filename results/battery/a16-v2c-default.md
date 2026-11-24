# Amendment 17's rule for v2c against v1 on A2 (Daniel's override of the German guard, 24 Nov)

Written by an inline call of `analysis/battery_A.py` `ratio_ci` on the A2 rows of `results/pilot/kl.tsv` (same resampler, 10,000 resamples, seed 0). A2 also selected v2c over v2m (amendment 16), so this reuses its data; v2m's line in `a16.md` is the unselected comparison.

- R (v2c / v1, A2 python, rust, go, typescript) = 0.773 [0.745, 0.803]
- python: 0.767 [0.732, 0.803]
- rust: 0.785 [0.722, 0.860]
- go: 0.782 [0.728, 0.831]
- typescript: 0.760 [0.691, 0.827]
- german (guard text, battery A): 0.868 [0.842, 0.895]
- rule (upper bound below 1, no code domain above 1.05, tasks passed +2.5 points): met
