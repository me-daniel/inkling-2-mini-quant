# Addendum H results

Written by `analysis/addendum_h.py`. Reading rules as registered in `CHANGELOG.md` (Addendum H). KL on S at P = 3. Sections appear once their runs exist.

## H0. The 2 x 2 on S

| | our layout | UD's layout |
|---|---|---|
| IQ4_NL down (Metal) | cand 0.02931 ± 0.00042 | mirror 0.02149 ± 0.00029 |
| NVFP4 down (CPU on a Mac) | hybrid 0.02813 ± 0.00049 | UD-IQ3_XXS 0.02050 ± 0.00029 |

- hybrid minus UD-IQ3_XXS +0.00762 (2 SE = 0.00113): our layout is worse than UD's even with UD's down projections.
- mirror minus cand -0.00782 (2 SE = 0.00101): UD's layout is better than ours at equal down too.

## Amendment 2. Which part of UD's layout

| Build (S, P = 3) | KL | drop from cand | share of the cand-to-mirror drop |
|---|---|---|---|
| cand + UD's non-expert tensors | 0.02941 ± 0.00041 | -0.00010 | -1% |
| cand + UD's gate_up experts | 0.02121 ± 0.00027 | +0.00810 | 104% |

