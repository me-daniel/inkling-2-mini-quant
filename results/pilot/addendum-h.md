# Addendum H results

Written by `analysis/addendum_h.py`. Reading rules as registered in `CHANGELOG.md` (Addendum H). KL on S at P = 3. Sections appear once their runs exist.

## H0. The 2 x 2 on S

| | our layout | UD's layout |
|---|---|---|
| IQ4_NL down (Metal) | cand 0.02931 ± 0.00042 | mirror 0.02149 ± 0.00029 |
| NVFP4 down (CPU on a Mac) | hybrid 0.02813 ± 0.00049 | UD-IQ3_XXS 0.02050 ± 0.00029 |

- hybrid minus UD-IQ3_XXS +0.00762 (2 SE = 0.00113): our layout is worse than UD's even with UD's down projections.
- mirror minus cand -0.00782 (2 SE = 0.00101): UD's layout is better than ours at equal down too.

## Paired contrasts on S (corrected analysis, amendment 4)

Per-chunk differences with a paired bootstrap; positive means b has lower KL than a. The registered H0 readings above use unpaired SEs and stay as registered.

| Contrast (S) | a | b | a - b [95% CI, paired] |
|---|---|---|---|
| down effect on our layout | sel-2 | hybrid | +0.00118 [+0.00063, +0.00171] |
| down effect on UD's layout | mirror | UD-IQ3_XXS | +0.00099 [+0.00070, +0.00125] |
| layout effect, IQ4_NL down | sel-2 | mirror | +0.00782 [+0.00709, +0.00853] |
| layout effect, NVFP4 down | hybrid | UD-IQ3_XXS | +0.00763 [+0.00690, +0.00837] |
| UD non-expert tensors | sel-2 | ne-ud | -0.00010 [-0.00058, +0.00038] |
| UD gate_up experts | sel-2 | gu-ud | +0.00810 [+0.00743, +0.00874] |
| UD gate_up, layers 1-5 | sel-2 | gu15-ud | +0.00248 [+0.00193, +0.00303] |
| UD gate_up, layers 9-11 | sel-2 | gu911-ud | +0.00300 [+0.00246, +0.00364] |
| our encoder on UD's map (gate_up only) | sel-2 | gu-v2a | +0.00214 [+0.00144, +0.00285] |
| UD's encoding at UD's map, beyond ours | gu-v2a | gu-ud | +0.00596 [+0.00524, +0.00670] |
| v2a against the mirror | mirror | v2a | -0.00630 [-0.00690, -0.00574] |
| v2a against UD-IQ3_XXS | UD-IQ3_XXS | v2a | -0.00729 [-0.00796, -0.00662] |

## Amendment 2. Which part of UD's layout

| Build (S, P = 3) | KL | drop from cand | share of the cand-to-mirror drop |
|---|---|---|---|
| cand + UD's non-expert tensors | 0.02941 ± 0.00041 | -0.00010 | -1% |
| cand + UD's gate_up experts | 0.02121 ± 0.00027 | +0.00810 | 104% |

