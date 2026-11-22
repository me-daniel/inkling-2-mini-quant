# Amendment 9 refinement pilot: offline errors and gains (9b + 9c guards)

Concatenation of two outputs of `scripts/gram-error.py --reg 0.1 --extra ...`: layers 9, 10, 11 (the S pilot, amendment 9d; `results/pilot/refine-pilot-9-10-11.md`) and layers 6, 20 (offline record; `results/pilot/refine-pilot-6-20.md`, with the unguarded-gain 9b build as refined9b). All errors are in-sample: the search minimised the same Gram it is scored on.

## gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined = `models/refined/gu-refined-9-10-11.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/refine-pilot-9-10-11.tsv`.
Grams: `models/grams/gu-9-10-11` (capture settings in its log, not restated here), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

### Layer 9 (IQ3_XXS; UD `.scale` present, min 3.921e-05, max 0.0001086)

- Gram layout: expert 50: max |strict lower| = 0, max |strict upper| = 1.761e+05
- Gram layout: expert 16: max |strict lower| = 0, max |strict upper| = 1.762e+05
- Gram layout: expert 86: max |strict lower| = 0, max |strict upper| = 1.77e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 1.46e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.9.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.681030 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 44, median 5049, max 51787

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.01802 | 0.01385 | 0.7687 | 128 of 128 | 0.01129 | 0.6265 | 89 of 128 |
| full | gate | count-weighted | 0.01792 | 0.01373 | 0.7660 | 128 of 128 | 0.00923 | 0.5150 | 89 of 128 |
| full | gate | trace-weighted | 0.01792 | 0.01374 | 0.7665 | 128 of 128 | 0.00921 | 0.5141 | 89 of 128 |
| full | up | unweighted | 0.02423 | 0.01866 | 0.7704 | 128 of 128 | 0.01479 | 0.6105 | 89 of 128 |
| full | up | count-weighted | 0.02398 | 0.01837 | 0.7662 | 128 of 128 | 0.01178 | 0.4911 | 89 of 128 |
| full | up | trace-weighted | 0.02402 | 0.01842 | 0.7668 | 128 of 128 | 0.01176 | 0.4896 | 89 of 128 |
| full | both | unweighted | 0.02060 | 0.01585 | 0.7696 | 128 of 128 | 0.01273 | 0.6183 | 89 of 128 |
| full | both | count-weighted | 0.02047 | 0.01568 | 0.7661 | 128 of 128 | 0.01030 | 0.5031 | 89 of 128 |
| full | both | trace-weighted | 0.02049 | 0.01570 | 0.7666 | 128 of 128 | 0.01028 | 0.5020 | 89 of 128 |
| reg | gate | unweighted | 0.01875 | 0.01492 | 0.7961 | 128 of 128 | 0.01289 | 0.6875 | 89 of 128 |
| reg | gate | count-weighted | 0.01865 | 0.01481 | 0.7937 | 128 of 128 | 0.01104 | 0.5920 | 89 of 128 |
| reg | gate | trace-weighted | 0.01865 | 0.01481 | 0.7941 | 128 of 128 | 0.01103 | 0.5911 | 89 of 128 |
| reg | up | unweighted | 0.02476 | 0.01975 | 0.7976 | 128 of 128 | 0.01668 | 0.6734 | 89 of 128 |
| reg | up | count-weighted | 0.02453 | 0.01947 | 0.7938 | 128 of 128 | 0.01400 | 0.5708 | 89 of 128 |
| reg | up | trace-weighted | 0.02457 | 0.01951 | 0.7943 | 128 of 128 | 0.01399 | 0.5696 | 89 of 128 |
| reg | both | unweighted | 0.02127 | 0.01695 | 0.7969 | 128 of 128 | 0.01447 | 0.6803 | 89 of 128 |
| reg | both | count-weighted | 0.02115 | 0.01679 | 0.7937 | 128 of 128 | 0.01230 | 0.5814 | 89 of 128 |
| reg | both | trace-weighted | 0.02116 | 0.01681 | 0.7942 | 128 of 128 | 0.01229 | 0.5805 | 89 of 128 |
| bias | gate | unweighted | 0.00295 | 9.601e-04 | 0.3258 | 128 of 128 | 0.00125 | 0.4253 | 89 of 128 |
| bias | gate | count-weighted | 0.00275 | 8.189e-04 | 0.2974 | 128 of 128 | 4.555e-04 | 0.1654 | 89 of 128 |
| bias | gate | trace-weighted | 0.00276 | 8.213e-04 | 0.2976 | 128 of 128 | 4.546e-04 | 0.1647 | 89 of 128 |
| bias | up | unweighted | 0.00393 | 0.00130 | 0.3319 | 128 of 128 | 0.00144 | 0.3663 | 89 of 128 |
| bias | up | count-weighted | 0.00369 | 0.00112 | 0.3024 | 128 of 128 | 2.896e-04 | 0.0784 | 89 of 128 |
| bias | up | trace-weighted | 0.00371 | 0.00112 | 0.3028 | 128 of 128 | 2.837e-04 | 0.0766 | 89 of 128 |
| bias | both | unweighted | 0.00336 | 0.00110 | 0.3289 | 128 of 128 | 0.00133 | 0.3960 | 89 of 128 |
| bias | both | count-weighted | 0.00315 | 9.441e-04 | 0.2999 | 128 of 128 | 3.850e-04 | 0.1223 | 89 of 128 |
| bias | both | trace-weighted | 0.00316 | 9.476e-04 | 0.3002 | 128 of 128 | 3.822e-04 | 0.1211 | 89 of 128 |
| rest | gate | unweighted | 0.01507 | 0.01289 | 0.8554 | 128 of 128 | 0.01003 | 0.6658 | 89 of 128 |
| rest | gate | count-weighted | 0.01517 | 0.01291 | 0.8511 | 128 of 128 | 0.00877 | 0.5785 | 89 of 128 |
| rest | gate | trace-weighted | 0.01516 | 0.01292 | 0.8518 | 128 of 128 | 0.00876 | 0.5777 | 89 of 128 |
| rest | up | unweighted | 0.02030 | 0.01736 | 0.8553 | 128 of 128 | 0.01335 | 0.6578 | 89 of 128 |
| rest | up | count-weighted | 0.02029 | 0.01726 | 0.8507 | 128 of 128 | 0.01149 | 0.5662 | 89 of 128 |
| rest | up | trace-weighted | 0.02032 | 0.01730 | 0.8514 | 128 of 128 | 0.01148 | 0.5650 | 89 of 128 |
| rest | both | unweighted | 0.01724 | 0.01475 | 0.8553 | 128 of 128 | 0.01141 | 0.6615 | 89 of 128 |
| rest | both | count-weighted | 0.01732 | 0.01473 | 0.8508 | 128 of 128 | 0.00991 | 0.5724 | 89 of 128 |
| rest | both | trace-weighted | 0.01733 | 0.01476 | 0.8516 | 128 of 128 | 0.00990 | 0.5713 | 89 of 128 |
| diag | gate | unweighted | 0.03184 | 0.03409 | 1.0704 | 0 of 128 | 0.04133 | 1.2979 | 0 of 128 |
| diag | gate | count-weighted | 0.03181 | 0.03406 | 1.0707 | 0 of 128 | 0.04326 | 1.3602 | 0 of 128 |
| diag | gate | trace-weighted | 0.03180 | 0.03405 | 1.0707 | 0 of 128 | 0.04329 | 1.3614 | 0 of 128 |
| diag | up | unweighted | 0.03187 | 0.03412 | 1.0705 | 0 of 128 | 0.04156 | 1.3039 | 0 of 128 |
| diag | up | count-weighted | 0.03182 | 0.03407 | 1.0707 | 0 of 128 | 0.04354 | 1.3683 | 0 of 128 |
| diag | up | trace-weighted | 0.03181 | 0.03406 | 1.0707 | 0 of 128 | 0.04357 | 1.3695 | 0 of 128 |
| diag | both | unweighted | 0.03186 | 0.03410 | 1.0705 | 0 of 128 | 0.04144 | 1.3009 | 0 of 128 |
| diag | both | count-weighted | 0.03181 | 0.03406 | 1.0707 | 0 of 128 | 0.04340 | 1.3643 | 0 of 128 |
| diag | both | trace-weighted | 0.03180 | 0.03405 | 1.0707 | 0 of 128 | 0.04343 | 1.3654 | 0 of 128 |
| plain | gate | unweighted | 0.18715 | 0.19324 | 1.0326 | 0 of 128 | 0.21852 | 1.1676 | 0 of 128 |
| plain | gate | count-weighted | 0.18715 | 0.19325 | 1.0326 | 0 of 128 | 0.22600 | 1.2076 | 0 of 128 |
| plain | gate | trace-weighted | 0.18716 | 0.19326 | 1.0326 | 0 of 128 | 0.22615 | 1.2083 | 0 of 128 |
| plain | up | unweighted | 0.18712 | 0.19321 | 1.0325 | 0 of 128 | 0.21906 | 1.1707 | 0 of 128 |
| plain | up | count-weighted | 0.18712 | 0.19321 | 1.0325 | 0 of 128 | 0.22676 | 1.2118 | 0 of 128 |
| plain | up | trace-weighted | 0.18713 | 0.19322 | 1.0326 | 0 of 128 | 0.22691 | 1.2126 | 0 of 128 |
| plain | both | unweighted | 0.18713 | 0.19322 | 1.0325 | 0 of 128 | 0.21879 | 1.1692 | 0 of 128 |
| plain | both | count-weighted | 0.18713 | 0.19323 | 1.0326 | 0 of 128 | 0.22638 | 1.2097 | 0 of 128 |
| plain | both | trace-weighted | 0.18714 | 0.19324 | 1.0326 | 0 of 128 | 0.22653 | 1.2105 | 0 of 128 |
| gain | gate | unweighted | 0.99618 | 0.99647 | 1.0003 | 45 of 128 | 0.99151 | 0.9953 | 89 of 128 |
| gain | gate | count-weighted | 0.99620 | 0.99639 | 1.0002 | 45 of 128 | 0.99029 | 0.9941 | 89 of 128 |
| gain | gate | trace-weighted | 0.99621 | 0.99641 | 1.0002 | 45 of 128 | 0.99029 | 0.9941 | 89 of 128 |
| gain | up | unweighted | 0.99621 | 0.99652 | 1.0003 | 36 of 128 | 0.99160 | 0.9954 | 89 of 128 |
| gain | up | count-weighted | 0.99625 | 0.99645 | 1.0002 | 36 of 128 | 0.99047 | 0.9942 | 89 of 128 |
| gain | up | trace-weighted | 0.99626 | 0.99647 | 1.0002 | 36 of 128 | 0.99046 | 0.9942 | 89 of 128 |
| gain | both | unweighted | 0.99620 | 0.99649 | 1.0003 | 41 of 128 | 0.99156 | 0.9953 | 89 of 128 |
| gain | both | count-weighted | 0.99622 | 0.99642 | 1.0002 | 41 of 128 | 0.99038 | 0.9941 | 89 of 128 |
| gain | both | trace-weighted | 0.99623 | 0.99644 | 1.0002 | 41 of 128 | 0.99038 | 0.9941 | 89 of 128 |
| energy | gate | unweighted | 1.02739 | 1.03028 | 1.0028 | 1 of 128 | 1.03125 | 1.0038 | 0 of 128 |
| energy | gate | count-weighted | 1.02742 | 1.03013 | 1.0026 | 1 of 128 | 1.03173 | 1.0042 | 0 of 128 |
| energy | gate | trace-weighted | 1.02745 | 1.03017 | 1.0027 | 1 of 128 | 1.03180 | 1.0042 | 0 of 128 |
| energy | up | unweighted | 1.02744 | 1.03036 | 1.0028 | 0 of 128 | 1.03168 | 1.0041 | 0 of 128 |
| energy | up | count-weighted | 1.02751 | 1.03023 | 1.0026 | 0 of 128 | 1.03243 | 1.0048 | 0 of 128 |
| energy | up | trace-weighted | 1.02753 | 1.03027 | 1.0027 | 0 of 128 | 1.03249 | 1.0048 | 0 of 128 |
| energy | both | unweighted | 1.02742 | 1.03032 | 1.0028 | 0 of 128 | 1.03146 | 1.0039 | 0 of 128 |
| energy | both | count-weighted | 1.02747 | 1.03018 | 1.0026 | 0 of 128 | 1.03208 | 1.0045 | 0 of 128 |
| energy | both | trace-weighted | 1.02749 | 1.03022 | 1.0027 | 0 of 128 | 1.03214 | 1.0045 | 0 of 128 |
| ogain | gate | unweighted | 0.99875 | 0.99890 | 1.0001 | 59 of 128 | 1.00629 | 1.0075 | 0 of 128 |
| ogain | gate | count-weighted | 0.99880 | 0.99883 | 1.0000 | 59 of 128 | 1.00901 | 1.0102 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99882 | 0.99883 | 1.0000 | 59 of 128 | 1.00908 | 1.0103 | 0 of 128 |
| ogain | up | unweighted | 0.99835 | 0.99861 | 1.0003 | 54 of 128 | 1.00435 | 1.0060 | 0 of 128 |
| ogain | up | count-weighted | 0.99839 | 0.99853 | 1.0001 | 54 of 128 | 1.00628 | 1.0079 | 0 of 128 |
| ogain | up | trace-weighted | 0.99838 | 0.99854 | 1.0002 | 54 of 128 | 1.00631 | 1.0079 | 0 of 128 |
| ogain | both | unweighted | 0.99858 | 0.99877 | 1.0002 | 53 of 128 | 1.00547 | 1.0069 | 0 of 128 |
| ogain | both | count-weighted | 0.99863 | 0.99870 | 1.0001 | 53 of 128 | 1.00785 | 1.0092 | 0 of 128 |
| ogain | both | trace-weighted | 0.99863 | 0.99870 | 1.0001 | 53 of 128 | 1.00790 | 1.0093 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -4.1939e-03 | [-4.3115e-03, -4.0802e-03] |
| up | -5.6070e-03 | [-5.6806e-03, -5.5278e-03] |
| both | -4.7879e-03 | [-4.8766e-03, -4.6998e-03] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -8.6912e-03 | [-8.9878e-03, -8.4098e-03] | -4.4973e-03 | [-4.7498e-03, -4.2747e-03] |
| full | up | -1.2205e-02 | [-1.2554e-02, -1.1882e-02] | -6.5980e-03 | [-6.9600e-03, -6.2767e-03] |
| full | both | -1.0169e-02 | [-1.0463e-02, -9.8998e-03] | -5.3812e-03 | [-5.6690e-03, -5.1288e-03] |
| reg | gate | -7.6116e-03 | [-7.8595e-03, -7.3753e-03] | -3.7628e-03 | [-3.9741e-03, -3.5765e-03] |
| reg | up | -1.0527e-02 | [-1.0809e-02, -1.0263e-02] | -5.4695e-03 | [-5.7689e-03, -5.2015e-03] |
| reg | both | -8.8513e-03 | [-9.0923e-03, -8.6273e-03] | -4.4890e-03 | [-4.7302e-03, -4.2773e-03] |
| bias | gate | -2.2979e-03 | [-2.3973e-03, -2.2004e-03] | -3.6335e-04 | [-4.1406e-04, -3.0869e-04] |
| bias | up | -3.4046e-03 | [-3.5220e-03, -3.2838e-03] | -8.2734e-04 | [-9.0066e-04, -7.4694e-04] |
| bias | both | -2.7633e-03 | [-2.8597e-03, -2.6674e-03] | -5.5905e-04 | [-6.1657e-04, -4.9586e-04] |
| rest | gate | -6.3933e-03 | [-6.6319e-03, -6.1745e-03] | -4.1340e-03 | [-4.3700e-03, -3.9302e-03] |
| rest | up | -8.8003e-03 | [-9.1063e-03, -8.5294e-03] | -5.7706e-03 | [-6.1058e-03, -5.4791e-03] |
| rest | both | -7.4057e-03 | [-7.6563e-03, -7.1797e-03] | -4.8222e-03 | [-5.0924e-03, -4.5915e-03] |
| diag | gate | +1.1458e-02 | [+1.1063e-02, +1.1929e-02] | +9.2079e-03 | [+8.8170e-03, +9.6715e-03] |
| diag | up | +1.1720e-02 | [+1.1323e-02, +1.2192e-02] | +9.4704e-03 | [+9.0773e-03, +9.9380e-03] |
| diag | both | +1.1589e-02 | [+1.1193e-02, +1.2060e-02] | +9.3391e-03 | [+8.9476e-03, +9.8031e-03] |

### Layer 10 (IQ3_XXS; UD `.scale` present, min 3.827e-05, max 0.0001814)

- Gram layout: expert 85: max |strict lower| = 0, max |strict upper| = 1.641e+05
- Gram layout: expert 88: max |strict lower| = 0, max |strict upper| = 1.154e+05
- Gram layout: expert 92: max |strict lower| = 0, max |strict upper| = 1.149e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 7.37e-09; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.10.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.626424 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 9, median 5360, max 56685

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.01924 | 0.01507 | 0.7830 | 128 of 128 | 0.01149 | 0.5969 | 93 of 128 |
| full | gate | count-weighted | 0.01929 | 0.01510 | 0.7829 | 128 of 128 | 0.00985 | 0.5104 | 93 of 128 |
| full | gate | trace-weighted | 0.01937 | 0.01518 | 0.7836 | 128 of 128 | 0.00987 | 0.5095 | 93 of 128 |
| full | up | unweighted | 0.02485 | 0.01943 | 0.7819 | 128 of 128 | 0.01465 | 0.5895 | 93 of 128 |
| full | up | count-weighted | 0.02505 | 0.01958 | 0.7818 | 128 of 128 | 0.01240 | 0.4949 | 93 of 128 |
| full | up | trace-weighted | 0.02511 | 0.01965 | 0.7824 | 128 of 128 | 0.01241 | 0.4941 | 93 of 128 |
| full | both | unweighted | 0.02151 | 0.01683 | 0.7825 | 128 of 128 | 0.01274 | 0.5921 | 93 of 128 |
| full | both | count-weighted | 0.02163 | 0.01692 | 0.7824 | 128 of 128 | 0.01087 | 0.5026 | 93 of 128 |
| full | both | trace-weighted | 0.02170 | 0.01699 | 0.7831 | 128 of 128 | 0.01089 | 0.5018 | 93 of 128 |
| reg | gate | unweighted | 0.01995 | 0.01614 | 0.8089 | 128 of 128 | 0.01324 | 0.6636 | 93 of 128 |
| reg | gate | count-weighted | 0.02001 | 0.01618 | 0.8088 | 128 of 128 | 0.01178 | 0.5888 | 93 of 128 |
| reg | gate | trace-weighted | 0.02008 | 0.01626 | 0.8095 | 128 of 128 | 0.01181 | 0.5881 | 93 of 128 |
| reg | up | unweighted | 0.02539 | 0.02051 | 0.8078 | 128 of 128 | 0.01667 | 0.6568 | 93 of 128 |
| reg | up | count-weighted | 0.02558 | 0.02067 | 0.8078 | 128 of 128 | 0.01471 | 0.5749 | 93 of 128 |
| reg | up | trace-weighted | 0.02565 | 0.02073 | 0.8083 | 128 of 128 | 0.01473 | 0.5742 | 93 of 128 |
| reg | both | unweighted | 0.02218 | 0.01793 | 0.8084 | 128 of 128 | 0.01462 | 0.6594 | 93 of 128 |
| reg | both | count-weighted | 0.02229 | 0.01802 | 0.8083 | 128 of 128 | 0.01297 | 0.5819 | 93 of 128 |
| reg | both | trace-weighted | 0.02237 | 0.01809 | 0.8090 | 128 of 128 | 0.01300 | 0.5812 | 93 of 128 |
| bias | gate | unweighted | 0.00309 | 0.00108 | 0.3488 | 128 of 128 | 0.00118 | 0.3814 | 93 of 128 |
| bias | gate | count-weighted | 0.00277 | 8.451e-04 | 0.3056 | 128 of 128 | 3.848e-04 | 0.1391 | 93 of 128 |
| bias | gate | trace-weighted | 0.00278 | 8.544e-04 | 0.3076 | 128 of 128 | 3.830e-04 | 0.1379 | 93 of 128 |
| bias | up | unweighted | 0.00400 | 0.00138 | 0.3443 | 128 of 128 | 0.00138 | 0.3461 | 93 of 128 |
| bias | up | count-weighted | 0.00358 | 0.00107 | 0.2980 | 128 of 128 | 2.588e-04 | 0.0724 | 93 of 128 |
| bias | up | trace-weighted | 0.00359 | 0.00107 | 0.2995 | 128 of 128 | 2.553e-04 | 0.0712 | 93 of 128 |
| bias | both | unweighted | 0.00346 | 0.00120 | 0.3467 | 128 of 128 | 0.00125 | 0.3621 | 93 of 128 |
| bias | both | count-weighted | 0.00309 | 9.357e-04 | 0.3024 | 128 of 128 | 3.319e-04 | 0.1072 | 93 of 128 |
| bias | both | trace-weighted | 0.00311 | 9.449e-04 | 0.3041 | 128 of 128 | 3.295e-04 | 0.1060 | 93 of 128 |
| rest | gate | unweighted | 0.01615 | 0.01399 | 0.8662 | 128 of 128 | 0.01031 | 0.6381 | 93 of 128 |
| rest | gate | count-weighted | 0.01653 | 0.01426 | 0.8628 | 128 of 128 | 0.00946 | 0.5725 | 93 of 128 |
| rest | gate | trace-weighted | 0.01659 | 0.01432 | 0.8633 | 128 of 128 | 0.00949 | 0.5717 | 93 of 128 |
| rest | up | unweighted | 0.02085 | 0.01805 | 0.8657 | 128 of 128 | 0.01327 | 0.6362 | 93 of 128 |
| rest | up | count-weighted | 0.02147 | 0.01852 | 0.8624 | 128 of 128 | 0.01214 | 0.5653 | 93 of 128 |
| rest | up | trace-weighted | 0.02153 | 0.01858 | 0.8629 | 128 of 128 | 0.01215 | 0.5646 | 93 of 128 |
| rest | both | unweighted | 0.01806 | 0.01564 | 0.8659 | 128 of 128 | 0.01149 | 0.6361 | 93 of 128 |
| rest | both | count-weighted | 0.01853 | 0.01599 | 0.8626 | 128 of 128 | 0.01054 | 0.5687 | 93 of 128 |
| rest | both | trace-weighted | 0.01859 | 0.01605 | 0.8631 | 128 of 128 | 0.01056 | 0.5679 | 93 of 128 |
| diag | gate | unweighted | 0.03262 | 0.03483 | 1.0678 | 0 of 128 | 0.04295 | 1.3166 | 0 of 128 |
| diag | gate | count-weighted | 0.03269 | 0.03492 | 1.0681 | 0 of 128 | 0.04469 | 1.3670 | 0 of 128 |
| diag | gate | trace-weighted | 0.03269 | 0.03492 | 1.0681 | 0 of 128 | 0.04470 | 1.3674 | 0 of 128 |
| diag | up | unweighted | 0.03264 | 0.03485 | 1.0677 | 0 of 128 | 0.04315 | 1.3218 | 0 of 128 |
| diag | up | count-weighted | 0.03269 | 0.03492 | 1.0681 | 0 of 128 | 0.04490 | 1.3735 | 0 of 128 |
| diag | up | trace-weighted | 0.03269 | 0.03491 | 1.0680 | 0 of 128 | 0.04491 | 1.3740 | 0 of 128 |
| diag | both | unweighted | 0.03263 | 0.03484 | 1.0677 | 0 of 128 | 0.04305 | 1.3192 | 0 of 128 |
| diag | both | count-weighted | 0.03269 | 0.03492 | 1.0681 | 0 of 128 | 0.04480 | 1.3703 | 0 of 128 |
| diag | both | trace-weighted | 0.03269 | 0.03491 | 1.0681 | 0 of 128 | 0.04481 | 1.3707 | 0 of 128 |
| plain | gate | unweighted | 0.18609 | 0.19193 | 1.0313 | 0 of 128 | 0.21584 | 1.1599 | 0 of 128 |
| plain | gate | count-weighted | 0.18606 | 0.19194 | 1.0316 | 0 of 128 | 0.22111 | 1.1884 | 0 of 128 |
| plain | gate | trace-weighted | 0.18605 | 0.19194 | 1.0316 | 0 of 128 | 0.22116 | 1.1887 | 0 of 128 |
| plain | up | unweighted | 0.18606 | 0.19189 | 1.0314 | 0 of 128 | 0.21620 | 1.1620 | 0 of 128 |
| plain | up | count-weighted | 0.18599 | 0.19187 | 1.0316 | 0 of 128 | 0.22153 | 1.1911 | 0 of 128 |
| plain | up | trace-weighted | 0.18599 | 0.19187 | 1.0316 | 0 of 128 | 0.22159 | 1.1914 | 0 of 128 |
| plain | both | unweighted | 0.18608 | 0.19191 | 1.0313 | 0 of 128 | 0.21602 | 1.1609 | 0 of 128 |
| plain | both | count-weighted | 0.18603 | 0.19191 | 1.0316 | 0 of 128 | 0.22132 | 1.1897 | 0 of 128 |
| plain | both | trace-weighted | 0.18602 | 0.19191 | 1.0316 | 0 of 128 | 0.22137 | 1.1901 | 0 of 128 |
| gain | gate | unweighted | 0.99611 | 0.99641 | 1.0003 | 24 of 128 | 0.99119 | 0.9951 | 93 of 128 |
| gain | gate | count-weighted | 0.99622 | 0.99652 | 1.0003 | 24 of 128 | 0.99017 | 0.9939 | 93 of 128 |
| gain | gate | trace-weighted | 0.99623 | 0.99653 | 1.0003 | 24 of 128 | 0.99017 | 0.9939 | 93 of 128 |
| gain | up | unweighted | 0.99613 | 0.99645 | 1.0003 | 28 of 128 | 0.99125 | 0.9951 | 93 of 128 |
| gain | up | count-weighted | 0.99625 | 0.99653 | 1.0003 | 28 of 128 | 0.99030 | 0.9940 | 93 of 128 |
| gain | up | trace-weighted | 0.99626 | 0.99654 | 1.0003 | 28 of 128 | 0.99030 | 0.9940 | 93 of 128 |
| gain | both | unweighted | 0.99612 | 0.99643 | 1.0003 | 26 of 128 | 0.99122 | 0.9951 | 93 of 128 |
| gain | both | count-weighted | 0.99624 | 0.99653 | 1.0003 | 26 of 128 | 0.99023 | 0.9940 | 93 of 128 |
| gain | both | trace-weighted | 0.99624 | 0.99653 | 1.0003 | 26 of 128 | 0.99024 | 0.9940 | 93 of 128 |
| energy | gate | unweighted | 1.02686 | 1.02966 | 1.0027 | 1 of 128 | 1.02934 | 1.0024 | 1 of 128 |
| energy | gate | count-weighted | 1.02706 | 1.02988 | 1.0027 | 1 of 128 | 1.02928 | 1.0022 | 1 of 128 |
| energy | gate | trace-weighted | 1.02708 | 1.02989 | 1.0027 | 1 of 128 | 1.02932 | 1.0022 | 1 of 128 |
| energy | up | unweighted | 1.02687 | 1.02972 | 1.0028 | 0 of 128 | 1.02962 | 1.0027 | 0 of 128 |
| energy | up | count-weighted | 1.02709 | 1.02989 | 1.0027 | 0 of 128 | 1.02973 | 1.0026 | 0 of 128 |
| energy | up | trace-weighted | 1.02711 | 1.02990 | 1.0027 | 0 of 128 | 1.02976 | 1.0026 | 0 of 128 |
| energy | both | unweighted | 1.02687 | 1.02969 | 1.0028 | 0 of 128 | 1.02948 | 1.0025 | 1 of 128 |
| energy | both | count-weighted | 1.02708 | 1.02988 | 1.0027 | 0 of 128 | 1.02950 | 1.0024 | 1 of 128 |
| energy | both | trace-weighted | 1.02709 | 1.02990 | 1.0027 | 0 of 128 | 1.02954 | 1.0024 | 1 of 128 |
| ogain | gate | unweighted | 0.99849 | 0.99871 | 1.0002 | 47 of 128 | 1.00587 | 1.0074 | 0 of 128 |
| ogain | gate | count-weighted | 0.99853 | 0.99869 | 1.0002 | 47 of 128 | 1.00790 | 1.0094 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99854 | 0.99869 | 1.0002 | 47 of 128 | 1.00792 | 1.0094 | 0 of 128 |
| ogain | up | unweighted | 0.99793 | 0.99824 | 1.0003 | 45 of 128 | 1.00411 | 1.0062 | 0 of 128 |
| ogain | up | count-weighted | 0.99792 | 0.99819 | 1.0003 | 45 of 128 | 1.00545 | 1.0076 | 0 of 128 |
| ogain | up | trace-weighted | 0.99792 | 0.99819 | 1.0003 | 45 of 128 | 1.00547 | 1.0076 | 0 of 128 |
| ogain | both | unweighted | 0.99826 | 0.99852 | 1.0003 | 43 of 128 | 1.00514 | 1.0069 | 0 of 128 |
| ogain | both | count-weighted | 0.99828 | 0.99849 | 1.0002 | 43 of 128 | 1.00688 | 1.0086 | 0 of 128 |
| ogain | both | trace-weighted | 0.99829 | 0.99849 | 1.0002 | 43 of 128 | 1.00689 | 1.0086 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -4.1883e-03 | [-4.3796e-03, -3.9824e-03] |
| up | -5.4664e-03 | [-5.5882e-03, -5.3497e-03] |
| both | -4.7057e-03 | [-4.8469e-03, -4.5536e-03] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -9.4467e-03 | [-1.0028e-02, -8.8696e-03] | -5.2583e-03 | [-5.7128e-03, -4.8324e-03] |
| full | up | -1.2653e-02 | [-1.3041e-02, -1.2296e-02] | -7.1861e-03 | [-7.5784e-03, -6.8320e-03] |
| full | both | -1.0757e-02 | [-1.1263e-02, -1.0267e-02] | -6.0510e-03 | [-6.4760e-03, -5.6506e-03] |
| reg | gate | -8.2263e-03 | [-8.7137e-03, -7.7403e-03] | -4.4009e-03 | [-4.7774e-03, -4.0443e-03] |
| reg | up | -1.0876e-02 | [-1.1191e-02, -1.0585e-02] | -5.9577e-03 | [-6.2812e-03, -5.6643e-03] |
| reg | both | -9.3210e-03 | [-9.7369e-03, -8.9167e-03] | -5.0483e-03 | [-5.4037e-03, -4.7170e-03] |
| bias | gate | -2.3806e-03 | [-2.5617e-03, -2.2008e-03] | -4.6034e-04 | [-5.6128e-04, -3.6896e-04] |
| bias | up | -3.3176e-03 | [-3.4518e-03, -3.1941e-03] | -8.0709e-04 | [-9.0673e-04, -7.1968e-04] |
| bias | both | -2.7630e-03 | [-2.9213e-03, -2.6088e-03] | -6.0384e-04 | [-7.0539e-04, -5.1342e-04] |
| rest | gate | -7.0661e-03 | [-7.4892e-03, -6.6558e-03] | -4.7980e-03 | [-5.1706e-03, -4.4499e-03] |
| rest | up | -9.3349e-03 | [-9.6260e-03, -9.0712e-03] | -6.3790e-03 | [-6.7069e-03, -6.0847e-03] |
| rest | both | -7.9937e-03 | [-8.3641e-03, -7.6421e-03] | -5.4472e-03 | [-5.8021e-03, -5.1179e-03] |
| diag | gate | +1.1998e-02 | [+1.1564e-02, +1.2510e-02] | +9.7706e-03 | [+9.3312e-03, +1.0286e-02] |
| diag | up | +1.2211e-02 | [+1.1783e-02, +1.2718e-02] | +9.9862e-03 | [+9.5580e-03, +1.0494e-02] |
| diag | both | +1.2105e-02 | [+1.1675e-02, +1.2610e-02] | +9.8784e-03 | [+9.4462e-03, +1.0388e-02] |

### Layer 11 (IQ3_XXS; UD `.scale` present, min 4.001e-05, max 0.0002121)

- Gram layout: expert 44: max |strict lower| = 0, max |strict upper| = 2.035e+05
- Gram layout: expert 113: max |strict lower| = 0, max |strict upper| = 1.418e+05
- Gram layout: expert 2: max |strict lower| = 0, max |strict upper| = 1.292e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 1.62e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.11.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = -0.000000 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 1, median 6532, max 60142

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.01842 | 0.01396 | 0.7580 | 128 of 128 | 0.01159 | 0.6293 | 86 of 128 |
| full | gate | count-weighted | 0.01862 | 0.01412 | 0.7583 | 128 of 128 | 0.00950 | 0.5100 | 86 of 128 |
| full | gate | trace-weighted | 0.01865 | 0.01416 | 0.7589 | 128 of 128 | 0.00948 | 0.5085 | 86 of 128 |
| full | up | unweighted | 0.02453 | 0.01857 | 0.7570 | 128 of 128 | 0.01542 | 0.6285 | 86 of 128 |
| full | up | count-weighted | 0.02435 | 0.01846 | 0.7579 | 128 of 128 | 0.01203 | 0.4938 | 86 of 128 |
| full | up | trace-weighted | 0.02440 | 0.01851 | 0.7586 | 128 of 128 | 0.01200 | 0.4920 | 86 of 128 |
| full | both | unweighted | 0.02073 | 0.01571 | 0.7577 | 128 of 128 | 0.01298 | 0.6264 | 86 of 128 |
| full | both | count-weighted | 0.02093 | 0.01587 | 0.7582 | 128 of 128 | 0.01051 | 0.5021 | 86 of 128 |
| full | both | trace-weighted | 0.02097 | 0.01591 | 0.7588 | 128 of 128 | 0.01049 | 0.5004 | 86 of 128 |
| reg | gate | unweighted | 0.01915 | 0.01507 | 0.7870 | 128 of 128 | 0.01321 | 0.6901 | 86 of 128 |
| reg | gate | count-weighted | 0.01936 | 0.01525 | 0.7874 | 128 of 128 | 0.01139 | 0.5883 | 86 of 128 |
| reg | gate | trace-weighted | 0.01939 | 0.01528 | 0.7880 | 128 of 128 | 0.01138 | 0.5870 | 86 of 128 |
| reg | up | unweighted | 0.02510 | 0.01973 | 0.7861 | 128 of 128 | 0.01728 | 0.6885 | 86 of 128 |
| reg | up | count-weighted | 0.02493 | 0.01962 | 0.7870 | 128 of 128 | 0.01431 | 0.5739 | 86 of 128 |
| reg | up | trace-weighted | 0.02497 | 0.01967 | 0.7876 | 128 of 128 | 0.01429 | 0.5723 | 86 of 128 |
| reg | both | unweighted | 0.02143 | 0.01686 | 0.7868 | 128 of 128 | 0.01473 | 0.6873 | 86 of 128 |
| reg | both | count-weighted | 0.02163 | 0.01703 | 0.7873 | 128 of 128 | 0.01257 | 0.5812 | 86 of 128 |
| reg | both | trace-weighted | 0.02167 | 0.01707 | 0.7878 | 128 of 128 | 0.01256 | 0.5799 | 86 of 128 |
| bias | gate | unweighted | 0.00311 | 0.00110 | 0.3520 | 128 of 128 | 0.00142 | 0.4576 | 86 of 128 |
| bias | gate | count-weighted | 0.00272 | 7.976e-04 | 0.2934 | 128 of 128 | 4.255e-04 | 0.1565 | 86 of 128 |
| bias | gate | trace-weighted | 0.00272 | 8.011e-04 | 0.2940 | 128 of 128 | 4.242e-04 | 0.1557 | 86 of 128 |
| bias | up | unweighted | 0.00418 | 0.00147 | 0.3508 | 128 of 128 | 0.00182 | 0.4349 | 86 of 128 |
| bias | up | count-weighted | 0.00357 | 0.00104 | 0.2926 | 128 of 128 | 3.235e-04 | 0.0907 | 86 of 128 |
| bias | up | trace-weighted | 0.00357 | 0.00105 | 0.2933 | 128 of 128 | 3.178e-04 | 0.0889 | 86 of 128 |
| bias | both | unweighted | 0.00351 | 0.00124 | 0.3516 | 128 of 128 | 0.00156 | 0.4427 | 86 of 128 |
| bias | both | count-weighted | 0.00306 | 8.968e-04 | 0.2930 | 128 of 128 | 3.818e-04 | 0.1247 | 86 of 128 |
| bias | both | trace-weighted | 0.00307 | 9.007e-04 | 0.2937 | 128 of 128 | 3.789e-04 | 0.1235 | 86 of 128 |
| rest | gate | unweighted | 0.01531 | 0.01287 | 0.8405 | 128 of 128 | 0.01017 | 0.6642 | 86 of 128 |
| rest | gate | count-weighted | 0.01590 | 0.01332 | 0.8378 | 128 of 128 | 0.00907 | 0.5704 | 86 of 128 |
| rest | gate | trace-weighted | 0.01593 | 0.01336 | 0.8384 | 128 of 128 | 0.00906 | 0.5688 | 86 of 128 |
| rest | up | unweighted | 0.02036 | 0.01711 | 0.8403 | 127 of 128 | 0.01360 | 0.6682 | 86 of 128 |
| rest | up | count-weighted | 0.02079 | 0.01741 | 0.8378 | 127 of 128 | 0.01170 | 0.5630 | 86 of 128 |
| rest | up | trace-weighted | 0.02082 | 0.01746 | 0.8385 | 127 of 128 | 0.01168 | 0.5611 | 86 of 128 |
| rest | both | unweighted | 0.01722 | 0.01447 | 0.8406 | 127 of 128 | 0.01143 | 0.6638 | 86 of 128 |
| rest | both | count-weighted | 0.01787 | 0.01497 | 0.8379 | 127 of 128 | 0.01013 | 0.5667 | 86 of 128 |
| rest | both | trace-weighted | 0.01790 | 0.01501 | 0.8385 | 127 of 128 | 0.01011 | 0.5650 | 86 of 128 |
| diag | gate | unweighted | 0.03280 | 0.03535 | 1.0776 | 0 of 128 | 0.04211 | 1.2837 | 0 of 128 |
| diag | gate | count-weighted | 0.03282 | 0.03539 | 1.0783 | 0 of 128 | 0.04486 | 1.3669 | 0 of 128 |
| diag | gate | trace-weighted | 0.03282 | 0.03539 | 1.0783 | 0 of 128 | 0.04491 | 1.3684 | 0 of 128 |
| diag | up | unweighted | 0.03282 | 0.03537 | 1.0776 | 0 of 128 | 0.04228 | 1.2883 | 0 of 128 |
| diag | up | count-weighted | 0.03281 | 0.03538 | 1.0783 | 0 of 128 | 0.04505 | 1.3732 | 0 of 128 |
| diag | up | trace-weighted | 0.03281 | 0.03538 | 1.0783 | 0 of 128 | 0.04510 | 1.3747 | 0 of 128 |
| diag | both | unweighted | 0.03281 | 0.03536 | 1.0776 | 0 of 128 | 0.04220 | 1.2860 | 0 of 128 |
| diag | both | count-weighted | 0.03281 | 0.03538 | 1.0783 | 0 of 128 | 0.04496 | 1.3700 | 0 of 128 |
| diag | both | trace-weighted | 0.03281 | 0.03538 | 1.0783 | 0 of 128 | 0.04500 | 1.3715 | 0 of 128 |
| plain | gate | unweighted | 0.18561 | 0.19218 | 1.0354 | 0 of 128 | 0.21193 | 1.1418 | 0 of 128 |
| plain | gate | count-weighted | 0.18552 | 0.19218 | 1.0359 | 0 of 128 | 0.21983 | 1.1849 | 0 of 128 |
| plain | gate | trace-weighted | 0.18551 | 0.19218 | 1.0359 | 0 of 128 | 0.21995 | 1.1856 | 0 of 128 |
| plain | up | unweighted | 0.18556 | 0.19213 | 1.0354 | 0 of 128 | 0.21220 | 1.1436 | 0 of 128 |
| plain | up | count-weighted | 0.18545 | 0.19211 | 1.0359 | 0 of 128 | 0.22019 | 1.1873 | 0 of 128 |
| plain | up | trace-weighted | 0.18545 | 0.19210 | 1.0359 | 0 of 128 | 0.22031 | 1.1880 | 0 of 128 |
| plain | both | unweighted | 0.18558 | 0.19216 | 1.0354 | 0 of 128 | 0.21207 | 1.1427 | 0 of 128 |
| plain | both | count-weighted | 0.18548 | 0.19215 | 1.0359 | 0 of 128 | 0.22001 | 1.1861 | 0 of 128 |
| plain | both | trace-weighted | 0.18548 | 0.19214 | 1.0359 | 0 of 128 | 0.22013 | 1.1868 | 0 of 128 |
| gain | gate | unweighted | 0.99616 | 0.99643 | 1.0003 | 32 of 128 | 0.99170 | 0.9955 | 86 of 128 |
| gain | gate | count-weighted | 0.99627 | 0.99651 | 1.0002 | 32 of 128 | 0.99028 | 0.9940 | 86 of 128 |
| gain | gate | trace-weighted | 0.99627 | 0.99651 | 1.0002 | 32 of 128 | 0.99028 | 0.9940 | 86 of 128 |
| gain | up | unweighted | 0.99620 | 0.99647 | 1.0003 | 31 of 128 | 0.99180 | 0.9956 | 86 of 128 |
| gain | up | count-weighted | 0.99630 | 0.99654 | 1.0002 | 31 of 128 | 0.99041 | 0.9941 | 86 of 128 |
| gain | up | trace-weighted | 0.99630 | 0.99654 | 1.0002 | 31 of 128 | 0.99040 | 0.9941 | 86 of 128 |
| gain | both | unweighted | 0.99618 | 0.99645 | 1.0003 | 32 of 128 | 0.99175 | 0.9956 | 86 of 128 |
| gain | both | count-weighted | 0.99628 | 0.99653 | 1.0002 | 32 of 128 | 0.99035 | 0.9940 | 86 of 128 |
| gain | both | trace-weighted | 0.99629 | 0.99652 | 1.0002 | 32 of 128 | 0.99034 | 0.9940 | 86 of 128 |
| energy | gate | unweighted | 1.02677 | 1.02979 | 1.0029 | 2 of 128 | 1.02868 | 1.0019 | 1 of 128 |
| energy | gate | count-weighted | 1.02695 | 1.02995 | 1.0029 | 2 of 128 | 1.02895 | 1.0020 | 1 of 128 |
| energy | gate | trace-weighted | 1.02695 | 1.02995 | 1.0029 | 2 of 128 | 1.02899 | 1.0020 | 1 of 128 |
| energy | up | unweighted | 1.02684 | 1.02986 | 1.0029 | 1 of 128 | 1.02901 | 1.0021 | 0 of 128 |
| energy | up | count-weighted | 1.02699 | 1.02999 | 1.0029 | 1 of 128 | 1.02936 | 1.0023 | 0 of 128 |
| energy | up | trace-weighted | 1.02699 | 1.02999 | 1.0029 | 1 of 128 | 1.02940 | 1.0023 | 0 of 128 |
| energy | both | unweighted | 1.02680 | 1.02982 | 1.0029 | 1 of 128 | 1.02884 | 1.0020 | 0 of 128 |
| energy | both | count-weighted | 1.02697 | 1.02997 | 1.0029 | 1 of 128 | 1.02916 | 1.0021 | 0 of 128 |
| energy | both | trace-weighted | 1.02697 | 1.02997 | 1.0029 | 1 of 128 | 1.02919 | 1.0022 | 0 of 128 |
| ogain | gate | unweighted | 0.99858 | 0.99864 | 1.0001 | 60 of 128 | 1.00554 | 1.0070 | 0 of 128 |
| ogain | gate | count-weighted | 0.99862 | 0.99867 | 1.0001 | 60 of 128 | 1.00822 | 1.0096 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99861 | 0.99866 | 1.0001 | 60 of 128 | 1.00824 | 1.0096 | 0 of 128 |
| ogain | up | unweighted | 0.99766 | 0.99812 | 1.0005 | 38 of 128 | 1.00370 | 1.0061 | 0 of 128 |
| ogain | up | count-weighted | 0.99775 | 0.99818 | 1.0004 | 38 of 128 | 1.00593 | 1.0082 | 0 of 128 |
| ogain | up | trace-weighted | 0.99774 | 0.99817 | 1.0004 | 38 of 128 | 1.00594 | 1.0082 | 0 of 128 |
| ogain | both | unweighted | 0.99825 | 0.99846 | 1.0002 | 49 of 128 | 1.00483 | 1.0066 | 0 of 128 |
| ogain | both | count-weighted | 0.99828 | 0.99849 | 1.0002 | 49 of 128 | 1.00728 | 1.0090 | 0 of 128 |
| ogain | both | trace-weighted | 0.99827 | 0.99848 | 1.0002 | 49 of 128 | 1.00729 | 1.0090 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -4.5002e-03 | [-4.6663e-03, -4.3323e-03] |
| up | -5.8951e-03 | [-6.0129e-03, -5.7786e-03] |
| both | -5.0610e-03 | [-5.1777e-03, -4.9344e-03] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -9.1255e-03 | [-9.5584e-03, -8.6928e-03] | -4.6253e-03 | [-4.9505e-03, -4.3036e-03] |
| full | up | -1.2327e-02 | [-1.2745e-02, -1.1914e-02] | -6.4321e-03 | [-6.8244e-03, -6.0390e-03] |
| full | both | -1.0422e-02 | [-1.0803e-02, -1.0036e-02] | -5.3608e-03 | [-5.6939e-03, -5.0272e-03] |
| reg | gate | -7.9716e-03 | [-8.3355e-03, -7.6088e-03] | -3.8560e-03 | [-4.1273e-03, -3.5858e-03] |
| reg | up | -1.0623e-02 | [-1.0965e-02, -1.0281e-02] | -5.3140e-03 | [-5.6403e-03, -4.9818e-03] |
| reg | both | -9.0581e-03 | [-9.3723e-03, -8.7362e-03] | -4.4570e-03 | [-4.7339e-03, -4.1766e-03] |
| bias | gate | -2.2935e-03 | [-2.4201e-03, -2.1666e-03] | -3.7214e-04 | [-4.3930e-04, -3.0654e-04] |
| bias | up | -3.2446e-03 | [-3.3684e-03, -3.1262e-03] | -7.2060e-04 | [-7.9966e-04, -6.3813e-04] |
| bias | both | -2.6792e-03 | [-2.7922e-03, -2.5685e-03] | -5.1501e-04 | [-5.8360e-04, -4.4625e-04] |
| rest | gate | -6.8320e-03 | [-7.1484e-03, -6.5161e-03] | -4.2531e-03 | [-4.5285e-03, -3.9834e-03] |
| rest | up | -9.0826e-03 | [-9.4028e-03, -8.7605e-03] | -5.7115e-03 | [-6.0408e-03, -5.3841e-03] |
| rest | both | -7.7425e-03 | [-8.0237e-03, -7.4561e-03] | -4.8458e-03 | [-5.1302e-03, -4.5673e-03] |
| diag | gate | +1.2039e-02 | [+1.1662e-02, +1.2452e-02] | +9.4680e-03 | [+9.0816e-03, +9.8887e-03] |
| diag | up | +1.2244e-02 | [+1.1862e-02, +1.2666e-02] | +9.6743e-03 | [+9.2854e-03, +1.0101e-02] |
| diag | both | +1.2141e-02 | [+1.1763e-02, +1.2557e-02] | +9.5710e-03 | [+9.1839e-03, +9.9954e-03] |

### Reading (amendment 7 rule, applied literally)

Rule: supported if UD is lower than ours on the full error and higher or equal on the diagonal one in both layers; if UD is also worse on the full error, layer-local output error does not explain UD's edge.

| mean | half | layer 9 full UD < ours | layer 9 diag UD >= ours | layer 10 full UD < ours | layer 10 diag UD >= ours | layer 11 full UD < ours | layer 11 diag UD >= ours | branch |
|---|---|---|---|---|---|---|---|---|
| unweighted | gate | yes | yes | yes | yes | yes | yes | supported |
| unweighted | up | yes | yes | yes | yes | yes | yes | supported |
| unweighted | both | yes | yes | yes | yes | yes | yes | supported |
| count-weighted | gate | yes | yes | yes | yes | yes | yes | supported |
| count-weighted | up | yes | yes | yes | yes | yes | yes | supported |
| count-weighted | both | yes | yes | yes | yes | yes | yes | supported |
| trace-weighted | gate | yes | yes | yes | yes | yes | yes | supported |
| trace-weighted | up | yes | yes | yes | yes | yes | yes | supported |
| trace-weighted | both | yes | yes | yes | yes | yes | yes | supported |

In layer 9 (both halves, count-weighted), UD's full error is 0.01568 against our 0.02047 (ratio 0.766; UD - ours -4.788e-03, 95% CI [-4.877e-03, -4.700e-03]) and its diagonal error 0.03406 against our 0.03181 (ratio 1.071); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 9 (added after registration, not part of the rule): the input-mean part is 0.00315 ours against 9.441e-04 UD (ratio 0.300), the centred rest 0.01732 against 0.01473 (ratio 0.851). In layer 10 (both halves, count-weighted), UD's full error is 0.01692 against our 0.02163 (ratio 0.782; UD - ours -4.706e-03, 95% CI [-4.847e-03, -4.554e-03]) and its diagonal error 0.03492 against our 0.03269 (ratio 1.068); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 10 (added after registration, not part of the rule): the input-mean part is 0.00309 ours against 9.357e-04 UD (ratio 0.302), the centred rest 0.01853 against 0.01599 (ratio 0.863). In layer 11 (both halves, count-weighted), UD's full error is 0.01587 against our 0.02093 (ratio 0.758; UD - ours -5.061e-03, 95% CI [-5.178e-03, -4.934e-03]) and its diagonal error 0.03538 against our 0.03281 (ratio 1.078); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 11 (added after registration, not part of the rule): the input-mean part is 0.00306 ours against 8.968e-04 UD (ratio 0.293), the centred rest 0.01787 against 0.01497 (ratio 0.838). Under the registered rule this is "supported" for both halves together, count-weighted; across the three means and three halves the branches are: supported.

### refined against ours and UD

In layer 9 (both halves, count-weighted), refined's full error is 0.01030 against our 0.02047 and UD's 0.01568 (refined - ours -1.017e-02, 95% CI [-1.046e-02, -9.900e-03]; (ours - refined) / (ours - UD) = 2.124; lower than ours in 89 of 128 experts). Bias part 3.850e-04 (ours 0.00315, UD 9.441e-04), rest 0.00991 (ours 0.01732, UD 0.01473). Regularised (the search objective) 0.01230 (ours 0.02115, UD 0.01679). Diagonal 0.04340 (ours 0.03181, UD 0.03406), plain 0.22638 (ours 0.18713, UD 0.19323). Gains (ours / UD / refined): plain 0.9962 / 0.9964 / 0.9904, energy 1.0275 / 1.0302 / 1.0321, output 0.9986 / 0.9987 / 1.0078. In layer 10 (both halves, count-weighted), refined's full error is 0.01087 against our 0.02163 and UD's 0.01692 (refined - ours -1.076e-02, 95% CI [-1.126e-02, -1.027e-02]; (ours - refined) / (ours - UD) = 2.286; lower than ours in 93 of 128 experts). Bias part 3.319e-04 (ours 0.00309, UD 9.357e-04), rest 0.01054 (ours 0.01853, UD 0.01599). Regularised (the search objective) 0.01297 (ours 0.02229, UD 0.01802). Diagonal 0.04480 (ours 0.03269, UD 0.03492), plain 0.22132 (ours 0.18603, UD 0.19191). Gains (ours / UD / refined): plain 0.9962 / 0.9965 / 0.9902, energy 1.0271 / 1.0299 / 1.0295, output 0.9983 / 0.9985 / 1.0069. In layer 11 (both halves, count-weighted), refined's full error is 0.01051 against our 0.02093 and UD's 0.01587 (refined - ours -1.042e-02, 95% CI [-1.080e-02, -1.004e-02]; (ours - refined) / (ours - UD) = 2.059; lower than ours in 86 of 128 experts). Bias part 3.818e-04 (ours 0.00306, UD 8.968e-04), rest 0.01013 (ours 0.01787, UD 0.01497). Regularised (the search objective) 0.01257 (ours 0.02163, UD 0.01703). Diagonal 0.04496 (ours 0.03281, UD 0.03538), plain 0.22001 (ours 0.18548, UD 0.19215). Gains (ours / UD / refined): plain 0.9963 / 0.9965 / 0.9903, energy 1.0270 / 1.0300 / 1.0292, output 0.9983 / 0.9985 / 1.0073.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined (count-w.) | refined / ours | refined / UD |
|---|---|---|---|---|---|---|---|---|
| 9 | < 2304 | 39 | 0.028 | 0.02032 | 0.01568 | 0.02032 | 1.000 | 1.296 |
| 9 | >= 2304 | 89 | 0.972 | 0.02047 | 0.01568 | 0.01001 | 0.489 | 0.639 |
| 10 | < 2304 | 35 | 0.021 | 0.02078 | 0.01615 | 0.02078 | 1.000 | 1.287 |
| 10 | >= 2304 | 93 | 0.979 | 0.02165 | 0.01694 | 0.01066 | 0.492 | 0.629 |
| 11 | < 2304 | 42 | 0.030 | 0.02057 | 0.01548 | 0.02057 | 1.000 | 1.328 |
| 11 | >= 2304 | 86 | 0.970 | 0.02094 | 0.01588 | 0.01020 | 0.487 | 0.642 |

Refinement log `models/refined/gu-refined-9-10-11.gguf.json` (scripts/refine-codes.py; wall time 5661 s for 384 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 9 | 89 | 39 | 4: 89 | 56361694 | 1845199 of 1845504 | 6985 | 2.69e-10 | 2.32e-10 | yes |
| 10 | 93 | 35 | 4: 93 | 59359136 | 1928193 of 1928448 | 7934 | 3.94e-10 | 2.91e-10 | yes |
| 11 | 86 | 42 | 4: 86 | 54000335 | 1783013 of 1783296 | 7372 | 2.22e-10 | 2.01e-10 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 9: max abs(log - here) on reg 2.61e-10 before, 2.24e-10 after; layer 10: max abs(log - here) on reg 3.69e-10 before, 2.83e-10 after; layer 11: max abs(log - here) on reg 2.11e-10 before, 1.92e-10 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304').

## gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined9b, refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined9b = `models/refined/gu-refined-6-20-9b.gguf`, refined = `models/refined/gu-refined-6-20.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/refine-pilot-6-20.tsv`.
Grams: `models/grams/gu-6-20` (patched llama-imatrix on cand.gguf at P = 3, first 100 chunks of `data/calib-wt.txt`, pooled over all visits), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

### Layer 6 (IQ2_S; UD `.scale` present, min 3.867e-05, max 0.0001495)

- Gram layout: expert 23: max |strict lower| = 0, max |strict upper| = 1.83e+05
- Gram layout: expert 105: max |strict lower| = 0, max |strict upper| = 8.698e+04
- Gram layout: expert 53: max |strict lower| = 0, max |strict upper| = 7.821e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 5.84e-09; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.6.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.313552 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 2, median 5615, max 52992

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined9b | refined9b / ours | refined9b < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03091 | 0.01994 | 0.6452 | 128 of 128 | 0.01676 | 0.5422 | 96 of 128 | 0.01958 | 0.6337 | 96 of 128 |
| full | gate | count-weighted | 0.03025 | 0.01957 | 0.6468 | 128 of 128 | 0.01288 | 0.4257 | 96 of 128 | 0.01643 | 0.5431 | 96 of 128 |
| full | gate | trace-weighted | 0.03016 | 0.01951 | 0.6469 | 128 of 128 | 0.01287 | 0.4268 | 96 of 128 | 0.01642 | 0.5444 | 96 of 128 |
| full | up | unweighted | 0.04257 | 0.02732 | 0.6417 | 128 of 128 | 0.02298 | 0.5397 | 96 of 128 | 0.02570 | 0.6036 | 96 of 128 |
| full | up | count-weighted | 0.04167 | 0.02685 | 0.6443 | 128 of 128 | 0.01766 | 0.4239 | 96 of 128 | 0.02100 | 0.5041 | 96 of 128 |
| full | up | trace-weighted | 0.04166 | 0.02685 | 0.6446 | 128 of 128 | 0.01771 | 0.4250 | 96 of 128 | 0.02105 | 0.5052 | 96 of 128 |
| full | both | unweighted | 0.03557 | 0.02290 | 0.6437 | 128 of 128 | 0.01925 | 0.5412 | 96 of 128 | 0.02202 | 0.6191 | 96 of 128 |
| full | both | count-weighted | 0.03487 | 0.02252 | 0.6457 | 128 of 128 | 0.01482 | 0.4250 | 96 of 128 | 0.01827 | 0.5240 | 96 of 128 |
| full | both | trace-weighted | 0.03480 | 0.02248 | 0.6459 | 128 of 128 | 0.01483 | 0.4261 | 96 of 128 | 0.01828 | 0.5252 | 96 of 128 |
| reg | gate | unweighted | 0.03231 | 0.02207 | 0.6832 | 128 of 128 | 0.01981 | 0.6133 | 96 of 128 | 0.02249 | 0.6962 | 96 of 128 |
| reg | gate | count-weighted | 0.03169 | 0.02170 | 0.6848 | 128 of 128 | 0.01625 | 0.5128 | 96 of 128 | 0.01963 | 0.6195 | 96 of 128 |
| reg | gate | trace-weighted | 0.03159 | 0.02164 | 0.6849 | 128 of 128 | 0.01623 | 0.5138 | 96 of 128 | 0.01960 | 0.6205 | 96 of 128 |
| reg | up | unweighted | 0.04379 | 0.02978 | 0.6800 | 128 of 128 | 0.02676 | 0.6112 | 96 of 128 | 0.02933 | 0.6699 | 96 of 128 |
| reg | up | count-weighted | 0.04293 | 0.02930 | 0.6825 | 128 of 128 | 0.02195 | 0.5113 | 96 of 128 | 0.02512 | 0.5851 | 96 of 128 |
| reg | up | trace-weighted | 0.04293 | 0.02931 | 0.6828 | 128 of 128 | 0.02199 | 0.5122 | 96 of 128 | 0.02516 | 0.5860 | 96 of 128 |
| reg | both | unweighted | 0.03695 | 0.02519 | 0.6818 | 128 of 128 | 0.02263 | 0.6124 | 96 of 128 | 0.02525 | 0.6834 | 96 of 128 |
| reg | both | count-weighted | 0.03629 | 0.02481 | 0.6838 | 128 of 128 | 0.01859 | 0.5122 | 96 of 128 | 0.02187 | 0.6026 | 96 of 128 |
| reg | both | trace-weighted | 0.03622 | 0.02477 | 0.6840 | 128 of 128 | 0.01858 | 0.5131 | 96 of 128 | 0.02186 | 0.6036 | 96 of 128 |
| bias | gate | unweighted | 0.00621 | 0.00142 | 0.2294 | 128 of 128 | 0.00191 | 0.3072 | 96 of 128 | 0.00276 | 0.4449 | 96 of 128 |
| bias | gate | count-weighted | 0.00554 | 0.00112 | 0.2012 | 128 of 128 | 1.574e-04 | 0.0284 | 96 of 128 | 0.00124 | 0.2231 | 96 of 128 |
| bias | gate | trace-weighted | 0.00552 | 0.00110 | 0.1997 | 128 of 128 | 1.596e-04 | 0.0289 | 96 of 128 | 0.00124 | 0.2247 | 96 of 128 |
| bias | up | unweighted | 0.00877 | 0.00199 | 0.2267 | 128 of 128 | 0.00271 | 0.3086 | 96 of 128 | 0.00325 | 0.3711 | 96 of 128 |
| bias | up | count-weighted | 0.00775 | 0.00152 | 0.1964 | 128 of 128 | 2.144e-04 | 0.0277 | 96 of 128 | 8.749e-04 | 0.1129 | 96 of 128 |
| bias | up | trace-weighted | 0.00774 | 0.00151 | 0.1954 | 128 of 128 | 2.184e-04 | 0.0282 | 96 of 128 | 8.832e-04 | 0.1141 | 96 of 128 |
| bias | both | unweighted | 0.00722 | 0.00165 | 0.2281 | 128 of 128 | 0.00222 | 0.3079 | 96 of 128 | 0.00295 | 0.4084 | 96 of 128 |
| bias | both | count-weighted | 0.00643 | 0.00128 | 0.1988 | 128 of 128 | 1.805e-04 | 0.0281 | 96 of 128 | 0.00108 | 0.1685 | 96 of 128 |
| bias | both | trace-weighted | 0.00641 | 0.00127 | 0.1976 | 128 of 128 | 1.833e-04 | 0.0286 | 96 of 128 | 0.00109 | 0.1699 | 96 of 128 |
| rest | gate | unweighted | 0.02470 | 0.01852 | 0.7497 | 128 of 128 | 0.01485 | 0.6013 | 96 of 128 | 0.01682 | 0.6811 | 96 of 128 |
| rest | gate | count-weighted | 0.02471 | 0.01845 | 0.7468 | 128 of 128 | 0.01272 | 0.5148 | 96 of 128 | 0.01519 | 0.6150 | 96 of 128 |
| rest | gate | trace-weighted | 0.02464 | 0.01841 | 0.7471 | 128 of 128 | 0.01271 | 0.5159 | 96 of 128 | 0.01518 | 0.6160 | 96 of 128 |
| rest | up | unweighted | 0.03381 | 0.02533 | 0.7494 | 128 of 128 | 0.02027 | 0.5997 | 96 of 128 | 0.02245 | 0.6640 | 96 of 128 |
| rest | up | count-weighted | 0.03391 | 0.02532 | 0.7467 | 128 of 128 | 0.01745 | 0.5145 | 96 of 128 | 0.02013 | 0.5935 | 96 of 128 |
| rest | up | trace-weighted | 0.03392 | 0.02534 | 0.7471 | 128 of 128 | 0.01749 | 0.5155 | 96 of 128 | 0.02016 | 0.5944 | 96 of 128 |
| rest | both | unweighted | 0.02835 | 0.02125 | 0.7496 | 128 of 128 | 0.01703 | 0.6007 | 96 of 128 | 0.01907 | 0.6728 | 96 of 128 |
| rest | both | count-weighted | 0.02844 | 0.02124 | 0.7468 | 128 of 128 | 0.01464 | 0.5148 | 96 of 128 | 0.01719 | 0.6044 | 96 of 128 |
| rest | both | trace-weighted | 0.02839 | 0.02121 | 0.7471 | 128 of 128 | 0.01465 | 0.5158 | 96 of 128 | 0.01719 | 0.6054 | 96 of 128 |
| diag | gate | unweighted | 0.06175 | 0.06570 | 1.0640 | 0 of 128 | 0.08225 | 1.3320 | 0 of 128 | 0.08189 | 1.3261 | 0 of 128 |
| diag | gate | count-weighted | 0.06203 | 0.06604 | 1.0647 | 0 of 128 | 0.08591 | 1.3851 | 0 of 128 | 0.08572 | 1.3820 | 0 of 128 |
| diag | gate | trace-weighted | 0.06202 | 0.06603 | 1.0647 | 0 of 128 | 0.08585 | 1.3844 | 0 of 128 | 0.08565 | 1.3811 | 0 of 128 |
| diag | up | unweighted | 0.06189 | 0.06585 | 1.0640 | 0 of 128 | 0.08247 | 1.3325 | 0 of 128 | 0.08285 | 1.3387 | 0 of 128 |
| diag | up | count-weighted | 0.06211 | 0.06613 | 1.0647 | 0 of 128 | 0.08608 | 1.3859 | 0 of 128 | 0.08674 | 1.3964 | 0 of 128 |
| diag | up | trace-weighted | 0.06210 | 0.06612 | 1.0647 | 0 of 128 | 0.08602 | 1.3852 | 0 of 128 | 0.08666 | 1.3955 | 0 of 128 |
| diag | both | unweighted | 0.06182 | 0.06577 | 1.0640 | 0 of 128 | 0.08236 | 1.3322 | 0 of 128 | 0.08237 | 1.3324 | 0 of 128 |
| diag | both | count-weighted | 0.06207 | 0.06609 | 1.0647 | 0 of 128 | 0.08600 | 1.3855 | 0 of 128 | 0.08623 | 1.3892 | 0 of 128 |
| diag | both | trace-weighted | 0.06206 | 0.06608 | 1.0647 | 0 of 128 | 0.08594 | 1.3848 | 0 of 128 | 0.08616 | 1.3883 | 0 of 128 |
| plain | gate | unweighted | 0.26149 | 0.26828 | 1.0259 | 0 of 128 | 0.31497 | 1.2045 | 0 of 128 | 0.31162 | 1.1917 | 0 of 128 |
| plain | gate | count-weighted | 0.26139 | 0.26816 | 1.0259 | 0 of 128 | 0.32532 | 1.2446 | 0 of 128 | 0.32160 | 1.2304 | 0 of 128 |
| plain | gate | trace-weighted | 0.26144 | 0.26821 | 1.0259 | 0 of 128 | 0.32550 | 1.2450 | 0 of 128 | 0.32174 | 1.2306 | 0 of 128 |
| plain | up | unweighted | 0.26148 | 0.26826 | 1.0259 | 0 of 128 | 0.31489 | 1.2043 | 0 of 128 | 0.31366 | 1.1995 | 0 of 128 |
| plain | up | count-weighted | 0.26137 | 0.26813 | 1.0259 | 0 of 128 | 0.32524 | 1.2444 | 0 of 128 | 0.32407 | 1.2399 | 0 of 128 |
| plain | up | trace-weighted | 0.26142 | 0.26818 | 1.0258 | 0 of 128 | 0.32541 | 1.2448 | 0 of 128 | 0.32422 | 1.2402 | 0 of 128 |
| plain | both | unweighted | 0.26149 | 0.26827 | 1.0259 | 0 of 128 | 0.31493 | 1.2044 | 0 of 128 | 0.31264 | 1.1956 | 0 of 128 |
| plain | both | count-weighted | 0.26138 | 0.26814 | 1.0259 | 0 of 128 | 0.32528 | 1.2445 | 0 of 128 | 0.32284 | 1.2351 | 0 of 128 |
| plain | both | trace-weighted | 0.26143 | 0.26819 | 1.0259 | 0 of 128 | 0.32546 | 1.2449 | 0 of 128 | 0.32298 | 1.2354 | 0 of 128 |
| gain | gate | unweighted | 0.99629 | 0.99714 | 1.0009 | 9 of 128 | 0.98438 | 0.9880 | 96 of 128 | 0.98777 | 0.9914 | 96 of 128 |
| gain | gate | count-weighted | 0.99629 | 0.99721 | 1.0009 | 9 of 128 | 0.98177 | 0.9854 | 96 of 128 | 0.98615 | 0.9898 | 96 of 128 |
| gain | gate | trace-weighted | 0.99629 | 0.99720 | 1.0009 | 9 of 128 | 0.98174 | 0.9854 | 96 of 128 | 0.98613 | 0.9898 | 96 of 128 |
| gain | up | unweighted | 0.99630 | 0.99717 | 1.0009 | 11 of 128 | 0.98449 | 0.9881 | 96 of 128 | 0.98775 | 0.9914 | 96 of 128 |
| gain | up | count-weighted | 0.99633 | 0.99725 | 1.0009 | 11 of 128 | 0.98192 | 0.9855 | 96 of 128 | 0.98613 | 0.9898 | 96 of 128 |
| gain | up | trace-weighted | 0.99633 | 0.99725 | 1.0009 | 11 of 128 | 0.98190 | 0.9855 | 96 of 128 | 0.98612 | 0.9898 | 96 of 128 |
| gain | both | unweighted | 0.99629 | 0.99716 | 1.0009 | 11 of 128 | 0.98443 | 0.9881 | 96 of 128 | 0.98776 | 0.9914 | 96 of 128 |
| gain | both | count-weighted | 0.99631 | 0.99723 | 1.0009 | 11 of 128 | 0.98185 | 0.9855 | 96 of 128 | 0.98614 | 0.9898 | 96 of 128 |
| gain | both | trace-weighted | 0.99631 | 0.99722 | 1.0009 | 11 of 128 | 0.98182 | 0.9855 | 96 of 128 | 0.98613 | 0.9898 | 96 of 128 |
| energy | gate | unweighted | 1.06096 | 1.06626 | 1.0050 | 0 of 128 | 1.06898 | 1.0076 | 0 of 128 | 1.07353 | 1.0119 | 0 of 128 |
| energy | gate | count-weighted | 1.06091 | 1.06633 | 1.0051 | 0 of 128 | 1.06952 | 1.0081 | 0 of 128 | 1.07586 | 1.0141 | 0 of 128 |
| energy | gate | trace-weighted | 1.06093 | 1.06633 | 1.0051 | 0 of 128 | 1.06958 | 1.0082 | 0 of 128 | 1.07590 | 1.0141 | 0 of 128 |
| energy | up | unweighted | 1.06098 | 1.06631 | 1.0050 | 0 of 128 | 1.06914 | 1.0077 | 0 of 128 | 1.07484 | 1.0131 | 0 of 128 |
| energy | up | count-weighted | 1.06098 | 1.06640 | 1.0051 | 0 of 128 | 1.06978 | 1.0083 | 0 of 128 | 1.07743 | 1.0155 | 0 of 128 |
| energy | up | trace-weighted | 1.06101 | 1.06641 | 1.0051 | 0 of 128 | 1.06985 | 1.0083 | 0 of 128 | 1.07750 | 1.0155 | 0 of 128 |
| energy | both | unweighted | 1.06097 | 1.06629 | 1.0050 | 0 of 128 | 1.06906 | 1.0076 | 0 of 128 | 1.07418 | 1.0125 | 0 of 128 |
| energy | both | count-weighted | 1.06095 | 1.06637 | 1.0051 | 0 of 128 | 1.06965 | 1.0082 | 0 of 128 | 1.07665 | 1.0148 | 0 of 128 |
| energy | both | trace-weighted | 1.06097 | 1.06637 | 1.0051 | 0 of 128 | 1.06972 | 1.0082 | 0 of 128 | 1.07670 | 1.0148 | 0 of 128 |
| ogain | gate | unweighted | 0.99919 | 0.99910 | 0.9999 | 66 of 128 | 0.99712 | 0.9979 | 83 of 128 | 1.01126 | 1.0121 | 0 of 128 |
| ogain | gate | count-weighted | 0.99903 | 0.99900 | 1.0000 | 66 of 128 | 0.99624 | 0.9972 | 83 of 128 | 1.01417 | 1.0152 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99899 | 0.99900 | 1.0000 | 66 of 128 | 0.99624 | 0.9972 | 83 of 128 | 1.01414 | 1.0152 | 0 of 128 |
| ogain | up | unweighted | 0.99580 | 0.99860 | 1.0028 | 3 of 128 | 0.99511 | 0.9993 | 73 of 128 | 1.00705 | 1.0113 | 0 of 128 |
| ogain | up | count-weighted | 0.99542 | 0.99851 | 1.0031 | 3 of 128 | 0.99480 | 0.9994 | 73 of 128 | 1.00956 | 1.0142 | 0 of 128 |
| ogain | up | trace-weighted | 0.99542 | 0.99852 | 1.0031 | 3 of 128 | 0.99479 | 0.9994 | 73 of 128 | 1.00955 | 1.0142 | 0 of 128 |
| ogain | both | unweighted | 0.99776 | 0.99890 | 1.0011 | 30 of 128 | 0.99629 | 0.9985 | 83 of 128 | 1.00950 | 1.0118 | 0 of 128 |
| ogain | both | count-weighted | 0.99750 | 0.99879 | 1.0013 | 30 of 128 | 0.99565 | 0.9981 | 83 of 128 | 1.01223 | 1.0148 | 0 of 128 |
| ogain | both | trace-weighted | 0.99747 | 0.99880 | 1.0013 | 30 of 128 | 0.99565 | 0.9982 | 83 of 128 | 1.01221 | 1.0148 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.0685e-02 | [-1.1155e-02, -1.0230e-02] |
| up | -1.4819e-02 | [-1.5154e-02, -1.4490e-02] |
| both | -1.2356e-02 | [-1.2757e-02, -1.1974e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined9b minus ours and refined9b minus UD:

| metric | half | refined9b - ours | 95% CI | refined9b - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.7374e-02 | [-1.8067e-02, -1.6679e-02] | -6.6891e-03 | [-7.1000e-03, -6.2996e-03] |
| full | up | -2.4002e-02 | [-2.4555e-02, -2.3465e-02] | -9.1831e-03 | [-9.6737e-03, -8.7351e-03] |
| full | both | -2.0052e-02 | [-2.0659e-02, -1.9442e-02] | -7.6968e-03 | [-8.1329e-03, -7.2918e-03] |
| reg | gate | -1.5437e-02 | [-1.6034e-02, -1.4841e-02] | -5.4496e-03 | [-5.8003e-03, -5.1173e-03] |
| reg | up | -2.0984e-02 | [-2.1442e-02, -2.0531e-02] | -7.3521e-03 | [-7.7688e-03, -6.9649e-03] |
| reg | both | -1.7703e-02 | [-1.8211e-02, -1.7187e-02] | -6.2268e-03 | [-6.5987e-03, -5.8798e-03] |
| bias | gate | -5.3868e-03 | [-5.6523e-03, -5.1409e-03] | -9.5801e-04 | [-1.0533e-03, -8.5927e-04] |
| bias | up | -7.5384e-03 | [-7.7781e-03, -7.2994e-03] | -1.3081e-03 | [-1.4219e-03, -1.1885e-03] |
| bias | both | -6.2536e-03 | [-6.4929e-03, -6.0282e-03] | -1.0989e-03 | [-1.1995e-03, -9.9219e-04] |
| rest | gate | -1.1987e-02 | [-1.2460e-02, -1.1503e-02] | -5.7311e-03 | [-6.0701e-03, -5.4165e-03] |
| rest | up | -1.6463e-02 | [-1.6877e-02, -1.6086e-02] | -7.8750e-03 | [-8.2766e-03, -7.5104e-03] |
| rest | both | -1.3799e-02 | [-1.4220e-02, -1.3373e-02] | -6.5978e-03 | [-6.9563e-03, -6.2696e-03] |
| diag | gate | +2.3885e-02 | [+2.3239e-02, +2.4651e-02] | +1.9871e-02 | [+1.9228e-02, +2.0638e-02] |
| diag | up | +2.3968e-02 | [+2.3312e-02, +2.4750e-02] | +1.9950e-02 | [+1.9297e-02, +2.0731e-02] |
| diag | both | +2.3926e-02 | [+2.3275e-02, +2.4701e-02] | +1.9910e-02 | [+1.9263e-02, +2.0687e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.3820e-02 | [-1.4416e-02, -1.3215e-02] | -3.1357e-03 | [-3.5073e-03, -2.7802e-03] |
| full | up | -2.0663e-02 | [-2.1182e-02, -2.0176e-02] | -5.8444e-03 | [-6.3239e-03, -5.3800e-03] |
| full | both | -1.6599e-02 | [-1.7121e-02, -1.6079e-02] | -4.2434e-03 | [-4.6511e-03, -3.8575e-03] |
| reg | gate | -1.2058e-02 | [-1.2558e-02, -1.1545e-02] | -2.0703e-03 | [-2.3920e-03, -1.7613e-03] |
| reg | up | -1.7814e-02 | [-1.8238e-02, -1.7411e-02] | -4.1822e-03 | [-4.5895e-03, -3.7790e-03] |
| reg | both | -1.4421e-02 | [-1.4855e-02, -1.3986e-02] | -2.9451e-03 | [-3.2972e-03, -2.6093e-03] |
| bias | gate | -4.3074e-03 | [-4.5334e-03, -4.0896e-03] | +1.2140e-04 | [+2.5425e-05, +2.2062e-04] |
| bias | up | -6.8779e-03 | [-7.0784e-03, -6.6805e-03] | -6.4755e-04 | [-7.6729e-04, -5.1602e-04] |
| bias | both | -5.3500e-03 | [-5.5451e-03, -5.1611e-03] | -1.9524e-04 | [-2.9760e-04, -8.5105e-05] |
| rest | gate | -9.5129e-03 | [-9.9154e-03, -9.1051e-03] | -3.2571e-03 | [-3.5579e-03, -2.9852e-03] |
| rest | up | -1.3785e-02 | [-1.4175e-02, -1.3425e-02] | -5.1968e-03 | [-5.5757e-03, -4.8414e-03] |
| rest | both | -1.1249e-02 | [-1.1618e-02, -1.0877e-02] | -4.0482e-03 | [-4.3743e-03, -3.7493e-03] |
| diag | gate | +2.3696e-02 | [+2.3088e-02, +2.4409e-02] | +1.9682e-02 | [+1.9078e-02, +2.0394e-02] |
| diag | up | +2.4621e-02 | [+2.3962e-02, +2.5377e-02] | +2.0603e-02 | [+1.9953e-02, +2.1360e-02] |
| diag | both | +2.4158e-02 | [+2.3527e-02, +2.4891e-02] | +2.0142e-02 | [+1.9516e-02, +2.0878e-02] |

### Layer 20 (IQ3_XXS; UD `.scale` absent)

- Gram layout: expert 102: max |strict lower| = 0, max |strict upper| = 3.719e+04
- Gram layout: expert 40: max |strict lower| = 0, max |strict upper| = 3.206e+04
- Gram layout: expert 120: max |strict lower| = 0, max |strict upper| = 3.096e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 4.16e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.20.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = -0.000000 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 1, median 1797, max 20118

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined9b | refined9b / ours | refined9b < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.02301 | 0.01952 | 0.8485 | 128 of 128 | 0.01750 | 0.7604 | 47 of 128 | 0.01815 | 0.7890 | 47 of 128 |
| full | gate | count-weighted | 0.02165 | 0.01794 | 0.8284 | 128 of 128 | 0.01008 | 0.4658 | 47 of 128 | 0.01151 | 0.5314 | 47 of 128 |
| full | gate | trace-weighted | 0.02178 | 0.01811 | 0.8314 | 128 of 128 | 0.01044 | 0.4791 | 47 of 128 | 0.01183 | 0.5430 | 47 of 128 |
| full | up | unweighted | 0.02917 | 0.02471 | 0.8470 | 128 of 128 | 0.02181 | 0.7476 | 47 of 128 | 0.02242 | 0.7687 | 47 of 128 |
| full | up | count-weighted | 0.02896 | 0.02397 | 0.8277 | 128 of 128 | 0.01327 | 0.4582 | 47 of 128 | 0.01461 | 0.5044 | 47 of 128 |
| full | up | trace-weighted | 0.02903 | 0.02411 | 0.8306 | 128 of 128 | 0.01366 | 0.4706 | 47 of 128 | 0.01497 | 0.5157 | 47 of 128 |
| full | both | unweighted | 0.02565 | 0.02175 | 0.8479 | 128 of 128 | 0.01936 | 0.7548 | 47 of 128 | 0.02000 | 0.7797 | 47 of 128 |
| full | both | count-weighted | 0.02472 | 0.02047 | 0.8281 | 128 of 128 | 0.01143 | 0.4624 | 47 of 128 | 0.01282 | 0.5185 | 47 of 128 |
| full | both | trace-weighted | 0.02483 | 0.02064 | 0.8311 | 128 of 128 | 0.01180 | 0.4753 | 47 of 128 | 0.01316 | 0.5300 | 47 of 128 |
| reg | gate | unweighted | 0.02364 | 0.02053 | 0.8687 | 128 of 128 | 0.01888 | 0.7987 | 47 of 128 | 0.01948 | 0.8242 | 47 of 128 |
| reg | gate | count-weighted | 0.02233 | 0.01899 | 0.8506 | 128 of 128 | 0.01230 | 0.5507 | 47 of 128 | 0.01360 | 0.6092 | 47 of 128 |
| reg | gate | trace-weighted | 0.02245 | 0.01916 | 0.8533 | 128 of 128 | 0.01262 | 0.5619 | 47 of 128 | 0.01390 | 0.6189 | 47 of 128 |
| reg | up | unweighted | 0.02947 | 0.02557 | 0.8675 | 128 of 128 | 0.02324 | 0.7887 | 47 of 128 | 0.02380 | 0.8076 | 47 of 128 |
| reg | up | count-weighted | 0.02927 | 0.02488 | 0.8500 | 128 of 128 | 0.01595 | 0.5447 | 47 of 128 | 0.01716 | 0.5861 | 47 of 128 |
| reg | up | trace-weighted | 0.02934 | 0.02501 | 0.8526 | 128 of 128 | 0.01629 | 0.5552 | 47 of 128 | 0.01747 | 0.5956 | 47 of 128 |
| reg | both | unweighted | 0.02617 | 0.02272 | 0.8682 | 128 of 128 | 0.02078 | 0.7943 | 47 of 128 | 0.02137 | 0.8165 | 47 of 128 |
| reg | both | count-weighted | 0.02528 | 0.02150 | 0.8503 | 128 of 128 | 0.01386 | 0.5480 | 47 of 128 | 0.01512 | 0.5981 | 47 of 128 |
| reg | both | trace-weighted | 0.02539 | 0.02166 | 0.8530 | 128 of 128 | 0.01419 | 0.5589 | 47 of 128 | 0.01543 | 0.6077 | 47 of 128 |
| bias | gate | unweighted | 0.00426 | 0.00206 | 0.4831 | 128 of 128 | 0.00284 | 0.6670 | 47 of 128 | 0.00301 | 0.7053 | 47 of 128 |
| bias | gate | count-weighted | 0.00368 | 0.00149 | 0.4043 | 128 of 128 | 6.751e-04 | 0.1834 | 47 of 128 | 0.00104 | 0.2819 | 47 of 128 |
| bias | gate | trace-weighted | 0.00368 | 0.00150 | 0.4067 | 128 of 128 | 7.331e-04 | 0.1994 | 47 of 128 | 0.00109 | 0.2956 | 47 of 128 |
| bias | up | unweighted | 0.00544 | 0.00262 | 0.4817 | 128 of 128 | 0.00356 | 0.6543 | 47 of 128 | 0.00364 | 0.6695 | 47 of 128 |
| bias | up | count-weighted | 0.00490 | 0.00199 | 0.4061 | 128 of 128 | 8.607e-04 | 0.1758 | 47 of 128 | 0.00103 | 0.2112 | 47 of 128 |
| bias | up | trace-weighted | 0.00488 | 0.00199 | 0.4084 | 128 of 128 | 9.308e-04 | 0.1909 | 47 of 128 | 0.00110 | 0.2256 | 47 of 128 |
| bias | both | unweighted | 0.00477 | 0.00230 | 0.4824 | 128 of 128 | 0.00315 | 0.6614 | 47 of 128 | 0.00328 | 0.6886 | 47 of 128 |
| bias | both | count-weighted | 0.00419 | 0.00170 | 0.4053 | 128 of 128 | 7.555e-04 | 0.1802 | 47 of 128 | 0.00104 | 0.2478 | 47 of 128 |
| bias | both | trace-weighted | 0.00418 | 0.00170 | 0.4076 | 128 of 128 | 8.189e-04 | 0.1959 | 47 of 128 | 0.00110 | 0.2620 | 47 of 128 |
| rest | gate | unweighted | 0.01875 | 0.01746 | 0.9317 | 124 of 128 | 0.01465 | 0.7816 | 47 of 128 | 0.01515 | 0.8080 | 47 of 128 |
| rest | gate | count-weighted | 0.01797 | 0.01645 | 0.9153 | 124 of 128 | 0.00941 | 0.5236 | 47 of 128 | 0.01047 | 0.5825 | 47 of 128 |
| rest | gate | trace-weighted | 0.01811 | 0.01662 | 0.9177 | 124 of 128 | 0.00970 | 0.5359 | 47 of 128 | 0.01074 | 0.5932 | 47 of 128 |
| rest | up | unweighted | 0.02373 | 0.02208 | 0.9308 | 124 of 128 | 0.01825 | 0.7690 | 47 of 128 | 0.01878 | 0.7915 | 47 of 128 |
| rest | up | count-weighted | 0.02406 | 0.02198 | 0.9135 | 124 of 128 | 0.01241 | 0.5157 | 47 of 128 | 0.01357 | 0.5641 | 47 of 128 |
| rest | up | trace-weighted | 0.02415 | 0.02212 | 0.9158 | 124 of 128 | 0.01273 | 0.5270 | 47 of 128 | 0.01387 | 0.5742 | 47 of 128 |
| rest | both | unweighted | 0.02088 | 0.01945 | 0.9313 | 124 of 128 | 0.01621 | 0.7760 | 47 of 128 | 0.01672 | 0.8005 | 47 of 128 |
| rest | both | count-weighted | 0.02053 | 0.01877 | 0.9145 | 124 of 128 | 0.01068 | 0.5200 | 47 of 128 | 0.01178 | 0.5738 | 47 of 128 |
| rest | both | trace-weighted | 0.02065 | 0.01893 | 0.9168 | 124 of 128 | 0.01098 | 0.5319 | 47 of 128 | 0.01206 | 0.5842 | 47 of 128 |
| diag | gate | unweighted | 0.03304 | 0.03546 | 1.0733 | 0 of 128 | 0.03966 | 1.2003 | 0 of 128 | 0.03943 | 1.1932 | 0 of 128 |
| diag | gate | count-weighted | 0.03299 | 0.03541 | 1.0733 | 0 of 128 | 0.04661 | 1.4130 | 0 of 128 | 0.04614 | 1.3987 | 0 of 128 |
| diag | gate | trace-weighted | 0.03299 | 0.03541 | 1.0733 | 0 of 128 | 0.04634 | 1.4046 | 0 of 128 | 0.04588 | 1.3906 | 0 of 128 |
| diag | up | unweighted | 0.03305 | 0.03548 | 1.0734 | 0 of 128 | 0.03968 | 1.2003 | 0 of 128 | 0.03956 | 1.1968 | 0 of 128 |
| diag | up | count-weighted | 0.03301 | 0.03543 | 1.0734 | 0 of 128 | 0.04664 | 1.4130 | 0 of 128 | 0.04642 | 1.4062 | 0 of 128 |
| diag | up | trace-weighted | 0.03301 | 0.03543 | 1.0734 | 0 of 128 | 0.04636 | 1.4046 | 0 of 128 | 0.04614 | 1.3979 | 0 of 128 |
| diag | both | unweighted | 0.03305 | 0.03547 | 1.0733 | 0 of 128 | 0.03967 | 1.2003 | 0 of 128 | 0.03949 | 1.1950 | 0 of 128 |
| diag | both | count-weighted | 0.03300 | 0.03542 | 1.0734 | 0 of 128 | 0.04663 | 1.4130 | 0 of 128 | 0.04628 | 1.4024 | 0 of 128 |
| diag | both | trace-weighted | 0.03300 | 0.03542 | 1.0734 | 0 of 128 | 0.04635 | 1.4046 | 0 of 128 | 0.04601 | 1.3942 | 0 of 128 |
| plain | gate | unweighted | 0.18317 | 0.18961 | 1.0352 | 0 of 128 | 0.20020 | 1.0930 | 0 of 128 | 0.19995 | 1.0916 | 0 of 128 |
| plain | gate | count-weighted | 0.18302 | 0.18952 | 1.0355 | 0 of 128 | 0.21817 | 1.1921 | 0 of 128 | 0.21768 | 1.1894 | 0 of 128 |
| plain | gate | trace-weighted | 0.18302 | 0.18952 | 1.0355 | 0 of 128 | 0.21746 | 1.1882 | 0 of 128 | 0.21697 | 1.1855 | 0 of 128 |
| plain | up | unweighted | 0.18316 | 0.18962 | 1.0353 | 0 of 128 | 0.20019 | 1.0930 | 0 of 128 | 0.20014 | 1.0927 | 0 of 128 |
| plain | up | count-weighted | 0.18302 | 0.18953 | 1.0356 | 0 of 128 | 0.21818 | 1.1921 | 0 of 128 | 0.21811 | 1.1917 | 0 of 128 |
| plain | up | trace-weighted | 0.18302 | 0.18953 | 1.0356 | 0 of 128 | 0.21746 | 1.1882 | 0 of 128 | 0.21739 | 1.1878 | 0 of 128 |
| plain | both | unweighted | 0.18316 | 0.18961 | 1.0352 | 0 of 128 | 0.20020 | 1.0930 | 0 of 128 | 0.20004 | 1.0922 | 0 of 128 |
| plain | both | count-weighted | 0.18302 | 0.18953 | 1.0356 | 0 of 128 | 0.21818 | 1.1921 | 0 of 128 | 0.21789 | 1.1905 | 0 of 128 |
| plain | both | trace-weighted | 0.18302 | 0.18953 | 1.0356 | 0 of 128 | 0.21746 | 1.1882 | 0 of 128 | 0.21718 | 1.1866 | 0 of 128 |
| gain | gate | unweighted | 0.99559 | 0.99579 | 1.0002 | 16 of 128 | 0.99157 | 0.9960 | 47 of 128 | 0.99288 | 0.9973 | 47 of 128 |
| gain | gate | count-weighted | 0.99566 | 0.99586 | 1.0002 | 16 of 128 | 0.98702 | 0.9913 | 47 of 128 | 0.98996 | 0.9943 | 47 of 128 |
| gain | gate | trace-weighted | 0.99566 | 0.99586 | 1.0002 | 16 of 128 | 0.98721 | 0.9915 | 47 of 128 | 0.99008 | 0.9944 | 47 of 128 |
| gain | up | unweighted | 0.99557 | 0.99578 | 1.0002 | 16 of 128 | 0.99155 | 0.9960 | 47 of 128 | 0.99284 | 0.9973 | 47 of 128 |
| gain | up | count-weighted | 0.99565 | 0.99583 | 1.0002 | 16 of 128 | 0.98701 | 0.9913 | 47 of 128 | 0.98992 | 0.9943 | 47 of 128 |
| gain | up | trace-weighted | 0.99565 | 0.99583 | 1.0002 | 16 of 128 | 0.98719 | 0.9915 | 47 of 128 | 0.99004 | 0.9944 | 47 of 128 |
| gain | both | unweighted | 0.99558 | 0.99578 | 1.0002 | 12 of 128 | 0.99156 | 0.9960 | 47 of 128 | 0.99286 | 0.9973 | 47 of 128 |
| gain | both | count-weighted | 0.99565 | 0.99584 | 1.0002 | 12 of 128 | 0.98701 | 0.9913 | 47 of 128 | 0.98994 | 0.9943 | 47 of 128 |
| gain | both | trace-weighted | 0.99566 | 0.99584 | 1.0002 | 12 of 128 | 0.98720 | 0.9915 | 47 of 128 | 0.99006 | 0.9944 | 47 of 128 |
| energy | gate | unweighted | 1.02473 | 1.02753 | 1.0027 | 0 of 128 | 1.02373 | 0.9990 | 41 of 128 | 1.02623 | 1.0015 | 0 of 128 |
| energy | gate | count-weighted | 1.02482 | 1.02763 | 1.0027 | 0 of 128 | 1.02195 | 0.9972 | 41 of 128 | 1.02759 | 1.0027 | 0 of 128 |
| energy | gate | trace-weighted | 1.02482 | 1.02763 | 1.0027 | 0 of 128 | 1.02203 | 0.9973 | 41 of 128 | 1.02754 | 1.0027 | 0 of 128 |
| energy | up | unweighted | 1.02468 | 1.02752 | 1.0028 | 0 of 128 | 1.02368 | 0.9990 | 40 of 128 | 1.02624 | 1.0015 | 0 of 128 |
| energy | up | count-weighted | 1.02479 | 1.02758 | 1.0027 | 0 of 128 | 1.02192 | 0.9972 | 40 of 128 | 1.02772 | 1.0029 | 0 of 128 |
| energy | up | trace-weighted | 1.02479 | 1.02758 | 1.0027 | 0 of 128 | 1.02199 | 0.9973 | 40 of 128 | 1.02766 | 1.0028 | 0 of 128 |
| energy | both | unweighted | 1.02470 | 1.02752 | 1.0028 | 0 of 128 | 1.02371 | 0.9990 | 40 of 128 | 1.02624 | 1.0015 | 0 of 128 |
| energy | both | count-weighted | 1.02480 | 1.02760 | 1.0027 | 0 of 128 | 1.02193 | 0.9972 | 40 of 128 | 1.02765 | 1.0028 | 0 of 128 |
| energy | both | trace-weighted | 1.02481 | 1.02761 | 1.0027 | 0 of 128 | 1.02201 | 0.9973 | 40 of 128 | 1.02760 | 1.0027 | 0 of 128 |
| ogain | gate | unweighted | 0.99686 | 0.99740 | 1.0005 | 37 of 128 | 0.99667 | 0.9998 | 29 of 128 | 1.00175 | 1.0049 | 0 of 128 |
| ogain | gate | count-weighted | 0.99682 | 0.99754 | 1.0007 | 37 of 128 | 0.99647 | 0.9997 | 29 of 128 | 1.00765 | 1.0109 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99682 | 0.99751 | 1.0007 | 37 of 128 | 0.99647 | 0.9996 | 29 of 128 | 1.00739 | 1.0106 | 0 of 128 |
| ogain | up | unweighted | 0.99616 | 0.99654 | 1.0004 | 35 of 128 | 0.99579 | 0.9996 | 37 of 128 | 1.00012 | 1.0040 | 0 of 128 |
| ogain | up | count-weighted | 0.99620 | 0.99668 | 1.0005 | 35 of 128 | 0.99538 | 0.9992 | 37 of 128 | 1.00486 | 1.0087 | 0 of 128 |
| ogain | up | trace-weighted | 0.99620 | 0.99667 | 1.0005 | 35 of 128 | 0.99539 | 0.9992 | 37 of 128 | 1.00465 | 1.0085 | 0 of 128 |
| ogain | both | unweighted | 0.99655 | 0.99703 | 1.0005 | 31 of 128 | 0.99629 | 0.9997 | 36 of 128 | 1.00105 | 1.0045 | 0 of 128 |
| ogain | both | count-weighted | 0.99656 | 0.99717 | 1.0006 | 31 of 128 | 0.99601 | 0.9994 | 36 of 128 | 1.00646 | 1.0099 | 0 of 128 |
| ogain | both | trace-weighted | 0.99656 | 0.99716 | 1.0006 | 31 of 128 | 0.99601 | 0.9995 | 36 of 128 | 1.00623 | 1.0097 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -3.7147e-03 | [-3.8593e-03, -3.5630e-03] |
| up | -4.9896e-03 | [-5.1732e-03, -4.7856e-03] |
| both | -4.2483e-03 | [-4.3965e-03, -4.0870e-03] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined9b minus ours and refined9b minus UD:

| metric | half | refined9b - ours | 95% CI | refined9b - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.1566e-02 | [-1.2439e-02, -1.0481e-02] | -7.8515e-03 | [-8.6742e-03, -6.8174e-03] |
| full | up | -1.5689e-02 | [-1.6752e-02, -1.4278e-02] | -1.0700e-02 | [-1.1731e-02, -9.3731e-03] |
| full | both | -1.3289e-02 | [-1.4233e-02, -1.2075e-02] | -9.0404e-03 | [-9.9382e-03, -7.8837e-03] |
| reg | gate | -1.0031e-02 | [-1.0778e-02, -9.0913e-03] | -6.6946e-03 | [-7.3964e-03, -5.8049e-03] |
| reg | up | -1.3329e-02 | [-1.4228e-02, -1.2127e-02] | -8.9362e-03 | [-9.8034e-03, -7.8167e-03] |
| reg | both | -1.1427e-02 | [-1.2233e-02, -1.0385e-02] | -7.6425e-03 | [-8.4022e-03, -6.6521e-03] |
| bias | gate | -3.0059e-03 | [-3.2848e-03, -2.6879e-03] | -8.1315e-04 | [-1.0598e-03, -5.1949e-04] |
| bias | up | -4.0363e-03 | [-4.3994e-03, -3.6258e-03] | -1.1280e-03 | [-1.4560e-03, -7.4323e-04] |
| bias | both | -3.4364e-03 | [-3.7484e-03, -3.0825e-03] | -9.4349e-04 | [-1.2243e-03, -6.1266e-04] |
| rest | gate | -8.5602e-03 | [-9.2123e-03, -7.7609e-03] | -7.0383e-03 | [-7.6704e-03, -6.2714e-03] |
| rest | up | -1.1653e-02 | [-1.2445e-02, -1.0613e-02] | -9.5718e-03 | [-1.0345e-02, -8.5850e-03] |
| rest | both | -9.8523e-03 | [-1.0554e-02, -8.9557e-03] | -8.0970e-03 | [-8.7787e-03, -7.2394e-03] |
| diag | gate | +1.3624e-02 | [+1.2453e-02, +1.4525e-02] | +1.1205e-02 | [+1.0030e-02, +1.2111e-02] |
| diag | up | +1.3632e-02 | [+1.2460e-02, +1.4542e-02] | +1.1207e-02 | [+1.0029e-02, +1.2119e-02] |
| diag | both | +1.3628e-02 | [+1.2456e-02, +1.4533e-02] | +1.1206e-02 | [+1.0027e-02, +1.2114e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.0145e-02 | [-1.0933e-02, -9.1827e-03] | -6.4308e-03 | [-7.1597e-03, -5.5283e-03] |
| full | up | -1.4351e-02 | [-1.5325e-02, -1.3066e-02] | -9.3613e-03 | [-1.0305e-02, -8.1511e-03] |
| full | both | -1.1903e-02 | [-1.2755e-02, -1.0811e-02] | -7.6546e-03 | [-8.4662e-03, -6.6264e-03] |
| reg | gate | -8.7251e-03 | [-9.3951e-03, -7.9036e-03] | -5.3883e-03 | [-6.0041e-03, -4.6170e-03] |
| reg | up | -1.2116e-02 | [-1.2935e-02, -1.1028e-02] | -7.7233e-03 | [-8.5116e-03, -6.7105e-03] |
| reg | both | -1.0160e-02 | [-1.0882e-02, -9.2317e-03] | -6.3762e-03 | [-7.0603e-03, -5.5027e-03] |
| bias | gate | -2.6433e-03 | [-2.8930e-03, -2.3613e-03] | -4.5055e-04 | [-6.6536e-04, -1.9505e-04] |
| bias | up | -3.8630e-03 | [-4.2046e-03, -3.4776e-03] | -9.5460e-04 | [-1.2595e-03, -5.9308e-04] |
| bias | both | -3.1530e-03 | [-3.4366e-03, -2.8293e-03] | -6.6005e-04 | [-9.1172e-04, -3.6137e-04] |
| rest | gate | -7.5022e-03 | [-8.0890e-03, -6.7965e-03] | -5.9802e-03 | [-6.5468e-03, -5.3059e-03] |
| rest | up | -1.0488e-02 | [-1.1214e-02, -9.5447e-03] | -8.4067e-03 | [-9.1117e-03, -7.5199e-03] |
| rest | both | -8.7499e-03 | [-9.3843e-03, -7.9502e-03] | -6.9946e-03 | [-7.6088e-03, -6.2343e-03] |
| diag | gate | +1.3152e-02 | [+1.2026e-02, +1.4028e-02] | +1.0734e-02 | [+9.6000e-03, +1.1610e-02] |
| diag | up | +1.3408e-02 | [+1.2255e-02, +1.4303e-02] | +1.0984e-02 | [+9.8234e-03, +1.1882e-02] |
| diag | both | +1.3280e-02 | [+1.2140e-02, +1.4162e-02] | +1.0859e-02 | [+9.7136e-03, +1.1745e-02] |

### Reading (amendment 7 rule, applied literally)

Rule: supported if UD is lower than ours on the full error and higher or equal on the diagonal one in both layers; if UD is also worse on the full error, layer-local output error does not explain UD's edge.

| mean | half | layer 6 full UD < ours | layer 6 diag UD >= ours | layer 20 full UD < ours | layer 20 diag UD >= ours | branch |
|---|---|---|---|---|---|---|
| unweighted | gate | yes | yes | yes | yes | supported |
| unweighted | up | yes | yes | yes | yes | supported |
| unweighted | both | yes | yes | yes | yes | supported |
| count-weighted | gate | yes | yes | yes | yes | supported |
| count-weighted | up | yes | yes | yes | yes | supported |
| count-weighted | both | yes | yes | yes | yes | supported |
| trace-weighted | gate | yes | yes | yes | yes | supported |
| trace-weighted | up | yes | yes | yes | yes | supported |
| trace-weighted | both | yes | yes | yes | yes | supported |

In layer 6 (both halves, count-weighted), UD's full error is 0.02252 against our 0.03487 (ratio 0.646; UD - ours -1.236e-02, 95% CI [-1.276e-02, -1.197e-02]) and its diagonal error 0.06609 against our 0.06207 (ratio 1.065); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 6 (added after registration, not part of the rule): the input-mean part is 0.00643 ours against 0.00128 UD (ratio 0.199), the centred rest 0.02844 against 0.02124 (ratio 0.747). In layer 20 (both halves, count-weighted), UD's full error is 0.02047 against our 0.02472 (ratio 0.828; UD - ours -4.248e-03, 95% CI [-4.396e-03, -4.087e-03]) and its diagonal error 0.03542 against our 0.03300 (ratio 1.073); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 20 (added after registration, not part of the rule): the input-mean part is 0.00419 ours against 0.00170 UD (ratio 0.405), the centred rest 0.02053 against 0.01877 (ratio 0.914). Under the registered rule this is "supported" for both halves together, count-weighted; across the three means and three halves the branches are: supported.

### refined9b against ours and UD

In layer 6 (both halves, count-weighted), refined9b's full error is 0.01482 against our 0.03487 and UD's 0.02252 (refined9b - ours -2.005e-02, 95% CI [-2.066e-02, -1.944e-02]; (ours - refined9b) / (ours - UD) = 1.623; lower than ours in 96 of 128 experts). Bias part 1.805e-04 (ours 0.00643, UD 0.00128), rest 0.01464 (ours 0.02844, UD 0.02124). Regularised (the search objective) 0.01859 (ours 0.03629, UD 0.02481). Diagonal 0.08600 (ours 0.06207, UD 0.06609), plain 0.32528 (ours 0.26138, UD 0.26814). Gains (ours / UD / refined9b): plain 0.9963 / 0.9972 / 0.9818, energy 1.0609 / 1.0664 / 1.0697, output 0.9975 / 0.9988 / 0.9956. In layer 20 (both halves, count-weighted), refined9b's full error is 0.01143 against our 0.02472 and UD's 0.02047 (refined9b - ours -1.329e-02, 95% CI [-1.423e-02, -1.207e-02]; (ours - refined9b) / (ours - UD) = 3.128; lower than ours in 47 of 128 experts). Bias part 7.555e-04 (ours 0.00419, UD 0.00170), rest 0.01068 (ours 0.02053, UD 0.01877). Regularised (the search objective) 0.01386 (ours 0.02528, UD 0.02150). Diagonal 0.04663 (ours 0.03300, UD 0.03542), plain 0.21818 (ours 0.18302, UD 0.18953). Gains (ours / UD / refined9b): plain 0.9957 / 0.9958 / 0.9870, energy 1.0248 / 1.0276 / 1.0219, output 0.9966 / 0.9972 / 0.9960.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined9b (count-w.) | refined9b / ours | refined9b / UD |
|---|---|---|---|---|---|---|---|---|
| 6 | < 2304 | 32 | 0.021 | 0.03889 | 0.02507 | 0.03889 | 1.000 | 1.551 |
| 6 | >= 2304 | 96 | 0.979 | 0.03479 | 0.02246 | 0.01430 | 0.411 | 0.637 |
| 20 | < 2304 | 81 | 0.184 | 0.02641 | 0.02288 | 0.02641 | 1.000 | 1.154 |
| 20 | >= 2304 | 47 | 0.816 | 0.02434 | 0.01993 | 0.00804 | 0.331 | 0.404 |

Refinement log `models/refined/gu-refined-6-20-9b.gguf.json` (scripts/refine-codes.py; wall time 1182 s for 256 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 96 | 32 | 4: 96 | 50681789 | 1990652 of 1990656 | 2755 | 4.86e-10 | 4.88e-10 | yes |
| 20 | 47 | 81 | 4: 47 | 34840912 | 974592 of 974592 | 1674 | 4.84e-10 | 2.75e-10 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 6: max abs(log - here) on reg 4.67e-10 before, 4.53e-10 after; layer 20: max abs(log - here) on reg 4.50e-10 before, 2.43e-10 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304').

### refined against ours and UD

In layer 6 (both halves, count-weighted), refined's full error is 0.01827 against our 0.03487 and UD's 0.02252 (refined - ours -1.660e-02, 95% CI [-1.712e-02, -1.608e-02]; (ours - refined) / (ours - UD) = 1.343; lower than ours in 96 of 128 experts). Bias part 0.00108 (ours 0.00643, UD 0.00128), rest 0.01719 (ours 0.02844, UD 0.02124). Regularised (the search objective) 0.02187 (ours 0.03629, UD 0.02481). Diagonal 0.08623 (ours 0.06207, UD 0.06609), plain 0.32284 (ours 0.26138, UD 0.26814). Gains (ours / UD / refined): plain 0.9963 / 0.9972 / 0.9861, energy 1.0609 / 1.0664 / 1.0766, output 0.9975 / 0.9988 / 1.0122. In layer 20 (both halves, count-weighted), refined's full error is 0.01282 against our 0.02472 and UD's 0.02047 (refined - ours -1.190e-02, 95% CI [-1.275e-02, -1.081e-02]; (ours - refined) / (ours - UD) = 2.802; lower than ours in 47 of 128 experts). Bias part 0.00104 (ours 0.00419, UD 0.00170), rest 0.01178 (ours 0.02053, UD 0.01877). Regularised (the search objective) 0.01512 (ours 0.02528, UD 0.02150). Diagonal 0.04628 (ours 0.03300, UD 0.03542), plain 0.21789 (ours 0.18302, UD 0.18953). Gains (ours / UD / refined): plain 0.9957 / 0.9958 / 0.9899, energy 1.0248 / 1.0276 / 1.0277, output 0.9966 / 0.9972 / 1.0065.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined (count-w.) | refined / ours | refined / UD |
|---|---|---|---|---|---|---|---|---|
| 6 | < 2304 | 32 | 0.021 | 0.03889 | 0.02507 | 0.03889 | 1.000 | 1.551 |
| 6 | >= 2304 | 96 | 0.979 | 0.03479 | 0.02246 | 0.01783 | 0.512 | 0.794 |
| 20 | < 2304 | 81 | 0.184 | 0.02641 | 0.02288 | 0.02641 | 1.000 | 1.154 |
| 20 | >= 2304 | 47 | 0.816 | 0.02434 | 0.01993 | 0.00974 | 0.400 | 0.489 |

Refinement log `models/refined/gu-refined-6-20.gguf.json` (scripts/refine-codes.py; wall time 1820 s for 256 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 96 | 32 | 4: 96 | 38386443 | 1971225 of 1990656 | 3232 | 4.86e-10 | 4.88e-10 | yes |
| 20 | 47 | 81 | 4: 47 | 29117280 | 974399 of 974592 | 3717 | 4.84e-10 | 3.13e-10 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 6: max abs(log - here) on reg 4.67e-10 before, 4.58e-10 after; layer 20: max abs(log - here) on reg 4.50e-10 before, 3.12e-10 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304').
