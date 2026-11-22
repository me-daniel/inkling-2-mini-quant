# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined = `models/refined/gu-refined-9-10-11.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/refine-pilot-9-10-11.tsv`.
Grams: `models/grams/gu-9-10-11` (capture settings in its log, not restated here), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

## Layer 9 (IQ3_XXS; UD `.scale` present, min 3.921e-05, max 0.0001086)

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

## Layer 10 (IQ3_XXS; UD `.scale` present, min 3.827e-05, max 0.0001814)

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

## Layer 11 (IQ3_XXS; UD `.scale` present, min 4.001e-05, max 0.0002121)

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

## Reading (amendment 7 rule, applied literally)

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

## refined against ours and UD

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
