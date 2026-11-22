# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined = `models/refined/gu-refined-7-8-12-13.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/v2r-offline-7-8-12-13.tsv`.
Grams: `models/grams/gu-7-8-12-13` (capture settings in its log, not restated here), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

## Layer 7 (IQ2_S; UD `.scale` present, min 3.885e-05, max 9.079e-05)

- Gram layout: expert 6: max |strict lower| = 0, max |strict upper| = 2.062e+05
- Gram layout: expert 71: max |strict lower| = 0, max |strict upper| = 1.153e+05
- Gram layout: expert 49: max |strict lower| = 0, max |strict upper| = 1.147e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 1.04e-07; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.7.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.725061 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 13, median 7006, max 66056

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03414 | 0.02329 | 0.6821 | 128 of 128 | 0.02114 | 0.6192 | 92 of 128 |
| full | gate | count-weighted | 0.03354 | 0.02282 | 0.6803 | 128 of 128 | 0.01704 | 0.5081 | 92 of 128 |
| full | gate | trace-weighted | 0.03358 | 0.02288 | 0.6814 | 128 of 128 | 0.01707 | 0.5083 | 92 of 128 |
| full | up | unweighted | 0.04301 | 0.02935 | 0.6823 | 128 of 128 | 0.02608 | 0.6062 | 92 of 128 |
| full | up | count-weighted | 0.04225 | 0.02874 | 0.6803 | 128 of 128 | 0.02065 | 0.4888 | 92 of 128 |
| full | up | trace-weighted | 0.04231 | 0.02883 | 0.6814 | 128 of 128 | 0.02069 | 0.4889 | 92 of 128 |
| full | both | unweighted | 0.03799 | 0.02592 | 0.6822 | 128 of 128 | 0.02327 | 0.6125 | 92 of 128 |
| full | both | count-weighted | 0.03733 | 0.02539 | 0.6802 | 128 of 128 | 0.01861 | 0.4985 | 92 of 128 |
| full | both | trace-weighted | 0.03738 | 0.02547 | 0.6813 | 128 of 128 | 0.01864 | 0.4986 | 92 of 128 |
| reg | gate | unweighted | 0.03557 | 0.02550 | 0.7171 | 128 of 128 | 0.02427 | 0.6825 | 92 of 128 |
| reg | gate | count-weighted | 0.03498 | 0.02503 | 0.7155 | 128 of 128 | 0.02058 | 0.5883 | 92 of 128 |
| reg | gate | trace-weighted | 0.03502 | 0.02509 | 0.7165 | 128 of 128 | 0.02061 | 0.5884 | 92 of 128 |
| reg | up | unweighted | 0.04421 | 0.03171 | 0.7172 | 128 of 128 | 0.02966 | 0.6709 | 92 of 128 |
| reg | up | count-weighted | 0.04348 | 0.03110 | 0.7154 | 128 of 128 | 0.02483 | 0.5710 | 92 of 128 |
| reg | up | trace-weighted | 0.04354 | 0.03120 | 0.7164 | 128 of 128 | 0.02487 | 0.5711 | 92 of 128 |
| reg | both | unweighted | 0.03934 | 0.02822 | 0.7172 | 128 of 128 | 0.02662 | 0.6765 | 92 of 128 |
| reg | both | count-weighted | 0.03871 | 0.02769 | 0.7154 | 128 of 128 | 0.02244 | 0.5797 | 92 of 128 |
| reg | both | trace-weighted | 0.03876 | 0.02777 | 0.7164 | 128 of 128 | 0.02247 | 0.5798 | 92 of 128 |
| bias | gate | unweighted | 0.00587 | 0.00155 | 0.2639 | 128 of 128 | 0.00224 | 0.3827 | 92 of 128 |
| bias | gate | count-weighted | 0.00547 | 0.00133 | 0.2437 | 128 of 128 | 7.817e-04 | 0.1429 | 92 of 128 |
| bias | gate | trace-weighted | 0.00547 | 0.00133 | 0.2442 | 128 of 128 | 7.832e-04 | 0.1433 | 92 of 128 |
| bias | up | unweighted | 0.00739 | 0.00195 | 0.2643 | 128 of 128 | 0.00252 | 0.3412 | 92 of 128 |
| bias | up | count-weighted | 0.00691 | 0.00168 | 0.2430 | 128 of 128 | 5.455e-04 | 0.0790 | 92 of 128 |
| bias | up | trace-weighted | 0.00691 | 0.00168 | 0.2435 | 128 of 128 | 5.463e-04 | 0.0791 | 92 of 128 |
| bias | both | unweighted | 0.00653 | 0.00172 | 0.2641 | 128 of 128 | 0.00236 | 0.3617 | 92 of 128 |
| bias | both | count-weighted | 0.00610 | 0.00148 | 0.2433 | 128 of 128 | 6.775e-04 | 0.1111 | 92 of 128 |
| bias | both | trace-weighted | 0.00609 | 0.00149 | 0.2438 | 128 of 128 | 6.788e-04 | 0.1114 | 92 of 128 |
| rest | gate | unweighted | 0.02828 | 0.02174 | 0.7689 | 128 of 128 | 0.01890 | 0.6682 | 92 of 128 |
| rest | gate | count-weighted | 0.02807 | 0.02149 | 0.7653 | 128 of 128 | 0.01626 | 0.5792 | 92 of 128 |
| rest | gate | trace-weighted | 0.02812 | 0.02155 | 0.7663 | 128 of 128 | 0.01629 | 0.5792 | 92 of 128 |
| rest | up | unweighted | 0.03562 | 0.02739 | 0.7691 | 128 of 128 | 0.02355 | 0.6613 | 92 of 128 |
| rest | up | count-weighted | 0.03534 | 0.02706 | 0.7657 | 128 of 128 | 0.02010 | 0.5689 | 92 of 128 |
| rest | up | trace-weighted | 0.03541 | 0.02715 | 0.7668 | 128 of 128 | 0.02014 | 0.5689 | 92 of 128 |
| rest | both | unweighted | 0.03146 | 0.02419 | 0.7690 | 128 of 128 | 0.02091 | 0.6645 | 92 of 128 |
| rest | both | count-weighted | 0.03123 | 0.02391 | 0.7655 | 128 of 128 | 0.01793 | 0.5741 | 92 of 128 |
| rest | both | trace-weighted | 0.03129 | 0.02398 | 0.7666 | 128 of 128 | 0.01796 | 0.5741 | 92 of 128 |
| diag | gate | unweighted | 0.06164 | 0.06574 | 1.0665 | 0 of 128 | 0.08120 | 1.3173 | 0 of 128 |
| diag | gate | count-weighted | 0.06166 | 0.06579 | 1.0670 | 0 of 128 | 0.08549 | 1.3864 | 0 of 128 |
| diag | gate | trace-weighted | 0.06165 | 0.06579 | 1.0670 | 0 of 128 | 0.08546 | 1.3862 | 0 of 128 |
| diag | up | unweighted | 0.06170 | 0.06580 | 1.0664 | 0 of 128 | 0.08150 | 1.3209 | 0 of 128 |
| diag | up | count-weighted | 0.06170 | 0.06583 | 1.0669 | 0 of 128 | 0.08583 | 1.3910 | 0 of 128 |
| diag | up | trace-weighted | 0.06169 | 0.06582 | 1.0670 | 0 of 128 | 0.08580 | 1.3908 | 0 of 128 |
| diag | both | unweighted | 0.06167 | 0.06577 | 1.0664 | 0 of 128 | 0.08135 | 1.3191 | 0 of 128 |
| diag | both | count-weighted | 0.06168 | 0.06581 | 1.0670 | 0 of 128 | 0.08566 | 1.3887 | 0 of 128 |
| diag | both | trace-weighted | 0.06167 | 0.06580 | 1.0670 | 0 of 128 | 0.08563 | 1.3885 | 0 of 128 |
| plain | gate | unweighted | 0.26275 | 0.26964 | 1.0262 | 0 of 128 | 0.31784 | 1.2097 | 0 of 128 |
| plain | gate | count-weighted | 0.26283 | 0.26978 | 1.0265 | 0 of 128 | 0.33188 | 1.2627 | 0 of 128 |
| plain | gate | trace-weighted | 0.26284 | 0.26980 | 1.0265 | 0 of 128 | 0.33190 | 1.2627 | 0 of 128 |
| plain | up | unweighted | 0.26279 | 0.26967 | 1.0262 | 0 of 128 | 0.31882 | 1.2132 | 0 of 128 |
| plain | up | count-weighted | 0.26287 | 0.26982 | 1.0264 | 0 of 128 | 0.33315 | 1.2673 | 0 of 128 |
| plain | up | trace-weighted | 0.26289 | 0.26983 | 1.0264 | 0 of 128 | 0.33318 | 1.2674 | 0 of 128 |
| plain | both | unweighted | 0.26277 | 0.26966 | 1.0262 | 0 of 128 | 0.31833 | 1.2115 | 0 of 128 |
| plain | both | count-weighted | 0.26285 | 0.26980 | 1.0264 | 0 of 128 | 0.33251 | 1.2650 | 0 of 128 |
| plain | both | trace-weighted | 0.26286 | 0.26981 | 1.0264 | 0 of 128 | 0.33254 | 1.2651 | 0 of 128 |
| gain | gate | unweighted | 0.99719 | 0.99831 | 1.0011 | 2 of 128 | 0.98874 | 0.9915 | 92 of 128 |
| gain | gate | count-weighted | 0.99720 | 0.99833 | 1.0011 | 2 of 128 | 0.98673 | 0.9895 | 92 of 128 |
| gain | gate | trace-weighted | 0.99721 | 0.99834 | 1.0011 | 2 of 128 | 0.98674 | 0.9895 | 92 of 128 |
| gain | up | unweighted | 0.99721 | 0.99834 | 1.0011 | 3 of 128 | 0.98867 | 0.9914 | 92 of 128 |
| gain | up | count-weighted | 0.99725 | 0.99840 | 1.0011 | 3 of 128 | 0.98666 | 0.9894 | 92 of 128 |
| gain | up | trace-weighted | 0.99726 | 0.99841 | 1.0011 | 3 of 128 | 0.98667 | 0.9894 | 92 of 128 |
| gain | both | unweighted | 0.99720 | 0.99832 | 1.0011 | 2 of 128 | 0.98871 | 0.9915 | 92 of 128 |
| gain | both | count-weighted | 0.99723 | 0.99836 | 1.0011 | 2 of 128 | 0.98669 | 0.9894 | 92 of 128 |
| gain | both | trace-weighted | 0.99724 | 0.99837 | 1.0011 | 2 of 128 | 0.98670 | 0.9894 | 92 of 128 |
| energy | gate | unweighted | 1.06341 | 1.06932 | 1.0056 | 0 of 128 | 1.07976 | 1.0154 | 0 of 128 |
| energy | gate | count-weighted | 1.06348 | 1.06944 | 1.0056 | 0 of 128 | 1.08381 | 1.0191 | 0 of 128 |
| energy | gate | trace-weighted | 1.06352 | 1.06947 | 1.0056 | 0 of 128 | 1.08385 | 1.0191 | 0 of 128 |
| energy | up | unweighted | 1.06348 | 1.06941 | 1.0056 | 0 of 128 | 1.08028 | 1.0158 | 0 of 128 |
| energy | up | count-weighted | 1.06361 | 1.06960 | 1.0056 | 0 of 128 | 1.08451 | 1.0196 | 0 of 128 |
| energy | up | trace-weighted | 1.06364 | 1.06963 | 1.0056 | 0 of 128 | 1.08455 | 1.0197 | 0 of 128 |
| energy | both | unweighted | 1.06345 | 1.06936 | 1.0056 | 0 of 128 | 1.08002 | 1.0156 | 0 of 128 |
| energy | both | count-weighted | 1.06355 | 1.06952 | 1.0056 | 0 of 128 | 1.08416 | 1.0194 | 0 of 128 |
| energy | both | trace-weighted | 1.06358 | 1.06955 | 1.0056 | 0 of 128 | 1.08420 | 1.0194 | 0 of 128 |
| ogain | gate | unweighted | 0.99654 | 0.99910 | 1.0026 | 7 of 128 | 1.00749 | 1.0110 | 0 of 128 |
| ogain | gate | count-weighted | 0.99683 | 0.99904 | 1.0022 | 7 of 128 | 1.01105 | 1.0143 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99683 | 0.99904 | 1.0022 | 7 of 128 | 1.01105 | 1.0143 | 0 of 128 |
| ogain | up | unweighted | 0.99620 | 0.99890 | 1.0027 | 2 of 128 | 1.00529 | 1.0091 | 0 of 128 |
| ogain | up | count-weighted | 0.99615 | 0.99888 | 1.0027 | 2 of 128 | 1.00779 | 1.0117 | 0 of 128 |
| ogain | up | trace-weighted | 0.99617 | 0.99889 | 1.0027 | 2 of 128 | 1.00779 | 1.0117 | 0 of 128 |
| ogain | both | unweighted | 0.99639 | 0.99901 | 1.0026 | 1 of 128 | 1.00652 | 1.0102 | 0 of 128 |
| ogain | both | count-weighted | 0.99653 | 0.99897 | 1.0025 | 1 of 128 | 1.00961 | 1.0131 | 0 of 128 |
| ogain | both | trace-weighted | 0.99653 | 0.99897 | 1.0025 | 1 of 128 | 1.00961 | 1.0131 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.0725e-02 | [-1.0979e-02, -1.0476e-02] |
| up | -1.3508e-02 | [-1.3713e-02, -1.3298e-02] |
| both | -1.1936e-02 | [-1.2144e-02, -1.1728e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.6501e-02 | [-1.7077e-02, -1.5921e-02] | -5.7754e-03 | [-6.2334e-03, -5.3227e-03] |
| full | up | -2.1596e-02 | [-2.2176e-02, -2.1045e-02] | -8.0883e-03 | [-8.6459e-03, -7.5616e-03] |
| full | both | -1.8722e-02 | [-1.9273e-02, -1.8183e-02] | -6.7852e-03 | [-7.2776e-03, -6.3103e-03] |
| reg | gate | -1.4404e-02 | [-1.4884e-02, -1.3920e-02] | -4.4503e-03 | [-4.8361e-03, -4.0690e-03] |
| reg | up | -1.8650e-02 | [-1.9122e-02, -1.8191e-02] | -6.2783e-03 | [-6.7450e-03, -5.8346e-03] |
| reg | both | -1.6270e-02 | [-1.6723e-02, -1.5825e-02] | -5.2551e-03 | [-5.6693e-03, -4.8542e-03] |
| bias | gate | -4.6879e-03 | [-4.8927e-03, -4.4927e-03] | -5.5100e-04 | [-6.4399e-04, -4.5218e-04] |
| bias | up | -6.3623e-03 | [-6.5660e-03, -6.1631e-03] | -1.1331e-03 | [-1.2435e-03, -1.0066e-03] |
| bias | both | -5.4179e-03 | [-5.6092e-03, -5.2348e-03] | -8.0546e-04 | [-9.0311e-04, -6.9948e-04] |
| rest | gate | -1.1813e-02 | [-1.2213e-02, -1.1399e-02] | -5.2244e-03 | [-5.6138e-03, -4.8457e-03] |
| rest | up | -1.5234e-02 | [-1.5665e-02, -1.4836e-02] | -6.9552e-03 | [-7.4263e-03, -6.5201e-03] |
| rest | both | -1.3304e-02 | [-1.3696e-02, -1.2914e-02] | -5.9798e-03 | [-6.3978e-03, -5.5813e-03] |
| diag | gate | +2.3827e-02 | [+2.3073e-02, +2.4598e-02] | +1.9695e-02 | [+1.8937e-02, +2.0473e-02] |
| diag | up | +2.4127e-02 | [+2.3369e-02, +2.4902e-02] | +1.9996e-02 | [+1.9228e-02, +2.0775e-02] |
| diag | both | +2.3977e-02 | [+2.3223e-02, +2.4748e-02] | +1.9845e-02 | [+1.9085e-02, +2.0622e-02] |

## Layer 8 (IQ2_S; UD `.scale` present, min 3.905e-05, max 0.0001062)

- Gram layout: expert 84: max |strict lower| = 0, max |strict upper| = 2.017e+05
- Gram layout: expert 50: max |strict lower| = 0, max |strict upper| = 1.974e+05
- Gram layout: expert 86: max |strict lower| = 0, max |strict upper| = 1.864e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 7.57e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.8.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.723775 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 33, median 5376, max 48774

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03449 | 0.02499 | 0.7248 | 128 of 128 | 0.02146 | 0.6224 | 92 of 128 |
| full | gate | count-weighted | 0.03447 | 0.02487 | 0.7214 | 128 of 128 | 0.01793 | 0.5200 | 92 of 128 |
| full | gate | trace-weighted | 0.03451 | 0.02492 | 0.7222 | 128 of 128 | 0.01796 | 0.5205 | 92 of 128 |
| full | up | unweighted | 0.04535 | 0.03284 | 0.7242 | 128 of 128 | 0.02741 | 0.6045 | 92 of 128 |
| full | up | count-weighted | 0.04486 | 0.03236 | 0.7212 | 128 of 128 | 0.02223 | 0.4955 | 92 of 128 |
| full | up | trace-weighted | 0.04497 | 0.03247 | 0.7220 | 128 of 128 | 0.02230 | 0.4958 | 92 of 128 |
| full | both | unweighted | 0.03905 | 0.02829 | 0.7245 | 128 of 128 | 0.02396 | 0.6135 | 92 of 128 |
| full | both | count-weighted | 0.03887 | 0.02804 | 0.7213 | 128 of 128 | 0.01974 | 0.5079 | 92 of 128 |
| full | both | trace-weighted | 0.03894 | 0.02812 | 0.7221 | 128 of 128 | 0.01979 | 0.5083 | 92 of 128 |
| reg | gate | unweighted | 0.03590 | 0.02710 | 0.7549 | 128 of 128 | 0.02460 | 0.6851 | 92 of 128 |
| reg | gate | count-weighted | 0.03589 | 0.02699 | 0.7519 | 128 of 128 | 0.02143 | 0.5972 | 92 of 128 |
| reg | gate | trace-weighted | 0.03592 | 0.02704 | 0.7527 | 128 of 128 | 0.02146 | 0.5975 | 92 of 128 |
| reg | up | unweighted | 0.04645 | 0.03504 | 0.7544 | 128 of 128 | 0.03108 | 0.6692 | 92 of 128 |
| reg | up | count-weighted | 0.04598 | 0.03457 | 0.7518 | 128 of 128 | 0.02646 | 0.5754 | 92 of 128 |
| reg | up | trace-weighted | 0.04609 | 0.03468 | 0.7525 | 128 of 128 | 0.02653 | 0.5757 | 92 of 128 |
| reg | both | unweighted | 0.04038 | 0.03048 | 0.7547 | 128 of 128 | 0.02734 | 0.6771 | 92 of 128 |
| reg | both | count-weighted | 0.04021 | 0.03023 | 0.7518 | 128 of 128 | 0.02358 | 0.5864 | 92 of 128 |
| reg | both | trace-weighted | 0.04027 | 0.03031 | 0.7526 | 128 of 128 | 0.02363 | 0.5867 | 92 of 128 |
| bias | gate | unweighted | 0.00593 | 0.00182 | 0.3073 | 128 of 128 | 0.00229 | 0.3861 | 92 of 128 |
| bias | gate | count-weighted | 0.00570 | 0.00162 | 0.2847 | 128 of 128 | 8.598e-04 | 0.1509 | 92 of 128 |
| bias | gate | trace-weighted | 0.00570 | 0.00162 | 0.2847 | 128 of 128 | 8.629e-04 | 0.1515 | 92 of 128 |
| bias | up | unweighted | 0.00785 | 0.00241 | 0.3074 | 128 of 128 | 0.00260 | 0.3307 | 92 of 128 |
| bias | up | count-weighted | 0.00741 | 0.00210 | 0.2840 | 128 of 128 | 5.508e-04 | 0.0743 | 92 of 128 |
| bias | up | trace-weighted | 0.00742 | 0.00211 | 0.2840 | 128 of 128 | 5.534e-04 | 0.0746 | 92 of 128 |
| bias | both | unweighted | 0.00674 | 0.00207 | 0.3073 | 128 of 128 | 0.00242 | 0.3585 | 92 of 128 |
| bias | both | count-weighted | 0.00642 | 0.00183 | 0.2843 | 128 of 128 | 7.269e-04 | 0.1132 | 92 of 128 |
| bias | both | trace-weighted | 0.00643 | 0.00183 | 0.2843 | 128 of 128 | 7.299e-04 | 0.1135 | 92 of 128 |
| rest | gate | unweighted | 0.02856 | 0.02317 | 0.8115 | 128 of 128 | 0.01917 | 0.6715 | 92 of 128 |
| rest | gate | count-weighted | 0.02877 | 0.02324 | 0.8078 | 128 of 128 | 0.01707 | 0.5931 | 92 of 128 |
| rest | gate | trace-weighted | 0.02881 | 0.02330 | 0.8087 | 128 of 128 | 0.01710 | 0.5934 | 92 of 128 |
| rest | up | unweighted | 0.03749 | 0.03043 | 0.8115 | 128 of 128 | 0.02482 | 0.6619 | 92 of 128 |
| rest | up | count-weighted | 0.03745 | 0.03025 | 0.8077 | 128 of 128 | 0.02168 | 0.5788 | 92 of 128 |
| rest | up | trace-weighted | 0.03755 | 0.03036 | 0.8086 | 128 of 128 | 0.02174 | 0.5791 | 92 of 128 |
| rest | both | unweighted | 0.03231 | 0.02622 | 0.8116 | 128 of 128 | 0.02154 | 0.6667 | 92 of 128 |
| rest | both | count-weighted | 0.03245 | 0.02621 | 0.8078 | 128 of 128 | 0.01902 | 0.5860 | 92 of 128 |
| rest | both | trace-weighted | 0.03251 | 0.02629 | 0.8086 | 128 of 128 | 0.01906 | 0.5864 | 92 of 128 |
| diag | gate | unweighted | 0.06168 | 0.06515 | 1.0563 | 0 of 128 | 0.08097 | 1.3128 | 0 of 128 |
| diag | gate | count-weighted | 0.06154 | 0.06505 | 1.0570 | 0 of 128 | 0.08414 | 1.3671 | 0 of 128 |
| diag | gate | trace-weighted | 0.06153 | 0.06504 | 1.0570 | 0 of 128 | 0.08408 | 1.3664 | 0 of 128 |
| diag | up | unweighted | 0.06177 | 0.06524 | 1.0562 | 0 of 128 | 0.08139 | 1.3176 | 0 of 128 |
| diag | up | count-weighted | 0.06161 | 0.06512 | 1.0569 | 0 of 128 | 0.08456 | 1.3724 | 0 of 128 |
| diag | up | trace-weighted | 0.06160 | 0.06511 | 1.0570 | 0 of 128 | 0.08450 | 1.3717 | 0 of 128 |
| diag | both | unweighted | 0.06173 | 0.06520 | 1.0563 | 0 of 128 | 0.08118 | 1.3152 | 0 of 128 |
| diag | both | count-weighted | 0.06158 | 0.06509 | 1.0570 | 0 of 128 | 0.08435 | 1.3697 | 0 of 128 |
| diag | both | trace-weighted | 0.06157 | 0.06507 | 1.0570 | 0 of 128 | 0.08429 | 1.3691 | 0 of 128 |
| plain | gate | unweighted | 0.26225 | 0.26812 | 1.0224 | 0 of 128 | 0.31336 | 1.1949 | 0 of 128 |
| plain | gate | count-weighted | 0.26239 | 0.26827 | 1.0224 | 0 of 128 | 0.32454 | 1.2368 | 0 of 128 |
| plain | gate | trace-weighted | 0.26242 | 0.26830 | 1.0224 | 0 of 128 | 0.32454 | 1.2367 | 0 of 128 |
| plain | up | unweighted | 0.26230 | 0.26817 | 1.0224 | 0 of 128 | 0.31447 | 1.1989 | 0 of 128 |
| plain | up | count-weighted | 0.26246 | 0.26833 | 1.0224 | 0 of 128 | 0.32592 | 1.2418 | 0 of 128 |
| plain | up | trace-weighted | 0.26249 | 0.26836 | 1.0224 | 0 of 128 | 0.32593 | 1.2417 | 0 of 128 |
| plain | both | unweighted | 0.26227 | 0.26814 | 1.0224 | 0 of 128 | 0.31392 | 1.1969 | 0 of 128 |
| plain | both | count-weighted | 0.26243 | 0.26830 | 1.0224 | 0 of 128 | 0.32523 | 1.2393 | 0 of 128 |
| plain | both | trace-weighted | 0.26245 | 0.26833 | 1.0224 | 0 of 128 | 0.32524 | 1.2392 | 0 of 128 |
| gain | gate | unweighted | 0.99745 | 0.99838 | 1.0009 | 5 of 128 | 0.98962 | 0.9922 | 92 of 128 |
| gain | gate | count-weighted | 0.99749 | 0.99840 | 1.0009 | 5 of 128 | 0.98814 | 0.9906 | 92 of 128 |
| gain | gate | trace-weighted | 0.99751 | 0.99842 | 1.0009 | 5 of 128 | 0.98816 | 0.9906 | 92 of 128 |
| gain | up | unweighted | 0.99747 | 0.99842 | 1.0010 | 3 of 128 | 0.98949 | 0.9920 | 92 of 128 |
| gain | up | count-weighted | 0.99757 | 0.99848 | 1.0009 | 3 of 128 | 0.98800 | 0.9904 | 92 of 128 |
| gain | up | trace-weighted | 0.99758 | 0.99849 | 1.0009 | 3 of 128 | 0.98802 | 0.9904 | 92 of 128 |
| gain | both | unweighted | 0.99746 | 0.99840 | 1.0009 | 5 of 128 | 0.98956 | 0.9921 | 92 of 128 |
| gain | both | count-weighted | 0.99753 | 0.99844 | 1.0009 | 5 of 128 | 0.98807 | 0.9905 | 92 of 128 |
| gain | both | trace-weighted | 0.99755 | 0.99846 | 1.0009 | 5 of 128 | 0.98809 | 0.9905 | 92 of 128 |
| energy | gate | unweighted | 1.06367 | 1.06864 | 1.0047 | 0 of 128 | 1.07854 | 1.0140 | 0 of 128 |
| energy | gate | count-weighted | 1.06384 | 1.06877 | 1.0046 | 0 of 128 | 1.08177 | 1.0169 | 0 of 128 |
| energy | gate | trace-weighted | 1.06389 | 1.06882 | 1.0046 | 0 of 128 | 1.08183 | 1.0169 | 0 of 128 |
| energy | up | unweighted | 1.06373 | 1.06875 | 1.0047 | 0 of 128 | 1.07902 | 1.0144 | 0 of 128 |
| energy | up | count-weighted | 1.06402 | 1.06897 | 1.0047 | 0 of 128 | 1.08240 | 1.0173 | 0 of 128 |
| energy | up | trace-weighted | 1.06406 | 1.06901 | 1.0046 | 0 of 128 | 1.08246 | 1.0173 | 0 of 128 |
| energy | both | unweighted | 1.06370 | 1.06870 | 1.0047 | 0 of 128 | 1.07878 | 1.0142 | 0 of 128 |
| energy | both | count-weighted | 1.06393 | 1.06887 | 1.0046 | 0 of 128 | 1.08209 | 1.0171 | 0 of 128 |
| energy | both | trace-weighted | 1.06398 | 1.06891 | 1.0046 | 0 of 128 | 1.08214 | 1.0171 | 0 of 128 |
| ogain | gate | unweighted | 0.99709 | 0.99923 | 1.0021 | 22 of 128 | 1.00830 | 1.0112 | 0 of 128 |
| ogain | gate | count-weighted | 0.99768 | 0.99904 | 1.0014 | 22 of 128 | 1.01226 | 1.0146 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99767 | 0.99904 | 1.0014 | 22 of 128 | 1.01227 | 1.0146 | 0 of 128 |
| ogain | up | unweighted | 0.99611 | 0.99881 | 1.0027 | 1 of 128 | 1.00546 | 1.0094 | 0 of 128 |
| ogain | up | count-weighted | 0.99624 | 0.99878 | 1.0025 | 1 of 128 | 1.00821 | 1.0120 | 0 of 128 |
| ogain | up | trace-weighted | 0.99627 | 0.99878 | 1.0025 | 1 of 128 | 1.00821 | 1.0120 | 0 of 128 |
| ogain | both | unweighted | 0.99666 | 0.99905 | 1.0024 | 4 of 128 | 1.00707 | 1.0105 | 0 of 128 |
| ogain | both | count-weighted | 0.99704 | 0.99893 | 1.0019 | 4 of 128 | 1.01050 | 1.0135 | 0 of 128 |
| ogain | both | trace-weighted | 0.99705 | 0.99893 | 1.0019 | 4 of 128 | 1.01051 | 1.0135 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -9.6052e-03 | [-9.8810e-03, -9.3092e-03] |
| up | -1.2505e-02 | [-1.2694e-02, -1.2307e-02] |
| both | -1.0835e-02 | [-1.1050e-02, -1.0608e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.6546e-02 | [-1.7128e-02, -1.5949e-02] | -6.9411e-03 | [-7.3768e-03, -6.5150e-03] |
| full | up | -2.2633e-02 | [-2.3265e-02, -2.2021e-02] | -1.0128e-02 | [-1.0681e-02, -9.5996e-03] |
| full | both | -1.9130e-02 | [-1.9683e-02, -1.8571e-02] | -8.2954e-03 | [-8.7575e-03, -7.8439e-03] |
| reg | gate | -1.4457e-02 | [-1.4944e-02, -1.3952e-02] | -5.5535e-03 | [-5.9144e-03, -5.1990e-03] |
| reg | up | -1.9525e-02 | [-2.0035e-02, -1.9023e-02] | -8.1117e-03 | [-8.5626e-03, -7.6719e-03] |
| reg | both | -1.6631e-02 | [-1.7084e-02, -1.6168e-02] | -6.6521e-03 | [-7.0354e-03, -6.2774e-03] |
| bias | gate | -4.8372e-03 | [-5.0593e-03, -4.6040e-03] | -7.6215e-04 | [-8.7732e-04, -6.3920e-04] |
| bias | up | -6.8574e-03 | [-7.0787e-03, -6.6309e-03] | -1.5530e-03 | [-1.6812e-03, -1.4127e-03] |
| bias | both | -5.6963e-03 | [-5.9072e-03, -5.4825e-03] | -1.0993e-03 | [-1.2161e-03, -9.7224e-04] |
| rest | gate | -1.1709e-02 | [-1.2121e-02, -1.1302e-02] | -6.1790e-03 | [-6.5495e-03, -5.8284e-03] |
| rest | up | -1.5776e-02 | [-1.6283e-02, -1.5307e-02] | -8.5752e-03 | [-9.0748e-03, -8.1140e-03] |
| rest | both | -1.3434e-02 | [-1.3850e-02, -1.3040e-02] | -7.1961e-03 | [-7.6091e-03, -6.8111e-03] |
| diag | gate | +2.2592e-02 | [+2.1907e-02, +2.3400e-02] | +1.9083e-02 | [+1.8397e-02, +1.9893e-02] |
| diag | up | +2.2945e-02 | [+2.2240e-02, +2.3768e-02] | +1.9437e-02 | [+1.8732e-02, +2.0258e-02] |
| diag | both | +2.2768e-02 | [+2.2074e-02, +2.3584e-02] | +1.9260e-02 | [+1.8564e-02, +2.0077e-02] |

## Layer 12 (IQ2_S; UD `.scale` present, min 4.008e-05, max 0.0002085)

- Gram layout: expert 31: max |strict lower| = 0, max |strict upper| = 7.543e+04
- Gram layout: expert 38: max |strict lower| = 0, max |strict upper| = 6.626e+04
- Gram layout: expert 65: max |strict lower| = 0, max |strict upper| = 6.091e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 4.68e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.12.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.634538 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 6, median 5313, max 45508

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03521 | 0.02478 | 0.7038 | 128 of 128 | 0.02218 | 0.6298 | 90 of 128 |
| full | gate | count-weighted | 0.03321 | 0.02321 | 0.6988 | 128 of 128 | 0.01665 | 0.5014 | 90 of 128 |
| full | gate | trace-weighted | 0.03328 | 0.02327 | 0.6992 | 128 of 128 | 0.01670 | 0.5017 | 90 of 128 |
| full | up | unweighted | 0.04347 | 0.03063 | 0.7045 | 128 of 128 | 0.02714 | 0.6244 | 90 of 128 |
| full | up | count-weighted | 0.04189 | 0.02938 | 0.7014 | 128 of 128 | 0.02078 | 0.4960 | 90 of 128 |
| full | up | trace-weighted | 0.04199 | 0.02947 | 0.7019 | 128 of 128 | 0.02084 | 0.4962 | 90 of 128 |
| full | both | unweighted | 0.03880 | 0.02732 | 0.7041 | 128 of 128 | 0.02434 | 0.6273 | 90 of 128 |
| full | both | count-weighted | 0.03692 | 0.02584 | 0.7000 | 128 of 128 | 0.01842 | 0.4988 | 90 of 128 |
| full | both | trace-weighted | 0.03700 | 0.02592 | 0.7005 | 128 of 128 | 0.01847 | 0.4991 | 90 of 128 |
| reg | gate | unweighted | 0.03668 | 0.02701 | 0.7363 | 128 of 128 | 0.02533 | 0.6907 | 90 of 128 |
| reg | gate | count-weighted | 0.03471 | 0.02540 | 0.7319 | 128 of 128 | 0.02020 | 0.5818 | 90 of 128 |
| reg | gate | trace-weighted | 0.03478 | 0.02547 | 0.7323 | 128 of 128 | 0.02025 | 0.5821 | 90 of 128 |
| reg | up | unweighted | 0.04475 | 0.03298 | 0.7370 | 128 of 128 | 0.03070 | 0.6861 | 90 of 128 |
| reg | up | count-weighted | 0.04323 | 0.03175 | 0.7344 | 128 of 128 | 0.02496 | 0.5774 | 90 of 128 |
| reg | up | trace-weighted | 0.04333 | 0.03184 | 0.7348 | 128 of 128 | 0.02503 | 0.5776 | 90 of 128 |
| reg | both | unweighted | 0.04021 | 0.02962 | 0.7367 | 128 of 128 | 0.02769 | 0.6886 | 90 of 128 |
| reg | both | count-weighted | 0.03838 | 0.02814 | 0.7330 | 128 of 128 | 0.02225 | 0.5797 | 90 of 128 |
| reg | both | trace-weighted | 0.03847 | 0.02822 | 0.7335 | 128 of 128 | 0.02231 | 0.5799 | 90 of 128 |
| bias | gate | unweighted | 0.00614 | 0.00199 | 0.3238 | 128 of 128 | 0.00261 | 0.4256 | 90 of 128 |
| bias | gate | count-weighted | 0.00518 | 0.00146 | 0.2814 | 128 of 128 | 6.246e-04 | 0.1206 | 90 of 128 |
| bias | gate | trace-weighted | 0.00518 | 0.00146 | 0.2816 | 128 of 128 | 6.242e-04 | 0.1205 | 90 of 128 |
| bias | up | unweighted | 0.00754 | 0.00245 | 0.3253 | 128 of 128 | 0.00305 | 0.4045 | 90 of 128 |
| bias | up | count-weighted | 0.00642 | 0.00184 | 0.2872 | 128 of 128 | 5.438e-04 | 0.0847 | 90 of 128 |
| bias | up | trace-weighted | 0.00642 | 0.00185 | 0.2873 | 128 of 128 | 5.411e-04 | 0.0842 | 90 of 128 |
| bias | both | unweighted | 0.00675 | 0.00219 | 0.3246 | 128 of 128 | 0.00281 | 0.4156 | 90 of 128 |
| bias | both | count-weighted | 0.00571 | 0.00162 | 0.2842 | 128 of 128 | 5.910e-04 | 0.1035 | 90 of 128 |
| bias | both | trace-weighted | 0.00571 | 0.00162 | 0.2844 | 128 of 128 | 5.897e-04 | 0.1032 | 90 of 128 |
| rest | gate | unweighted | 0.02907 | 0.02279 | 0.7841 | 128 of 128 | 0.01956 | 0.6730 | 90 of 128 |
| rest | gate | count-weighted | 0.02803 | 0.02175 | 0.7759 | 128 of 128 | 0.01603 | 0.5718 | 90 of 128 |
| rest | gate | trace-weighted | 0.02810 | 0.02181 | 0.7762 | 128 of 128 | 0.01607 | 0.5720 | 90 of 128 |
| rest | up | unweighted | 0.03593 | 0.02817 | 0.7841 | 128 of 128 | 0.02409 | 0.6705 | 90 of 128 |
| rest | up | count-weighted | 0.03547 | 0.02754 | 0.7764 | 128 of 128 | 0.02023 | 0.5705 | 90 of 128 |
| rest | up | trace-weighted | 0.03557 | 0.02763 | 0.7767 | 128 of 128 | 0.02030 | 0.5706 | 90 of 128 |
| rest | both | unweighted | 0.03205 | 0.02513 | 0.7841 | 128 of 128 | 0.02153 | 0.6719 | 90 of 128 |
| rest | both | count-weighted | 0.03121 | 0.02422 | 0.7761 | 128 of 128 | 0.01782 | 0.5712 | 90 of 128 |
| rest | both | trace-weighted | 0.03129 | 0.02430 | 0.7764 | 128 of 128 | 0.01788 | 0.5714 | 90 of 128 |
| diag | gate | unweighted | 0.06413 | 0.06817 | 1.0630 | 0 of 128 | 0.08420 | 1.3130 | 0 of 128 |
| diag | gate | count-weighted | 0.06420 | 0.06829 | 1.0637 | 0 of 128 | 0.08895 | 1.3855 | 0 of 128 |
| diag | gate | trace-weighted | 0.06420 | 0.06829 | 1.0637 | 0 of 128 | 0.08892 | 1.3849 | 0 of 128 |
| diag | up | unweighted | 0.06416 | 0.06820 | 1.0631 | 0 of 128 | 0.08441 | 1.3157 | 0 of 128 |
| diag | up | count-weighted | 0.06421 | 0.06831 | 1.0638 | 0 of 128 | 0.08921 | 1.3892 | 0 of 128 |
| diag | up | trace-weighted | 0.06421 | 0.06831 | 1.0638 | 0 of 128 | 0.08918 | 1.3887 | 0 of 128 |
| diag | both | unweighted | 0.06414 | 0.06819 | 1.0631 | 0 of 128 | 0.08431 | 1.3143 | 0 of 128 |
| diag | both | count-weighted | 0.06421 | 0.06830 | 1.0637 | 0 of 128 | 0.08908 | 1.3873 | 0 of 128 |
| diag | both | trace-weighted | 0.06421 | 0.06830 | 1.0638 | 0 of 128 | 0.08905 | 1.3868 | 0 of 128 |
| plain | gate | unweighted | 0.25671 | 0.26370 | 1.0272 | 0 of 128 | 0.29566 | 1.1517 | 0 of 128 |
| plain | gate | count-weighted | 0.25675 | 0.26384 | 1.0276 | 0 of 128 | 0.30506 | 1.1882 | 0 of 128 |
| plain | gate | trace-weighted | 0.25675 | 0.26384 | 1.0276 | 0 of 128 | 0.30500 | 1.1879 | 0 of 128 |
| plain | up | unweighted | 0.25670 | 0.26369 | 1.0272 | 0 of 128 | 0.29597 | 1.1530 | 0 of 128 |
| plain | up | count-weighted | 0.25674 | 0.26384 | 1.0276 | 0 of 128 | 0.30550 | 1.1899 | 0 of 128 |
| plain | up | trace-weighted | 0.25674 | 0.26384 | 1.0277 | 0 of 128 | 0.30544 | 1.1897 | 0 of 128 |
| plain | both | unweighted | 0.25671 | 0.26370 | 1.0272 | 0 of 128 | 0.29582 | 1.1524 | 0 of 128 |
| plain | both | count-weighted | 0.25674 | 0.26384 | 1.0276 | 0 of 128 | 0.30528 | 1.1890 | 0 of 128 |
| plain | both | trace-weighted | 0.25674 | 0.26384 | 1.0276 | 0 of 128 | 0.30522 | 1.1888 | 0 of 128 |
| gain | gate | unweighted | 0.99576 | 0.99698 | 1.0012 | 3 of 128 | 0.98853 | 0.9927 | 90 of 128 |
| gain | gate | count-weighted | 0.99573 | 0.99703 | 1.0013 | 3 of 128 | 0.98657 | 0.9908 | 90 of 128 |
| gain | gate | trace-weighted | 0.99573 | 0.99704 | 1.0013 | 3 of 128 | 0.98658 | 0.9908 | 90 of 128 |
| gain | up | unweighted | 0.99576 | 0.99700 | 1.0012 | 1 of 128 | 0.98877 | 0.9930 | 90 of 128 |
| gain | up | count-weighted | 0.99577 | 0.99708 | 1.0013 | 1 of 128 | 0.98701 | 0.9912 | 90 of 128 |
| gain | up | trace-weighted | 0.99577 | 0.99709 | 1.0013 | 1 of 128 | 0.98702 | 0.9912 | 90 of 128 |
| gain | both | unweighted | 0.99576 | 0.99699 | 1.0012 | 1 of 128 | 0.98865 | 0.9929 | 90 of 128 |
| gain | both | count-weighted | 0.99575 | 0.99705 | 1.0013 | 1 of 128 | 0.98679 | 0.9910 | 90 of 128 |
| gain | both | trace-weighted | 0.99575 | 0.99706 | 1.0013 | 1 of 128 | 0.98680 | 0.9910 | 90 of 128 |
| energy | gate | unweighted | 1.05742 | 1.06351 | 1.0058 | 0 of 128 | 1.06517 | 1.0073 | 0 of 128 |
| energy | gate | count-weighted | 1.05738 | 1.06367 | 1.0060 | 0 of 128 | 1.06631 | 1.0084 | 0 of 128 |
| energy | gate | trace-weighted | 1.05738 | 1.06369 | 1.0060 | 0 of 128 | 1.06629 | 1.0084 | 0 of 128 |
| energy | up | unweighted | 1.05742 | 1.06353 | 1.0058 | 0 of 128 | 1.06585 | 1.0080 | 0 of 128 |
| energy | up | count-weighted | 1.05745 | 1.06376 | 1.0060 | 0 of 128 | 1.06744 | 1.0095 | 0 of 128 |
| energy | up | trace-weighted | 1.05745 | 1.06378 | 1.0060 | 0 of 128 | 1.06743 | 1.0094 | 0 of 128 |
| energy | both | unweighted | 1.05742 | 1.06352 | 1.0058 | 0 of 128 | 1.06551 | 1.0076 | 0 of 128 |
| energy | both | count-weighted | 1.05741 | 1.06372 | 1.0060 | 0 of 128 | 1.06688 | 1.0090 | 0 of 128 |
| energy | both | trace-weighted | 1.05742 | 1.06374 | 1.0060 | 0 of 128 | 1.06686 | 1.0089 | 0 of 128 |
| ogain | gate | unweighted | 0.99206 | 0.99867 | 1.0067 | 1 of 128 | 1.00390 | 1.0119 | 0 of 128 |
| ogain | gate | count-weighted | 0.99198 | 0.99857 | 1.0066 | 1 of 128 | 1.00779 | 1.0159 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99198 | 0.99859 | 1.0067 | 1 of 128 | 1.00778 | 1.0159 | 0 of 128 |
| ogain | up | unweighted | 0.99488 | 0.99799 | 1.0031 | 3 of 128 | 1.00415 | 1.0093 | 0 of 128 |
| ogain | up | count-weighted | 0.99498 | 0.99808 | 1.0031 | 3 of 128 | 1.00691 | 1.0120 | 0 of 128 |
| ogain | up | trace-weighted | 0.99500 | 0.99809 | 1.0031 | 3 of 128 | 1.00690 | 1.0120 | 0 of 128 |
| ogain | both | unweighted | 0.99331 | 0.99837 | 1.0051 | 0 of 128 | 1.00402 | 1.0108 | 0 of 128 |
| ogain | both | count-weighted | 0.99329 | 0.99835 | 1.0051 | 0 of 128 | 1.00743 | 1.0142 | 0 of 128 |
| ogain | both | trace-weighted | 0.99330 | 0.99837 | 1.0051 | 0 of 128 | 1.00742 | 1.0142 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.0005e-02 | [-1.0271e-02, -9.7434e-03] |
| up | -1.2508e-02 | [-1.2726e-02, -1.2292e-02] |
| both | -1.1076e-02 | [-1.1275e-02, -1.0884e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.6559e-02 | [-1.7197e-02, -1.5955e-02] | -6.5543e-03 | [-7.0201e-03, -6.1310e-03] |
| full | up | -2.1110e-02 | [-2.1828e-02, -2.0442e-02] | -8.6025e-03 | [-9.1836e-03, -8.0545e-03] |
| full | both | -1.8503e-02 | [-1.9105e-02, -1.7943e-02] | -7.4272e-03 | [-7.9306e-03, -6.9653e-03] |
| reg | gate | -1.4515e-02 | [-1.5048e-02, -1.4003e-02] | -5.2079e-03 | [-5.5995e-03, -4.8512e-03] |
| reg | up | -1.8267e-02 | [-1.8857e-02, -1.7717e-02] | -6.7837e-03 | [-7.2682e-03, -6.3226e-03] |
| reg | both | -1.6132e-02 | [-1.6632e-02, -1.5666e-02] | -5.8849e-03 | [-6.3033e-03, -5.4939e-03] |
| bias | gate | -4.5546e-03 | [-4.7746e-03, -4.3449e-03] | -8.3280e-04 | [-9.3846e-04, -7.2274e-04] |
| bias | up | -5.8766e-03 | [-6.1480e-03, -5.6287e-03] | -1.3002e-03 | [-1.4365e-03, -1.1575e-03] |
| bias | both | -5.1192e-03 | [-5.3451e-03, -4.9123e-03] | -1.0319e-03 | [-1.1478e-03, -9.1373e-04] |
| rest | gate | -1.2005e-02 | [-1.2449e-02, -1.1585e-02] | -5.7216e-03 | [-6.0945e-03, -5.3818e-03] |
| rest | up | -1.5234e-02 | [-1.5716e-02, -1.4779e-02] | -7.3024e-03 | [-7.7734e-03, -6.8723e-03] |
| rest | both | -1.3383e-02 | [-1.3801e-02, -1.3004e-02] | -6.3953e-03 | [-6.8037e-03, -6.0342e-03] |
| diag | gate | +2.4748e-02 | [+2.4112e-02, +2.5484e-02] | +2.0661e-02 | [+2.0023e-02, +2.1403e-02] |
| diag | up | +2.4993e-02 | [+2.4358e-02, +2.5728e-02] | +2.0899e-02 | [+2.0255e-02, +2.1643e-02] |
| diag | both | +2.4871e-02 | [+2.4235e-02, +2.5603e-02] | +2.0780e-02 | [+2.0138e-02, +2.1520e-02] |

## Layer 13 (IQ3_XXS; UD `.scale` present, min 3.936e-05, max 0.0001701)

- Gram layout: expert 122: max |strict lower| = 0, max |strict upper| = 1.936e+05
- Gram layout: expert 50: max |strict lower| = 0, max |strict upper| = 1.521e+05
- Gram layout: expert 92: max |strict lower| = 0, max |strict upper| = 1.386e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 1.87e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.13.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.714882 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 29, median 5601, max 75023

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.01974 | 0.01529 | 0.7743 | 128 of 128 | 0.01297 | 0.6571 | 86 of 128 |
| full | gate | count-weighted | 0.01896 | 0.01462 | 0.7710 | 128 of 128 | 0.00999 | 0.5268 | 86 of 128 |
| full | gate | trace-weighted | 0.01896 | 0.01461 | 0.7709 | 128 of 128 | 0.00999 | 0.5271 | 86 of 128 |
| full | up | unweighted | 0.02412 | 0.01868 | 0.7744 | 128 of 128 | 0.01553 | 0.6436 | 86 of 128 |
| full | up | count-weighted | 0.02337 | 0.01806 | 0.7726 | 128 of 128 | 0.01198 | 0.5126 | 86 of 128 |
| full | up | trace-weighted | 0.02336 | 0.01805 | 0.7726 | 128 of 128 | 0.01198 | 0.5128 | 86 of 128 |
| full | both | unweighted | 0.02164 | 0.01675 | 0.7743 | 128 of 128 | 0.01408 | 0.6508 | 86 of 128 |
| full | both | count-weighted | 0.02085 | 0.01609 | 0.7717 | 128 of 128 | 0.01084 | 0.5200 | 86 of 128 |
| full | both | trace-weighted | 0.02085 | 0.01609 | 0.7717 | 128 of 128 | 0.01084 | 0.5202 | 86 of 128 |
| reg | gate | unweighted | 0.02048 | 0.01641 | 0.8014 | 128 of 128 | 0.01460 | 0.7129 | 86 of 128 |
| reg | gate | count-weighted | 0.01971 | 0.01573 | 0.7984 | 128 of 128 | 0.01187 | 0.6023 | 86 of 128 |
| reg | gate | trace-weighted | 0.01970 | 0.01573 | 0.7984 | 128 of 128 | 0.01187 | 0.6025 | 86 of 128 |
| reg | up | unweighted | 0.02471 | 0.01980 | 0.8014 | 128 of 128 | 0.01733 | 0.7014 | 86 of 128 |
| reg | up | count-weighted | 0.02399 | 0.01919 | 0.7999 | 128 of 128 | 0.01415 | 0.5899 | 86 of 128 |
| reg | up | trace-weighted | 0.02398 | 0.01918 | 0.7998 | 128 of 128 | 0.01415 | 0.5900 | 86 of 128 |
| reg | both | unweighted | 0.02232 | 0.01789 | 0.8014 | 128 of 128 | 0.01579 | 0.7075 | 86 of 128 |
| reg | both | count-weighted | 0.02156 | 0.01723 | 0.7991 | 128 of 128 | 0.01286 | 0.5963 | 86 of 128 |
| reg | both | trace-weighted | 0.02156 | 0.01723 | 0.7991 | 128 of 128 | 0.01286 | 0.5965 | 86 of 128 |
| bias | gate | unweighted | 0.00293 | 0.00106 | 0.3608 | 128 of 128 | 0.00133 | 0.4551 | 86 of 128 |
| bias | gate | count-weighted | 0.00252 | 8.020e-04 | 0.3185 | 128 of 128 | 3.409e-04 | 0.1354 | 86 of 128 |
| bias | gate | trace-weighted | 0.00251 | 7.996e-04 | 0.3181 | 128 of 128 | 3.392e-04 | 0.1349 | 86 of 128 |
| bias | up | unweighted | 0.00360 | 0.00129 | 0.3592 | 128 of 128 | 0.00150 | 0.4177 | 86 of 128 |
| bias | up | count-weighted | 0.00310 | 9.938e-04 | 0.3207 | 128 of 128 | 2.529e-04 | 0.0816 | 86 of 128 |
| bias | up | trace-weighted | 0.00309 | 9.908e-04 | 0.3203 | 128 of 128 | 2.504e-04 | 0.0810 | 86 of 128 |
| bias | both | unweighted | 0.00322 | 0.00116 | 0.3600 | 128 of 128 | 0.00141 | 0.4375 | 86 of 128 |
| bias | both | count-weighted | 0.00277 | 8.845e-04 | 0.3196 | 128 of 128 | 3.031e-04 | 0.1095 | 86 of 128 |
| bias | both | trace-weighted | 0.00276 | 8.819e-04 | 0.3192 | 128 of 128 | 3.011e-04 | 0.1090 | 86 of 128 |
| rest | gate | unweighted | 0.01681 | 0.01423 | 0.8464 | 128 of 128 | 0.01164 | 0.6923 | 86 of 128 |
| rest | gate | count-weighted | 0.01644 | 0.01382 | 0.8403 | 128 of 128 | 0.00965 | 0.5868 | 86 of 128 |
| rest | gate | trace-weighted | 0.01644 | 0.01381 | 0.8402 | 128 of 128 | 0.00965 | 0.5870 | 86 of 128 |
| rest | up | unweighted | 0.02052 | 0.01739 | 0.8472 | 128 of 128 | 0.01402 | 0.6833 | 86 of 128 |
| rest | up | count-weighted | 0.02027 | 0.01706 | 0.8417 | 128 of 128 | 0.01173 | 0.5785 | 86 of 128 |
| rest | up | trace-weighted | 0.02027 | 0.01706 | 0.8416 | 128 of 128 | 0.01173 | 0.5787 | 86 of 128 |
| rest | both | unweighted | 0.01842 | 0.01559 | 0.8467 | 128 of 128 | 0.01267 | 0.6881 | 86 of 128 |
| rest | both | count-weighted | 0.01809 | 0.01521 | 0.8409 | 128 of 128 | 0.01054 | 0.5828 | 86 of 128 |
| rest | both | trace-weighted | 0.01809 | 0.01521 | 0.8408 | 128 of 128 | 0.01054 | 0.5830 | 86 of 128 |
| diag | gate | unweighted | 0.03295 | 0.03535 | 1.0730 | 0 of 128 | 0.04213 | 1.2787 | 0 of 128 |
| diag | gate | count-weighted | 0.03295 | 0.03536 | 1.0732 | 0 of 128 | 0.04466 | 1.3555 | 0 of 128 |
| diag | gate | trace-weighted | 0.03295 | 0.03536 | 1.0732 | 0 of 128 | 0.04465 | 1.3553 | 0 of 128 |
| diag | up | unweighted | 0.03297 | 0.03538 | 1.0730 | 0 of 128 | 0.04229 | 1.2825 | 0 of 128 |
| diag | up | count-weighted | 0.03296 | 0.03537 | 1.0732 | 0 of 128 | 0.04483 | 1.3602 | 0 of 128 |
| diag | up | trace-weighted | 0.03296 | 0.03537 | 1.0732 | 0 of 128 | 0.04482 | 1.3600 | 0 of 128 |
| diag | both | unweighted | 0.03296 | 0.03537 | 1.0730 | 0 of 128 | 0.04221 | 1.2806 | 0 of 128 |
| diag | both | count-weighted | 0.03295 | 0.03537 | 1.0732 | 0 of 128 | 0.04475 | 1.3578 | 0 of 128 |
| diag | both | trace-weighted | 0.03295 | 0.03537 | 1.0732 | 0 of 128 | 0.04474 | 1.3576 | 0 of 128 |
| plain | gate | unweighted | 0.18483 | 0.19094 | 1.0331 | 0 of 128 | 0.20973 | 1.1347 | 0 of 128 |
| plain | gate | count-weighted | 0.18478 | 0.19096 | 1.0334 | 0 of 128 | 0.21680 | 1.1733 | 0 of 128 |
| plain | gate | trace-weighted | 0.18477 | 0.19095 | 1.0335 | 0 of 128 | 0.21678 | 1.1732 | 0 of 128 |
| plain | up | unweighted | 0.18483 | 0.19095 | 1.0331 | 0 of 128 | 0.20999 | 1.1361 | 0 of 128 |
| plain | up | count-weighted | 0.18475 | 0.19094 | 1.0335 | 0 of 128 | 0.21708 | 1.1750 | 0 of 128 |
| plain | up | trace-weighted | 0.18474 | 0.19094 | 1.0335 | 0 of 128 | 0.21707 | 1.1750 | 0 of 128 |
| plain | both | unweighted | 0.18483 | 0.19095 | 1.0331 | 0 of 128 | 0.20986 | 1.1354 | 0 of 128 |
| plain | both | count-weighted | 0.18476 | 0.19095 | 1.0335 | 0 of 128 | 0.21694 | 1.1742 | 0 of 128 |
| plain | both | trace-weighted | 0.18476 | 0.19095 | 1.0335 | 0 of 128 | 0.21692 | 1.1741 | 0 of 128 |
| gain | gate | unweighted | 0.99634 | 0.99666 | 1.0003 | 29 of 128 | 0.99190 | 0.9955 | 86 of 128 |
| gain | gate | count-weighted | 0.99638 | 0.99674 | 1.0004 | 29 of 128 | 0.99045 | 0.9941 | 86 of 128 |
| gain | gate | trace-weighted | 0.99638 | 0.99674 | 1.0004 | 29 of 128 | 0.99046 | 0.9941 | 86 of 128 |
| gain | up | unweighted | 0.99636 | 0.99670 | 1.0003 | 28 of 128 | 0.99194 | 0.9956 | 86 of 128 |
| gain | up | count-weighted | 0.99642 | 0.99681 | 1.0004 | 28 of 128 | 0.99054 | 0.9941 | 86 of 128 |
| gain | up | trace-weighted | 0.99642 | 0.99681 | 1.0004 | 28 of 128 | 0.99054 | 0.9941 | 86 of 128 |
| gain | both | unweighted | 0.99635 | 0.99668 | 1.0003 | 28 of 128 | 0.99192 | 0.9956 | 86 of 128 |
| gain | both | count-weighted | 0.99640 | 0.99677 | 1.0004 | 28 of 128 | 0.99050 | 0.9941 | 86 of 128 |
| gain | both | trace-weighted | 0.99640 | 0.99677 | 1.0004 | 28 of 128 | 0.99050 | 0.9941 | 86 of 128 |
| energy | gate | unweighted | 1.02684 | 1.02979 | 1.0029 | 1 of 128 | 1.02812 | 1.0012 | 5 of 128 |
| energy | gate | count-weighted | 1.02690 | 1.02994 | 1.0030 | 1 of 128 | 1.02796 | 1.0010 | 5 of 128 |
| energy | gate | trace-weighted | 1.02691 | 1.02994 | 1.0030 | 1 of 128 | 1.02796 | 1.0010 | 5 of 128 |
| energy | up | unweighted | 1.02689 | 1.02987 | 1.0029 | 0 of 128 | 1.02830 | 1.0014 | 1 of 128 |
| energy | up | count-weighted | 1.02697 | 1.03007 | 1.0030 | 0 of 128 | 1.02825 | 1.0013 | 1 of 128 |
| energy | up | trace-weighted | 1.02697 | 1.03007 | 1.0030 | 0 of 128 | 1.02825 | 1.0012 | 1 of 128 |
| energy | both | unweighted | 1.02686 | 1.02983 | 1.0029 | 1 of 128 | 1.02821 | 1.0013 | 1 of 128 |
| energy | both | count-weighted | 1.02694 | 1.03001 | 1.0030 | 1 of 128 | 1.02811 | 1.0011 | 1 of 128 |
| energy | both | trace-weighted | 1.02694 | 1.03001 | 1.0030 | 1 of 128 | 1.02810 | 1.0011 | 1 of 128 |
| ogain | gate | unweighted | 0.99795 | 0.99848 | 1.0005 | 31 of 128 | 1.00423 | 1.0063 | 0 of 128 |
| ogain | gate | count-weighted | 0.99812 | 0.99857 | 1.0005 | 31 of 128 | 1.00665 | 1.0085 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99813 | 0.99857 | 1.0004 | 31 of 128 | 1.00664 | 1.0085 | 0 of 128 |
| ogain | up | unweighted | 0.99755 | 0.99801 | 1.0005 | 39 of 128 | 1.00275 | 1.0052 | 0 of 128 |
| ogain | up | count-weighted | 0.99771 | 0.99830 | 1.0006 | 39 of 128 | 1.00471 | 1.0070 | 0 of 128 |
| ogain | up | trace-weighted | 0.99772 | 0.99830 | 1.0006 | 39 of 128 | 1.00470 | 1.0070 | 0 of 128 |
| ogain | both | unweighted | 0.99778 | 0.99828 | 1.0005 | 27 of 128 | 1.00359 | 1.0058 | 0 of 128 |
| ogain | both | count-weighted | 0.99794 | 0.99846 | 1.0005 | 27 of 128 | 1.00581 | 1.0079 | 0 of 128 |
| ogain | both | trace-weighted | 0.99795 | 0.99846 | 1.0005 | 27 of 128 | 1.00580 | 1.0079 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -4.3430e-03 | [-4.4768e-03, -4.2036e-03] |
| up | -5.3139e-03 | [-5.4231e-03, -5.2187e-03] |
| both | -4.7603e-03 | [-4.8667e-03, -4.6562e-03] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -8.9719e-03 | [-9.3761e-03, -8.6034e-03] | -4.6289e-03 | [-4.9567e-03, -4.3513e-03] |
| full | up | -1.1390e-02 | [-1.1893e-02, -1.0964e-02] | -6.0756e-03 | [-6.5181e-03, -5.7013e-03] |
| full | both | -1.0010e-02 | [-1.0428e-02, -9.6607e-03] | -5.2493e-03 | [-5.6120e-03, -4.9492e-03] |
| reg | gate | -7.8369e-03 | [-8.1739e-03, -7.5284e-03] | -3.8639e-03 | [-4.1358e-03, -3.6319e-03] |
| reg | up | -9.8379e-03 | [-1.0247e-02, -9.4891e-03] | -5.0374e-03 | [-5.4017e-03, -4.7283e-03] |
| reg | both | -8.7038e-03 | [-9.0464e-03, -8.4162e-03] | -4.3719e-03 | [-4.6746e-03, -4.1209e-03] |
| bias | gate | -2.1773e-03 | [-2.2881e-03, -2.0696e-03] | -4.6108e-04 | [-5.1687e-04, -4.0525e-04] |
| bias | up | -2.8456e-03 | [-2.9949e-03, -2.7219e-03] | -7.4090e-04 | [-8.1337e-04, -6.7089e-04] |
| bias | both | -2.4643e-03 | [-2.5830e-03, -2.3627e-03] | -5.8139e-04 | [-6.4230e-04, -5.2298e-04] |
| rest | gate | -6.7946e-03 | [-7.1001e-03, -6.5271e-03] | -4.1678e-03 | [-4.4511e-03, -3.9317e-03] |
| rest | up | -8.5440e-03 | [-8.9162e-03, -8.2297e-03] | -5.3347e-03 | [-5.7146e-03, -5.0149e-03] |
| rest | both | -7.5452e-03 | [-7.8577e-03, -7.2861e-03] | -4.6679e-03 | [-4.9833e-03, -4.4119e-03] |
| diag | gate | +1.1713e-02 | [+1.1320e-02, +1.2176e-02] | +9.3023e-03 | [+8.9073e-03, +9.7681e-03] |
| diag | up | +1.1872e-02 | [+1.1462e-02, +1.2356e-02] | +9.4584e-03 | [+9.0425e-03, +9.9449e-03] |
| diag | both | +1.1792e-02 | [+1.1393e-02, +1.2265e-02] | +9.3803e-03 | [+8.9768e-03, +9.8555e-03] |

## Reading (amendment 7 rule, applied literally)

Rule: supported if UD is lower than ours on the full error and higher or equal on the diagonal one in both layers; if UD is also worse on the full error, layer-local output error does not explain UD's edge.

| mean | half | layer 7 full UD < ours | layer 7 diag UD >= ours | layer 8 full UD < ours | layer 8 diag UD >= ours | layer 12 full UD < ours | layer 12 diag UD >= ours | layer 13 full UD < ours | layer 13 diag UD >= ours | branch |
|---|---|---|---|---|---|---|---|---|---|---|
| unweighted | gate | yes | yes | yes | yes | yes | yes | yes | yes | supported |
| unweighted | up | yes | yes | yes | yes | yes | yes | yes | yes | supported |
| unweighted | both | yes | yes | yes | yes | yes | yes | yes | yes | supported |
| count-weighted | gate | yes | yes | yes | yes | yes | yes | yes | yes | supported |
| count-weighted | up | yes | yes | yes | yes | yes | yes | yes | yes | supported |
| count-weighted | both | yes | yes | yes | yes | yes | yes | yes | yes | supported |
| trace-weighted | gate | yes | yes | yes | yes | yes | yes | yes | yes | supported |
| trace-weighted | up | yes | yes | yes | yes | yes | yes | yes | yes | supported |
| trace-weighted | both | yes | yes | yes | yes | yes | yes | yes | yes | supported |

In layer 7 (both halves, count-weighted), UD's full error is 0.02539 against our 0.03733 (ratio 0.680; UD - ours -1.194e-02, 95% CI [-1.214e-02, -1.173e-02]) and its diagonal error 0.06581 against our 0.06168 (ratio 1.067); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 7 (added after registration, not part of the rule): the input-mean part is 0.00610 ours against 0.00148 UD (ratio 0.243), the centred rest 0.03123 against 0.02391 (ratio 0.766). In layer 8 (both halves, count-weighted), UD's full error is 0.02804 against our 0.03887 (ratio 0.721; UD - ours -1.083e-02, 95% CI [-1.105e-02, -1.061e-02]) and its diagonal error 0.06509 against our 0.06158 (ratio 1.057); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 8 (added after registration, not part of the rule): the input-mean part is 0.00642 ours against 0.00183 UD (ratio 0.284), the centred rest 0.03245 against 0.02621 (ratio 0.808). In layer 12 (both halves, count-weighted), UD's full error is 0.02584 against our 0.03692 (ratio 0.700; UD - ours -1.108e-02, 95% CI [-1.128e-02, -1.088e-02]) and its diagonal error 0.06830 against our 0.06421 (ratio 1.064); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 12 (added after registration, not part of the rule): the input-mean part is 0.00571 ours against 0.00162 UD (ratio 0.284), the centred rest 0.03121 against 0.02422 (ratio 0.776). In layer 13 (both halves, count-weighted), UD's full error is 0.01609 against our 0.02085 (ratio 0.772; UD - ours -4.760e-03, 95% CI [-4.867e-03, -4.656e-03]) and its diagonal error 0.03537 against our 0.03295 (ratio 1.073); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 13 (added after registration, not part of the rule): the input-mean part is 0.00277 ours against 8.845e-04 UD (ratio 0.320), the centred rest 0.01809 against 0.01521 (ratio 0.841). Under the registered rule this is "supported" for both halves together, count-weighted; across the three means and three halves the branches are: supported.

## refined against ours and UD

In layer 7 (both halves, count-weighted), refined's full error is 0.01861 against our 0.03733 and UD's 0.02539 (refined - ours -1.872e-02, 95% CI [-1.927e-02, -1.818e-02]; (ours - refined) / (ours - UD) = 1.568; lower than ours in 92 of 128 experts). Bias part 6.775e-04 (ours 0.00610, UD 0.00148), rest 0.01793 (ours 0.03123, UD 0.02391). Regularised (the search objective) 0.02244 (ours 0.03871, UD 0.02769). Diagonal 0.08566 (ours 0.06168, UD 0.06581), plain 0.33251 (ours 0.26285, UD 0.26980). Gains (ours / UD / refined): plain 0.9972 / 0.9984 / 0.9867, energy 1.0635 / 1.0695 / 1.0842, output 0.9965 / 0.9990 / 1.0096. In layer 8 (both halves, count-weighted), refined's full error is 0.01974 against our 0.03887 and UD's 0.02804 (refined - ours -1.913e-02, 95% CI [-1.968e-02, -1.857e-02]; (ours - refined) / (ours - UD) = 1.766; lower than ours in 92 of 128 experts). Bias part 7.269e-04 (ours 0.00642, UD 0.00183), rest 0.01902 (ours 0.03245, UD 0.02621). Regularised (the search objective) 0.02358 (ours 0.04021, UD 0.03023). Diagonal 0.08435 (ours 0.06158, UD 0.06509), plain 0.32523 (ours 0.26243, UD 0.26830). Gains (ours / UD / refined): plain 0.9975 / 0.9984 / 0.9881, energy 1.0639 / 1.0689 / 1.0821, output 0.9970 / 0.9989 / 1.0105. In layer 12 (both halves, count-weighted), refined's full error is 0.01842 against our 0.03692 and UD's 0.02584 (refined - ours -1.850e-02, 95% CI [-1.911e-02, -1.794e-02]; (ours - refined) / (ours - UD) = 1.671; lower than ours in 90 of 128 experts). Bias part 5.910e-04 (ours 0.00571, UD 0.00162), rest 0.01782 (ours 0.03121, UD 0.02422). Regularised (the search objective) 0.02225 (ours 0.03838, UD 0.02814). Diagonal 0.08908 (ours 0.06421, UD 0.06830), plain 0.30528 (ours 0.25674, UD 0.26384). Gains (ours / UD / refined): plain 0.9957 / 0.9971 / 0.9868, energy 1.0574 / 1.0637 / 1.0669, output 0.9933 / 0.9984 / 1.0074. In layer 13 (both halves, count-weighted), refined's full error is 0.01084 against our 0.02085 and UD's 0.01609 (refined - ours -1.001e-02, 95% CI [-1.043e-02, -9.661e-03]; (ours - refined) / (ours - UD) = 2.103; lower than ours in 86 of 128 experts). Bias part 3.031e-04 (ours 0.00277, UD 8.845e-04), rest 0.01054 (ours 0.01809, UD 0.01521). Regularised (the search objective) 0.01286 (ours 0.02156, UD 0.01723). Diagonal 0.04475 (ours 0.03295, UD 0.03537), plain 0.21694 (ours 0.18476, UD 0.19095). Gains (ours / UD / refined): plain 0.9964 / 0.9968 / 0.9905, energy 1.0269 / 1.0300 / 1.0281, output 0.9979 / 0.9985 / 1.0058.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined (count-w.) | refined / ours | refined / UD |
|---|---|---|---|---|---|---|---|---|
| 7 | < 2304 | 36 | 0.032 | 0.03815 | 0.02594 | 0.03815 | 1.000 | 1.470 |
| 7 | >= 2304 | 92 | 0.968 | 0.03730 | 0.02537 | 0.01797 | 0.482 | 0.708 |
| 8 | < 2304 | 36 | 0.028 | 0.03957 | 0.02887 | 0.03957 | 1.000 | 1.371 |
| 8 | >= 2304 | 92 | 0.972 | 0.03885 | 0.02802 | 0.01917 | 0.493 | 0.684 |
| 12 | < 2304 | 38 | 0.025 | 0.04104 | 0.02928 | 0.04104 | 1.000 | 1.402 |
| 12 | >= 2304 | 90 | 0.975 | 0.03681 | 0.02575 | 0.01783 | 0.484 | 0.692 |
| 13 | < 2304 | 42 | 0.025 | 0.02229 | 0.01737 | 0.02229 | 1.000 | 1.284 |
| 13 | >= 2304 | 86 | 0.975 | 0.02082 | 0.01606 | 0.01055 | 0.507 | 0.657 |

Refinement log `models/refined/gu-refined-7-8-12-13.gguf.json` (scripts/refine-codes.py; wall time 5146 s for 512 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 92 | 36 | 4: 92 | 44730417 | 1907002 of 1907712 | 7014 | 4.54e-10 | 3.86e-10 | yes |
| 8 | 92 | 36 | 4: 92 | 43704790 | 1906837 of 1907712 | 5106 | 6.62e-10 | 6.44e-10 | yes |
| 12 | 90 | 38 | 4: 90 | 40846166 | 1865427 of 1866240 | 3759 | 4.31e-10 | 3.98e-10 | yes |
| 13 | 86 | 42 | 4: 86 | 55357746 | 1783106 of 1783296 | 4441 | 2.35e-10 | 1.83e-10 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 7: max abs(log - here) on reg 4.30e-10 before, 3.76e-10 after; layer 8: max abs(log - here) on reg 6.18e-10 before, 6.27e-10 after; layer 12: max abs(log - here) on reg 4.01e-10 before, 3.94e-10 after; layer 13: max abs(log - here) on reg 2.26e-10 before, 1.82e-10 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304').
