# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined = `models/refined/gu-refined-mixc-22-23.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/v2c-offline-22-23.tsv`.
Grams: `models/grams/gu-22-23-mixc` (capture settings in its log, not restated here), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

## Layer 22 (IQ3_XXS; UD `.scale` absent)

- Gram layout: expert 50: max |strict lower| = 0, max |strict upper| = 2.683e+05
- Gram layout: expert 33: max |strict lower| = 0, max |strict upper| = 2.111e+05
- Gram layout: expert 118: max |strict lower| = 0, max |strict upper| = 1.765e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 8.97e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.22.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.633713 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 714, median 11536, max 31833

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.02199 | 0.01610 | 0.7322 | 128 of 128 | 0.01066 | 0.4847 | 124 of 128 |
| full | gate | count-weighted | 0.02144 | 0.01555 | 0.7253 | 128 of 128 | 0.01039 | 0.4845 | 124 of 128 |
| full | gate | trace-weighted | 0.02169 | 0.01574 | 0.7259 | 128 of 128 | 0.01048 | 0.4834 | 124 of 128 |
| full | up | unweighted | 0.02752 | 0.02009 | 0.7300 | 128 of 128 | 0.01272 | 0.4623 | 124 of 128 |
| full | up | count-weighted | 0.02751 | 0.01989 | 0.7232 | 128 of 128 | 0.01269 | 0.4612 | 124 of 128 |
| full | up | trace-weighted | 0.02758 | 0.01996 | 0.7238 | 128 of 128 | 0.01271 | 0.4607 | 124 of 128 |
| full | both | unweighted | 0.02430 | 0.01777 | 0.7313 | 128 of 128 | 0.01151 | 0.4737 | 124 of 128 |
| full | both | count-weighted | 0.02395 | 0.01735 | 0.7244 | 128 of 128 | 0.01133 | 0.4732 | 124 of 128 |
| full | both | trace-weighted | 0.02413 | 0.01749 | 0.7250 | 128 of 128 | 0.01140 | 0.4724 | 124 of 128 |
| reg | gate | unweighted | 0.02256 | 0.01724 | 0.7640 | 128 of 128 | 0.01283 | 0.5685 | 124 of 128 |
| reg | gate | count-weighted | 0.02203 | 0.01669 | 0.7577 | 128 of 128 | 0.01250 | 0.5674 | 124 of 128 |
| reg | gate | trace-weighted | 0.02227 | 0.01689 | 0.7582 | 128 of 128 | 0.01262 | 0.5665 | 124 of 128 |
| reg | up | unweighted | 0.02781 | 0.02119 | 0.7620 | 128 of 128 | 0.01525 | 0.5484 | 124 of 128 |
| reg | up | count-weighted | 0.02781 | 0.02102 | 0.7558 | 128 of 128 | 0.01520 | 0.5466 | 124 of 128 |
| reg | up | trace-weighted | 0.02787 | 0.02108 | 0.7563 | 128 of 128 | 0.01522 | 0.5462 | 124 of 128 |
| reg | both | unweighted | 0.02477 | 0.01891 | 0.7631 | 128 of 128 | 0.01384 | 0.5586 | 124 of 128 |
| reg | both | count-weighted | 0.02445 | 0.01851 | 0.7569 | 128 of 128 | 0.01363 | 0.5573 | 124 of 128 |
| reg | both | trace-weighted | 0.02462 | 0.01865 | 0.7574 | 128 of 128 | 0.01371 | 0.5566 | 124 of 128 |
| bias | gate | unweighted | 0.00393 | 0.00101 | 0.2581 | 128 of 128 | 6.111e-04 | 0.1555 | 124 of 128 |
| bias | gate | count-weighted | 0.00389 | 9.770e-04 | 0.2514 | 128 of 128 | 5.509e-04 | 0.1417 | 124 of 128 |
| bias | gate | trace-weighted | 0.00392 | 9.918e-04 | 0.2527 | 128 of 128 | 5.470e-04 | 0.1394 | 124 of 128 |
| bias | up | unweighted | 0.00496 | 0.00126 | 0.2541 | 128 of 128 | 4.222e-04 | 0.0850 | 124 of 128 |
| bias | up | count-weighted | 0.00505 | 0.00126 | 0.2491 | 128 of 128 | 3.413e-04 | 0.0676 | 124 of 128 |
| bias | up | trace-weighted | 0.00505 | 0.00126 | 0.2501 | 128 of 128 | 3.402e-04 | 0.0674 | 124 of 128 |
| bias | both | unweighted | 0.00436 | 0.00112 | 0.2560 | 128 of 128 | 5.275e-04 | 0.1210 | 124 of 128 |
| bias | both | count-weighted | 0.00436 | 0.00109 | 0.2502 | 128 of 128 | 4.621e-04 | 0.1059 | 124 of 128 |
| bias | both | trace-weighted | 0.00439 | 0.00110 | 0.2514 | 128 of 128 | 4.590e-04 | 0.1046 | 124 of 128 |
| rest | gate | unweighted | 0.01806 | 0.01509 | 0.8354 | 128 of 128 | 0.01005 | 0.5564 | 124 of 128 |
| rest | gate | count-weighted | 0.01755 | 0.01457 | 0.8303 | 128 of 128 | 0.00984 | 0.5604 | 124 of 128 |
| rest | gate | trace-weighted | 0.01776 | 0.01475 | 0.8304 | 128 of 128 | 0.00994 | 0.5594 | 124 of 128 |
| rest | up | unweighted | 0.02256 | 0.01883 | 0.8348 | 128 of 128 | 0.01230 | 0.5453 | 124 of 128 |
| rest | up | count-weighted | 0.02246 | 0.01864 | 0.8297 | 128 of 128 | 0.01235 | 0.5496 | 124 of 128 |
| rest | up | trace-weighted | 0.02253 | 0.01870 | 0.8299 | 128 of 128 | 0.01237 | 0.5488 | 124 of 128 |
| rest | both | unweighted | 0.01994 | 0.01665 | 0.8352 | 128 of 128 | 0.01098 | 0.5508 | 124 of 128 |
| rest | both | count-weighted | 0.01958 | 0.01626 | 0.8301 | 128 of 128 | 0.01087 | 0.5551 | 124 of 128 |
| rest | both | trace-weighted | 0.01974 | 0.01639 | 0.8303 | 128 of 128 | 0.01094 | 0.5541 | 124 of 128 |
| diag | gate | unweighted | 0.03173 | 0.03440 | 1.0842 | 0 of 128 | 0.04439 | 1.3991 | 0 of 128 |
| diag | gate | count-weighted | 0.03174 | 0.03440 | 1.0839 | 0 of 128 | 0.04421 | 1.3930 | 0 of 128 |
| diag | gate | trace-weighted | 0.03174 | 0.03440 | 1.0838 | 0 of 128 | 0.04424 | 1.3939 | 0 of 128 |
| diag | up | unweighted | 0.03169 | 0.03436 | 1.0842 | 0 of 128 | 0.04460 | 1.4073 | 0 of 128 |
| diag | up | count-weighted | 0.03170 | 0.03436 | 1.0840 | 0 of 128 | 0.04443 | 1.4017 | 0 of 128 |
| diag | up | trace-weighted | 0.03170 | 0.03437 | 1.0840 | 0 of 128 | 0.04446 | 1.4024 | 0 of 128 |
| diag | both | unweighted | 0.03171 | 0.03438 | 1.0842 | 0 of 128 | 0.04449 | 1.4032 | 0 of 128 |
| diag | both | count-weighted | 0.03172 | 0.03438 | 1.0839 | 0 of 128 | 0.04432 | 1.3974 | 0 of 128 |
| diag | both | trace-weighted | 0.03172 | 0.03438 | 1.0839 | 0 of 128 | 0.04435 | 1.3981 | 0 of 128 |
| plain | gate | unweighted | 0.18447 | 0.19071 | 1.0338 | 0 of 128 | 0.21960 | 1.1905 | 0 of 128 |
| plain | gate | count-weighted | 0.18445 | 0.19064 | 1.0336 | 0 of 128 | 0.21915 | 1.1882 | 0 of 128 |
| plain | gate | trace-weighted | 0.18446 | 0.19066 | 1.0336 | 0 of 128 | 0.21922 | 1.1885 | 0 of 128 |
| plain | up | unweighted | 0.18442 | 0.19065 | 1.0338 | 0 of 128 | 0.22003 | 1.1931 | 0 of 128 |
| plain | up | count-weighted | 0.18439 | 0.19058 | 1.0336 | 0 of 128 | 0.21961 | 1.1910 | 0 of 128 |
| plain | up | trace-weighted | 0.18441 | 0.19061 | 1.0336 | 0 of 128 | 0.21966 | 1.1912 | 0 of 128 |
| plain | both | unweighted | 0.18445 | 0.19068 | 1.0338 | 0 of 128 | 0.21982 | 1.1918 | 0 of 128 |
| plain | both | count-weighted | 0.18442 | 0.19061 | 1.0336 | 0 of 128 | 0.21938 | 1.1896 | 0 of 128 |
| plain | both | trace-weighted | 0.18444 | 0.19063 | 1.0336 | 0 of 128 | 0.21944 | 1.1898 | 0 of 128 |
| gain | gate | unweighted | 0.99484 | 0.99501 | 1.0002 | 24 of 128 | 0.98904 | 0.9942 | 124 of 128 |
| gain | gate | count-weighted | 0.99484 | 0.99504 | 1.0002 | 24 of 128 | 0.98907 | 0.9942 | 124 of 128 |
| gain | gate | trace-weighted | 0.99484 | 0.99503 | 1.0002 | 24 of 128 | 0.98906 | 0.9942 | 124 of 128 |
| gain | up | unweighted | 0.99480 | 0.99498 | 1.0002 | 20 of 128 | 0.98895 | 0.9941 | 124 of 128 |
| gain | up | count-weighted | 0.99481 | 0.99502 | 1.0002 | 20 of 128 | 0.98900 | 0.9942 | 124 of 128 |
| gain | up | trace-weighted | 0.99480 | 0.99501 | 1.0002 | 20 of 128 | 0.98899 | 0.9942 | 124 of 128 |
| gain | both | unweighted | 0.99482 | 0.99499 | 1.0002 | 17 of 128 | 0.98899 | 0.9941 | 124 of 128 |
| gain | both | count-weighted | 0.99483 | 0.99503 | 1.0002 | 17 of 128 | 0.98904 | 0.9942 | 124 of 128 |
| gain | both | trace-weighted | 0.99482 | 0.99502 | 1.0002 | 17 of 128 | 0.98903 | 0.9942 | 124 of 128 |
| energy | gate | unweighted | 1.02370 | 1.02638 | 1.0026 | 1 of 128 | 1.02636 | 1.0026 | 0 of 128 |
| energy | gate | count-weighted | 1.02371 | 1.02642 | 1.0027 | 1 of 128 | 1.02619 | 1.0024 | 0 of 128 |
| energy | gate | trace-weighted | 1.02370 | 1.02641 | 1.0026 | 1 of 128 | 1.02621 | 1.0025 | 0 of 128 |
| energy | up | unweighted | 1.02361 | 1.02632 | 1.0026 | 1 of 128 | 1.02637 | 1.0027 | 0 of 128 |
| energy | up | count-weighted | 1.02362 | 1.02636 | 1.0027 | 1 of 128 | 1.02626 | 1.0026 | 0 of 128 |
| energy | up | trace-weighted | 1.02361 | 1.02635 | 1.0027 | 1 of 128 | 1.02626 | 1.0026 | 0 of 128 |
| energy | both | unweighted | 1.02365 | 1.02635 | 1.0026 | 1 of 128 | 1.02637 | 1.0026 | 0 of 128 |
| energy | both | count-weighted | 1.02366 | 1.02639 | 1.0027 | 1 of 128 | 1.02622 | 1.0025 | 0 of 128 |
| energy | both | trace-weighted | 1.02365 | 1.02638 | 1.0027 | 1 of 128 | 1.02624 | 1.0025 | 0 of 128 |
| ogain | gate | unweighted | 0.99822 | 0.99771 | 0.9995 | 85 of 128 | 1.00979 | 1.0116 | 0 of 128 |
| ogain | gate | count-weighted | 0.99818 | 0.99776 | 0.9996 | 85 of 128 | 1.01026 | 1.0121 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99817 | 0.99774 | 0.9996 | 85 of 128 | 1.01020 | 1.0120 | 0 of 128 |
| ogain | up | unweighted | 0.99736 | 0.99706 | 0.9997 | 77 of 128 | 1.00703 | 1.0097 | 0 of 128 |
| ogain | up | count-weighted | 0.99724 | 0.99701 | 0.9998 | 77 of 128 | 1.00734 | 1.0101 | 0 of 128 |
| ogain | up | trace-weighted | 0.99724 | 0.99701 | 0.9998 | 77 of 128 | 1.00732 | 1.0101 | 0 of 128 |
| ogain | both | unweighted | 0.99785 | 0.99743 | 0.9996 | 84 of 128 | 1.00860 | 1.0108 | 0 of 128 |
| ogain | both | count-weighted | 0.99779 | 0.99744 | 0.9997 | 84 of 128 | 1.00901 | 1.0113 | 0 of 128 |
| ogain | both | trace-weighted | 0.99778 | 0.99743 | 0.9997 | 84 of 128 | 1.00896 | 1.0112 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -5.8887e-03 | [-6.1209e-03, -5.6638e-03] |
| up | -7.6153e-03 | [-7.8417e-03, -7.3807e-03] |
| both | -6.5995e-03 | [-6.8099e-03, -6.3934e-03] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.1051e-02 | [-1.1564e-02, -1.0557e-02] | -5.1618e-03 | [-5.5230e-03, -4.8289e-03] |
| full | up | -1.4823e-02 | [-1.5239e-02, -1.4409e-02] | -7.2076e-03 | [-7.5425e-03, -6.8929e-03] |
| full | both | -1.2615e-02 | [-1.3067e-02, -1.2181e-02] | -6.0158e-03 | [-6.3617e-03, -5.6932e-03] |
| reg | gate | -9.5306e-03 | [-9.9516e-03, -9.1286e-03] | -4.1924e-03 | [-4.4903e-03, -3.9174e-03] |
| reg | up | -1.2609e-02 | [-1.2942e-02, -1.2279e-02] | -5.8185e-03 | [-6.0955e-03, -5.5588e-03] |
| reg | both | -1.0824e-02 | [-1.1189e-02, -1.0474e-02] | -4.8801e-03 | [-5.1661e-03, -4.6151e-03] |
| bias | gate | -3.3358e-03 | [-3.4871e-03, -3.1898e-03] | -4.2613e-04 | [-4.9491e-04, -3.6241e-04] |
| bias | up | -4.7058e-03 | [-4.8500e-03, -4.5627e-03] | -9.1580e-04 | [-9.8217e-04, -8.5298e-04] |
| bias | both | -3.9027e-03 | [-4.0318e-03, -3.7765e-03] | -6.3014e-04 | [-6.9607e-04, -5.6965e-04] |
| rest | gate | -7.7147e-03 | [-8.1013e-03, -7.3477e-03] | -4.7357e-03 | [-5.0381e-03, -4.4581e-03] |
| rest | up | -1.0117e-02 | [-1.0440e-02, -9.8010e-03] | -6.2918e-03 | [-6.5810e-03, -6.0192e-03] |
| rest | both | -8.7126e-03 | [-9.0627e-03, -8.3746e-03] | -5.3856e-03 | [-5.6796e-03, -5.1139e-03] |
| diag | gate | +1.2474e-02 | [+1.2212e-02, +1.2750e-02] | +9.8126e-03 | [+9.5578e-03, +1.0082e-02] |
| diag | up | +1.2733e-02 | [+1.2471e-02, +1.3009e-02] | +1.0071e-02 | [+9.8135e-03, +1.0340e-02] |
| diag | both | +1.2604e-02 | [+1.2342e-02, +1.2879e-02] | +9.9418e-03 | [+9.6861e-03, +1.0211e-02] |

## Layer 23 (IQ3_XXS; UD `.scale` absent)

- Gram layout: expert 4: max |strict lower| = 0, max |strict upper| = 4.089e+05
- Gram layout: expert 122: max |strict lower| = 0, max |strict upper| = 3.552e+05
- Gram layout: expert 93: max |strict lower| = 0, max |strict upper| = 3.633e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 3.14e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.23.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.739810 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 1655, median 11892, max 37143

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.01939 | 0.01378 | 0.7106 | 128 of 128 | 0.00905 | 0.4664 | 126 of 128 |
| full | gate | count-weighted | 0.01847 | 0.01307 | 0.7076 | 128 of 128 | 0.00876 | 0.4744 | 126 of 128 |
| full | gate | trace-weighted | 0.01875 | 0.01328 | 0.7081 | 128 of 128 | 0.00886 | 0.4726 | 126 of 128 |
| full | up | unweighted | 0.02370 | 0.01682 | 0.7097 | 128 of 128 | 0.01069 | 0.4511 | 126 of 128 |
| full | up | count-weighted | 0.02352 | 0.01661 | 0.7062 | 128 of 128 | 0.01072 | 0.4558 | 126 of 128 |
| full | up | trace-weighted | 0.02356 | 0.01666 | 0.7069 | 128 of 128 | 0.01073 | 0.4552 | 126 of 128 |
| full | both | unweighted | 0.02067 | 0.01468 | 0.7102 | 128 of 128 | 0.00952 | 0.4606 | 126 of 128 |
| full | both | count-weighted | 0.02006 | 0.01418 | 0.7070 | 128 of 128 | 0.00937 | 0.4671 | 126 of 128 |
| full | both | trace-weighted | 0.02025 | 0.01433 | 0.7075 | 128 of 128 | 0.00943 | 0.4659 | 126 of 128 |
| reg | gate | unweighted | 0.01997 | 0.01487 | 0.7447 | 128 of 128 | 0.01103 | 0.5524 | 126 of 128 |
| reg | gate | count-weighted | 0.01908 | 0.01415 | 0.7419 | 128 of 128 | 0.01065 | 0.5582 | 126 of 128 |
| reg | gate | trace-weighted | 0.01935 | 0.01436 | 0.7423 | 128 of 128 | 0.01077 | 0.5567 | 126 of 128 |
| reg | up | unweighted | 0.02415 | 0.01797 | 0.7439 | 128 of 128 | 0.01301 | 0.5386 | 126 of 128 |
| reg | up | count-weighted | 0.02399 | 0.01777 | 0.7407 | 128 of 128 | 0.01299 | 0.5416 | 126 of 128 |
| reg | up | trace-weighted | 0.02403 | 0.01781 | 0.7413 | 128 of 128 | 0.01300 | 0.5412 | 126 of 128 |
| reg | both | unweighted | 0.02125 | 0.01582 | 0.7444 | 128 of 128 | 0.01162 | 0.5470 | 126 of 128 |
| reg | both | count-weighted | 0.02066 | 0.01532 | 0.7414 | 128 of 128 | 0.01140 | 0.5515 | 126 of 128 |
| reg | both | trace-weighted | 0.02085 | 0.01547 | 0.7418 | 128 of 128 | 0.01148 | 0.5505 | 126 of 128 |
| bias | gate | unweighted | 0.00345 | 9.067e-04 | 0.2627 | 128 of 128 | 5.232e-04 | 0.1516 | 126 of 128 |
| bias | gate | count-weighted | 0.00343 | 8.756e-04 | 0.2555 | 128 of 128 | 5.275e-04 | 0.1539 | 126 of 128 |
| bias | gate | trace-weighted | 0.00345 | 8.866e-04 | 0.2566 | 128 of 128 | 5.195e-04 | 0.1504 | 126 of 128 |
| bias | up | unweighted | 0.00435 | 0.00113 | 0.2593 | 128 of 128 | 4.163e-04 | 0.0958 | 126 of 128 |
| bias | up | count-weighted | 0.00450 | 0.00114 | 0.2524 | 128 of 128 | 3.993e-04 | 0.0887 | 126 of 128 |
| bias | up | trace-weighted | 0.00447 | 0.00113 | 0.2535 | 128 of 128 | 3.954e-04 | 0.0884 | 126 of 128 |
| bias | both | unweighted | 0.00372 | 9.724e-04 | 0.2615 | 128 of 128 | 4.828e-04 | 0.1298 | 126 of 128 |
| bias | both | count-weighted | 0.00377 | 9.579e-04 | 0.2542 | 128 of 128 | 4.796e-04 | 0.1273 | 126 of 128 |
| bias | both | trace-weighted | 0.00378 | 9.641e-04 | 0.2553 | 128 of 128 | 4.732e-04 | 0.1253 | 126 of 128 |
| rest | gate | unweighted | 0.01594 | 0.01287 | 0.8076 | 128 of 128 | 0.00852 | 0.5346 | 126 of 128 |
| rest | gate | count-weighted | 0.01504 | 0.01220 | 0.8107 | 128 of 128 | 0.00824 | 0.5474 | 126 of 128 |
| rest | gate | trace-weighted | 0.01530 | 0.01239 | 0.8100 | 128 of 128 | 0.00834 | 0.5454 | 126 of 128 |
| rest | up | unweighted | 0.01935 | 0.01569 | 0.8109 | 128 of 128 | 0.01027 | 0.5309 | 126 of 128 |
| rest | up | count-weighted | 0.01902 | 0.01548 | 0.8136 | 128 of 128 | 0.01032 | 0.5427 | 126 of 128 |
| rest | up | trace-weighted | 0.01909 | 0.01552 | 0.8132 | 128 of 128 | 0.01033 | 0.5412 | 126 of 128 |
| rest | both | unweighted | 0.01695 | 0.01371 | 0.8087 | 128 of 128 | 0.00904 | 0.5331 | 126 of 128 |
| rest | both | count-weighted | 0.01629 | 0.01323 | 0.8117 | 128 of 128 | 0.00889 | 0.5457 | 126 of 128 |
| rest | both | trace-weighted | 0.01648 | 0.01336 | 0.8112 | 128 of 128 | 0.00896 | 0.5439 | 126 of 128 |
| diag | gate | unweighted | 0.03107 | 0.03374 | 1.0860 | 0 of 128 | 0.04354 | 1.4013 | 0 of 128 |
| diag | gate | count-weighted | 0.03100 | 0.03365 | 1.0857 | 0 of 128 | 0.04302 | 1.3880 | 0 of 128 |
| diag | gate | trace-weighted | 0.03103 | 0.03368 | 1.0856 | 0 of 128 | 0.04311 | 1.3893 | 0 of 128 |
| diag | up | unweighted | 0.03105 | 0.03371 | 1.0858 | 0 of 128 | 0.04374 | 1.4090 | 0 of 128 |
| diag | up | count-weighted | 0.03097 | 0.03361 | 1.0855 | 0 of 128 | 0.04323 | 1.3962 | 0 of 128 |
| diag | up | trace-weighted | 0.03100 | 0.03365 | 1.0854 | 0 of 128 | 0.04331 | 1.3972 | 0 of 128 |
| diag | both | unweighted | 0.03106 | 0.03373 | 1.0859 | 0 of 128 | 0.04364 | 1.4051 | 0 of 128 |
| diag | both | count-weighted | 0.03098 | 0.03363 | 1.0856 | 0 of 128 | 0.04313 | 1.3921 | 0 of 128 |
| diag | both | trace-weighted | 0.03101 | 0.03367 | 1.0855 | 0 of 128 | 0.04321 | 1.3933 | 0 of 128 |
| plain | gate | unweighted | 0.18493 | 0.19082 | 1.0319 | 0 of 128 | 0.22092 | 1.1946 | 0 of 128 |
| plain | gate | count-weighted | 0.18496 | 0.19082 | 1.0317 | 0 of 128 | 0.21989 | 1.1888 | 0 of 128 |
| plain | gate | trace-weighted | 0.18496 | 0.19082 | 1.0317 | 0 of 128 | 0.21996 | 1.1893 | 0 of 128 |
| plain | up | unweighted | 0.18489 | 0.19077 | 1.0318 | 0 of 128 | 0.22136 | 1.1972 | 0 of 128 |
| plain | up | count-weighted | 0.18491 | 0.19075 | 1.0316 | 0 of 128 | 0.22035 | 1.1916 | 0 of 128 |
| plain | up | trace-weighted | 0.18491 | 0.19075 | 1.0316 | 0 of 128 | 0.22041 | 1.1920 | 0 of 128 |
| plain | both | unweighted | 0.18491 | 0.19080 | 1.0318 | 0 of 128 | 0.22114 | 1.1959 | 0 of 128 |
| plain | both | count-weighted | 0.18494 | 0.19079 | 1.0316 | 0 of 128 | 0.22012 | 1.1902 | 0 of 128 |
| plain | both | trace-weighted | 0.18493 | 0.19079 | 1.0317 | 0 of 128 | 0.22019 | 1.1907 | 0 of 128 |
| gain | gate | unweighted | 0.99443 | 0.99473 | 1.0003 | 14 of 128 | 0.98853 | 0.9941 | 126 of 128 |
| gain | gate | count-weighted | 0.99442 | 0.99474 | 1.0003 | 14 of 128 | 0.98866 | 0.9942 | 126 of 128 |
| gain | gate | trace-weighted | 0.99442 | 0.99474 | 1.0003 | 14 of 128 | 0.98865 | 0.9942 | 126 of 128 |
| gain | up | unweighted | 0.99442 | 0.99474 | 1.0003 | 12 of 128 | 0.98854 | 0.9941 | 126 of 128 |
| gain | up | count-weighted | 0.99440 | 0.99474 | 1.0003 | 12 of 128 | 0.98868 | 0.9942 | 126 of 128 |
| gain | up | trace-weighted | 0.99440 | 0.99474 | 1.0003 | 12 of 128 | 0.98867 | 0.9942 | 126 of 128 |
| gain | both | unweighted | 0.99443 | 0.99473 | 1.0003 | 11 of 128 | 0.98854 | 0.9941 | 126 of 128 |
| gain | both | count-weighted | 0.99441 | 0.99474 | 1.0003 | 11 of 128 | 0.98867 | 0.9942 | 126 of 128 |
| gain | both | trace-weighted | 0.99441 | 0.99474 | 1.0003 | 11 of 128 | 0.98866 | 0.9942 | 126 of 128 |
| energy | gate | unweighted | 1.02307 | 1.02588 | 1.0027 | 0 of 128 | 1.02592 | 1.0028 | 1 of 128 |
| energy | gate | count-weighted | 1.02305 | 1.02589 | 1.0028 | 0 of 128 | 1.02568 | 1.0026 | 1 of 128 |
| energy | gate | trace-weighted | 1.02305 | 1.02589 | 1.0028 | 0 of 128 | 1.02571 | 1.0026 | 1 of 128 |
| energy | up | unweighted | 1.02302 | 1.02586 | 1.0028 | 0 of 128 | 1.02613 | 1.0030 | 0 of 128 |
| energy | up | count-weighted | 1.02299 | 1.02586 | 1.0028 | 0 of 128 | 1.02593 | 1.0029 | 0 of 128 |
| energy | up | trace-weighted | 1.02300 | 1.02586 | 1.0028 | 0 of 128 | 1.02595 | 1.0029 | 0 of 128 |
| energy | both | unweighted | 1.02304 | 1.02587 | 1.0028 | 0 of 128 | 1.02602 | 1.0029 | 0 of 128 |
| energy | both | count-weighted | 1.02302 | 1.02588 | 1.0028 | 0 of 128 | 1.02581 | 1.0027 | 0 of 128 |
| energy | both | trace-weighted | 1.02302 | 1.02588 | 1.0028 | 0 of 128 | 1.02583 | 1.0027 | 0 of 128 |
| ogain | gate | unweighted | 0.99938 | 0.99802 | 0.9986 | 109 of 128 | 1.01024 | 1.0109 | 0 of 128 |
| ogain | gate | count-weighted | 0.99950 | 0.99808 | 0.9986 | 109 of 128 | 1.01075 | 1.0112 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99944 | 0.99805 | 0.9986 | 109 of 128 | 1.01063 | 1.0112 | 0 of 128 |
| ogain | up | unweighted | 0.99840 | 0.99760 | 0.9992 | 90 of 128 | 1.00851 | 1.0101 | 0 of 128 |
| ogain | up | count-weighted | 0.99839 | 0.99758 | 0.9992 | 90 of 128 | 1.00888 | 1.0105 | 0 of 128 |
| ogain | up | trace-weighted | 0.99838 | 0.99758 | 0.9992 | 90 of 128 | 1.00883 | 1.0105 | 0 of 128 |
| ogain | both | unweighted | 0.99903 | 0.99787 | 0.9988 | 115 of 128 | 1.00960 | 1.0106 | 0 of 128 |
| ogain | both | count-weighted | 0.99910 | 0.99791 | 0.9988 | 115 of 128 | 1.01006 | 1.0110 | 0 of 128 |
| ogain | both | trace-weighted | 0.99906 | 0.99789 | 0.9988 | 115 of 128 | 1.00997 | 1.0109 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -5.4002e-03 | [-5.7469e-03, -5.0730e-03] |
| up | -6.9102e-03 | [-7.2133e-03, -6.6050e-03] |
| both | -5.8778e-03 | [-6.1648e-03, -5.5990e-03] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -9.7091e-03 | [-1.0400e-02, -9.0508e-03] | -4.3089e-03 | [-4.6992e-03, -3.9395e-03] |
| full | up | -1.2800e-02 | [-1.3368e-02, -1.2246e-02] | -5.8902e-03 | [-6.2396e-03, -5.5572e-03] |
| full | both | -1.0691e-02 | [-1.1234e-02, -1.0158e-02] | -4.8131e-03 | [-5.1436e-03, -4.4957e-03] |
| reg | gate | -8.4282e-03 | [-9.0003e-03, -7.8792e-03] | -3.5055e-03 | [-3.8231e-03, -3.2033e-03] |
| reg | up | -1.0994e-02 | [-1.1451e-02, -1.0546e-02] | -4.7748e-03 | [-5.0573e-03, -4.5060e-03] |
| reg | both | -9.2672e-03 | [-9.7145e-03, -8.8271e-03] | -3.9228e-03 | [-4.1928e-03, -3.6639e-03] |
| bias | gate | -2.8997e-03 | [-3.0955e-03, -2.7003e-03] | -3.4807e-04 | [-4.2828e-04, -2.7285e-04] |
| bias | up | -4.1019e-03 | [-4.3239e-03, -3.8756e-03] | -7.3703e-04 | [-8.0224e-04, -6.7236e-04] |
| bias | both | -3.2888e-03 | [-3.4549e-03, -3.1187e-03] | -4.7836e-04 | [-5.4100e-04, -4.1667e-04] |
| rest | gate | -6.8094e-03 | [-7.3319e-03, -6.3205e-03] | -3.9608e-03 | [-4.2825e-03, -3.6575e-03] |
| rest | up | -8.6985e-03 | [-9.1145e-03, -8.3036e-03] | -5.1532e-03 | [-5.4514e-03, -4.8684e-03] |
| rest | both | -7.4022e-03 | [-7.8330e-03, -6.9927e-03] | -4.3348e-03 | [-4.6138e-03, -4.0698e-03] |
| diag | gate | +1.2026e-02 | [+1.1759e-02, +1.2314e-02] | +9.3683e-03 | [+9.1121e-03, +9.6458e-03] |
| diag | up | +1.2268e-02 | [+1.2001e-02, +1.2553e-02] | +9.6194e-03 | [+9.3630e-03, +9.8912e-03] |
| diag | both | +1.2147e-02 | [+1.1880e-02, +1.2433e-02] | +9.4937e-03 | [+9.2377e-03, +9.7669e-03] |

## Reading (amendment 7 rule, applied literally)

Rule: supported if UD is lower than ours on the full error and higher or equal on the diagonal one in both layers; if UD is also worse on the full error, layer-local output error does not explain UD's edge.

| mean | half | layer 22 full UD < ours | layer 22 diag UD >= ours | layer 23 full UD < ours | layer 23 diag UD >= ours | branch |
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

In layer 22 (both halves, count-weighted), UD's full error is 0.01735 against our 0.02395 (ratio 0.724; UD - ours -6.600e-03, 95% CI [-6.810e-03, -6.393e-03]) and its diagonal error 0.03438 against our 0.03172 (ratio 1.084); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 22 (added after registration, not part of the rule): the input-mean part is 0.00436 ours against 0.00109 UD (ratio 0.250), the centred rest 0.01958 against 0.01626 (ratio 0.830). In layer 23 (both halves, count-weighted), UD's full error is 0.01418 against our 0.02006 (ratio 0.707; UD - ours -5.878e-03, 95% CI [-6.165e-03, -5.599e-03]) and its diagonal error 0.03363 against our 0.03098 (ratio 1.086); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 23 (added after registration, not part of the rule): the input-mean part is 0.00377 ours against 9.579e-04 UD (ratio 0.254), the centred rest 0.01629 against 0.01323 (ratio 0.812). Under the registered rule this is "supported" for both halves together, count-weighted; across the three means and three halves the branches are: supported.

## refined against ours and UD

In layer 22 (both halves, count-weighted), refined's full error is 0.01133 against our 0.02395 and UD's 0.01735 (refined - ours -1.262e-02, 95% CI [-1.307e-02, -1.218e-02]; (ours - refined) / (ours - UD) = 1.912; lower than ours in 124 of 128 experts). Bias part 4.621e-04 (ours 0.00436, UD 0.00109), rest 0.01087 (ours 0.01958, UD 0.01626). Regularised (the search objective) 0.01363 (ours 0.02445, UD 0.01851). Diagonal 0.04432 (ours 0.03172, UD 0.03438), plain 0.21938 (ours 0.18442, UD 0.19061). Gains (ours / UD / refined): plain 0.9948 / 0.9950 / 0.9890, energy 1.0237 / 1.0264 / 1.0262, output 0.9978 / 0.9974 / 1.0090. In layer 23 (both halves, count-weighted), refined's full error is 0.00937 against our 0.02006 and UD's 0.01418 (refined - ours -1.069e-02, 95% CI [-1.123e-02, -1.016e-02]; (ours - refined) / (ours - UD) = 1.819; lower than ours in 126 of 128 experts). Bias part 4.796e-04 (ours 0.00377, UD 9.579e-04), rest 0.00889 (ours 0.01629, UD 0.01323). Regularised (the search objective) 0.01140 (ours 0.02066, UD 0.01532). Diagonal 0.04313 (ours 0.03098, UD 0.03363), plain 0.22012 (ours 0.18494, UD 0.19079). Gains (ours / UD / refined): plain 0.9944 / 0.9947 / 0.9887, energy 1.0230 / 1.0259 / 1.0258, output 0.9991 / 0.9979 / 1.0101.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined (count-w.) | refined / ours | refined / UD |
|---|---|---|---|---|---|---|---|---|
| 22 | < 2304 | 4 | 0.004 | 0.02165 | 0.01582 | 0.02165 | 1.000 | 1.368 |
| 22 | >= 2304 | 124 | 0.996 | 0.02396 | 0.01735 | 0.01129 | 0.471 | 0.650 |
| 23 | < 2304 | 2 | 0.002 | 0.01961 | 0.01378 | 0.01961 | 1.000 | 1.423 |
| 23 | >= 2304 | 126 | 0.998 | 0.02006 | 0.01418 | 0.00935 | 0.466 | 0.659 |

Refinement log `models/refined/gu-refined-mixc-22-23.gguf.json` (scripts/refine-codes.py; wall time 3418 s for 256 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 22 | 124 | 4 | 4: 124 | 74235773 | 2570234 of 2571264 | 6727 | 6.94e-17 | 2.95e-17 | yes |
| 23 | 126 | 2 | 4: 126 | 73780460 | 2611177 of 2612736 | 6814 | 4.86e-17 | 2.43e-17 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 22: max abs(log - here) on reg 6.94e-17 before, 3.64e-17 after; layer 23: max abs(log - here) on reg 7.29e-17 before, 4.34e-17 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304').
