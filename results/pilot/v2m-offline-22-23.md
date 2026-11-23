# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined = `models/refined/gu-refined-mix-22-23.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/v2m-offline-22-23.tsv`.
Grams: `models/grams/gu-22-23-mix` (capture settings in its log, not restated here), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

## Layer 22 (IQ3_XXS; UD `.scale` absent)

- Gram layout: expert 50: max |strict lower| = 0, max |strict upper| = 2.808e+05
- Gram layout: expert 84: max |strict lower| = 0, max |strict upper| = 1.878e+05
- Gram layout: expert 11: max |strict lower| = 0, max |strict upper| = 2.249e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 1.07e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.22.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.627687 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 598, median 7726, max 33089

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.02165 | 0.01559 | 0.7199 | 128 of 128 | 0.01043 | 0.4816 | 120 of 128 |
| full | gate | count-weighted | 0.02068 | 0.01471 | 0.7113 | 128 of 128 | 0.00991 | 0.4789 | 120 of 128 |
| full | gate | trace-weighted | 0.02089 | 0.01484 | 0.7106 | 128 of 128 | 0.00996 | 0.4769 | 120 of 128 |
| full | up | unweighted | 0.02705 | 0.01942 | 0.7177 | 128 of 128 | 0.01246 | 0.4607 | 120 of 128 |
| full | up | count-weighted | 0.02711 | 0.01924 | 0.7098 | 128 of 128 | 0.01234 | 0.4552 | 120 of 128 |
| full | up | trace-weighted | 0.02714 | 0.01925 | 0.7092 | 128 of 128 | 0.01232 | 0.4540 | 120 of 128 |
| full | both | unweighted | 0.02390 | 0.01719 | 0.7189 | 128 of 128 | 0.01127 | 0.4714 | 120 of 128 |
| full | both | count-weighted | 0.02331 | 0.01657 | 0.7107 | 128 of 128 | 0.01090 | 0.4674 | 120 of 128 |
| full | both | trace-weighted | 0.02345 | 0.01665 | 0.7100 | 128 of 128 | 0.01092 | 0.4658 | 120 of 128 |
| reg | gate | unweighted | 0.02223 | 0.01675 | 0.7533 | 128 of 128 | 0.01263 | 0.5679 | 120 of 128 |
| reg | gate | count-weighted | 0.02130 | 0.01588 | 0.7454 | 128 of 128 | 0.01201 | 0.5638 | 120 of 128 |
| reg | gate | trace-weighted | 0.02150 | 0.01601 | 0.7448 | 128 of 128 | 0.01209 | 0.5622 | 120 of 128 |
| reg | up | unweighted | 0.02736 | 0.02056 | 0.7512 | 128 of 128 | 0.01502 | 0.5491 | 120 of 128 |
| reg | up | count-weighted | 0.02743 | 0.02041 | 0.7440 | 128 of 128 | 0.01488 | 0.5424 | 120 of 128 |
| reg | up | trace-weighted | 0.02746 | 0.02041 | 0.7434 | 128 of 128 | 0.01487 | 0.5415 | 120 of 128 |
| reg | both | unweighted | 0.02440 | 0.01835 | 0.7524 | 128 of 128 | 0.01363 | 0.5587 | 120 of 128 |
| reg | both | count-weighted | 0.02384 | 0.01776 | 0.7448 | 128 of 128 | 0.01319 | 0.5534 | 120 of 128 |
| reg | both | trace-weighted | 0.02397 | 0.01784 | 0.7442 | 128 of 128 | 0.01324 | 0.5521 | 120 of 128 |
| bias | gate | unweighted | 0.00380 | 9.104e-04 | 0.2394 | 128 of 128 | 6.833e-04 | 0.1797 | 120 of 128 |
| bias | gate | count-weighted | 0.00376 | 8.721e-04 | 0.2316 | 128 of 128 | 5.657e-04 | 0.1503 | 120 of 128 |
| bias | gate | trace-weighted | 0.00380 | 8.819e-04 | 0.2322 | 128 of 128 | 5.636e-04 | 0.1484 | 120 of 128 |
| bias | up | unweighted | 0.00481 | 0.00114 | 0.2363 | 128 of 128 | 5.388e-04 | 0.1119 | 120 of 128 |
| bias | up | count-weighted | 0.00501 | 0.00116 | 0.2309 | 128 of 128 | 3.643e-04 | 0.0727 | 120 of 128 |
| bias | up | trace-weighted | 0.00501 | 0.00116 | 0.2311 | 128 of 128 | 3.656e-04 | 0.0730 | 120 of 128 |
| bias | both | unweighted | 0.00422 | 0.00100 | 0.2376 | 128 of 128 | 6.194e-04 | 0.1467 | 120 of 128 |
| bias | both | count-weighted | 0.00427 | 9.876e-04 | 0.2311 | 128 of 128 | 4.818e-04 | 0.1128 | 120 of 128 |
| bias | both | trace-weighted | 0.00429 | 9.940e-04 | 0.2316 | 128 of 128 | 4.808e-04 | 0.1120 | 120 of 128 |
| rest | gate | unweighted | 0.01785 | 0.01468 | 0.8223 | 128 of 128 | 0.00975 | 0.5460 | 120 of 128 |
| rest | gate | count-weighted | 0.01692 | 0.01384 | 0.8181 | 128 of 128 | 0.00934 | 0.5521 | 120 of 128 |
| rest | gate | trace-weighted | 0.01709 | 0.01396 | 0.8169 | 128 of 128 | 0.00940 | 0.5498 | 120 of 128 |
| rest | up | unweighted | 0.02224 | 0.01828 | 0.8220 | 128 of 128 | 0.01193 | 0.5363 | 120 of 128 |
| rest | up | count-weighted | 0.02210 | 0.01809 | 0.8184 | 128 of 128 | 0.01198 | 0.5419 | 120 of 128 |
| rest | up | trace-weighted | 0.02213 | 0.01809 | 0.8174 | 128 of 128 | 0.01195 | 0.5402 | 120 of 128 |
| rest | both | unweighted | 0.01968 | 0.01618 | 0.8222 | 128 of 128 | 0.01065 | 0.5410 | 120 of 128 |
| rest | both | count-weighted | 0.01904 | 0.01558 | 0.8183 | 128 of 128 | 0.01041 | 0.5470 | 120 of 128 |
| rest | both | trace-weighted | 0.01916 | 0.01566 | 0.8172 | 128 of 128 | 0.01044 | 0.5450 | 120 of 128 |
| diag | gate | unweighted | 0.03155 | 0.03434 | 1.0883 | 0 of 128 | 0.04483 | 1.4209 | 0 of 128 |
| diag | gate | count-weighted | 0.03161 | 0.03434 | 1.0865 | 0 of 128 | 0.04446 | 1.4065 | 0 of 128 |
| diag | gate | trace-weighted | 0.03160 | 0.03434 | 1.0868 | 0 of 128 | 0.04451 | 1.4085 | 0 of 128 |
| diag | up | unweighted | 0.03152 | 0.03430 | 1.0883 | 0 of 128 | 0.04505 | 1.4295 | 0 of 128 |
| diag | up | count-weighted | 0.03156 | 0.03430 | 1.0867 | 0 of 128 | 0.04469 | 1.4159 | 0 of 128 |
| diag | up | trace-weighted | 0.03155 | 0.03430 | 1.0870 | 0 of 128 | 0.04473 | 1.4177 | 0 of 128 |
| diag | both | unweighted | 0.03154 | 0.03432 | 1.0883 | 0 of 128 | 0.04494 | 1.4252 | 0 of 128 |
| diag | both | count-weighted | 0.03158 | 0.03432 | 1.0866 | 0 of 128 | 0.04457 | 1.4112 | 0 of 128 |
| diag | both | trace-weighted | 0.03158 | 0.03432 | 1.0869 | 0 of 128 | 0.04462 | 1.4131 | 0 of 128 |
| plain | gate | unweighted | 0.18447 | 0.19071 | 1.0338 | 0 of 128 | 0.22126 | 1.1994 | 0 of 128 |
| plain | gate | count-weighted | 0.18441 | 0.19060 | 1.0336 | 0 of 128 | 0.22022 | 1.1942 | 0 of 128 |
| plain | gate | trace-weighted | 0.18443 | 0.19063 | 1.0336 | 0 of 128 | 0.22039 | 1.1950 | 0 of 128 |
| plain | up | unweighted | 0.18442 | 0.19065 | 1.0338 | 0 of 128 | 0.22170 | 1.2021 | 0 of 128 |
| plain | up | count-weighted | 0.18436 | 0.19055 | 1.0336 | 0 of 128 | 0.22071 | 1.1971 | 0 of 128 |
| plain | up | trace-weighted | 0.18438 | 0.19058 | 1.0336 | 0 of 128 | 0.22087 | 1.1979 | 0 of 128 |
| plain | both | unweighted | 0.18445 | 0.19068 | 1.0338 | 0 of 128 | 0.22148 | 1.2008 | 0 of 128 |
| plain | both | count-weighted | 0.18439 | 0.19058 | 1.0336 | 0 of 128 | 0.22046 | 1.1957 | 0 of 128 |
| plain | both | trace-weighted | 0.18440 | 0.19060 | 1.0336 | 0 of 128 | 0.22063 | 1.1964 | 0 of 128 |
| gain | gate | unweighted | 0.99484 | 0.99501 | 1.0002 | 24 of 128 | 0.98883 | 0.9940 | 120 of 128 |
| gain | gate | count-weighted | 0.99486 | 0.99506 | 1.0002 | 24 of 128 | 0.98893 | 0.9940 | 120 of 128 |
| gain | gate | trace-weighted | 0.99486 | 0.99505 | 1.0002 | 24 of 128 | 0.98891 | 0.9940 | 120 of 128 |
| gain | up | unweighted | 0.99480 | 0.99498 | 1.0002 | 20 of 128 | 0.98872 | 0.9939 | 120 of 128 |
| gain | up | count-weighted | 0.99482 | 0.99504 | 1.0002 | 20 of 128 | 0.98884 | 0.9940 | 120 of 128 |
| gain | up | trace-weighted | 0.99482 | 0.99503 | 1.0002 | 20 of 128 | 0.98882 | 0.9940 | 120 of 128 |
| gain | both | unweighted | 0.99482 | 0.99499 | 1.0002 | 17 of 128 | 0.98877 | 0.9939 | 120 of 128 |
| gain | both | count-weighted | 0.99484 | 0.99505 | 1.0002 | 17 of 128 | 0.98889 | 0.9940 | 120 of 128 |
| gain | both | trace-weighted | 0.99484 | 0.99504 | 1.0002 | 17 of 128 | 0.98887 | 0.9940 | 120 of 128 |
| energy | gate | unweighted | 1.02370 | 1.02638 | 1.0026 | 1 of 128 | 1.02673 | 1.0030 | 0 of 128 |
| energy | gate | count-weighted | 1.02374 | 1.02646 | 1.0027 | 1 of 128 | 1.02640 | 1.0026 | 0 of 128 |
| energy | gate | trace-weighted | 1.02373 | 1.02644 | 1.0027 | 1 of 128 | 1.02644 | 1.0027 | 0 of 128 |
| energy | up | unweighted | 1.02361 | 1.02632 | 1.0026 | 1 of 128 | 1.02670 | 1.0030 | 0 of 128 |
| energy | up | count-weighted | 1.02364 | 1.02639 | 1.0027 | 1 of 128 | 1.02643 | 1.0027 | 0 of 128 |
| energy | up | trace-weighted | 1.02363 | 1.02637 | 1.0027 | 1 of 128 | 1.02647 | 1.0028 | 0 of 128 |
| energy | both | unweighted | 1.02365 | 1.02635 | 1.0026 | 1 of 128 | 1.02672 | 1.0030 | 0 of 128 |
| energy | both | count-weighted | 1.02369 | 1.02642 | 1.0027 | 1 of 128 | 1.02642 | 1.0027 | 0 of 128 |
| energy | both | trace-weighted | 1.02368 | 1.02641 | 1.0027 | 1 of 128 | 1.02645 | 1.0027 | 0 of 128 |
| ogain | gate | unweighted | 0.99833 | 0.99779 | 0.9995 | 86 of 128 | 1.00949 | 1.0112 | 0 of 128 |
| ogain | gate | count-weighted | 0.99829 | 0.99786 | 0.9996 | 86 of 128 | 1.01036 | 1.0121 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99829 | 0.99786 | 0.9996 | 86 of 128 | 1.01033 | 1.0121 | 0 of 128 |
| ogain | up | unweighted | 0.99748 | 0.99710 | 0.9996 | 73 of 128 | 1.00695 | 1.0095 | 0 of 128 |
| ogain | up | count-weighted | 0.99728 | 0.99704 | 0.9998 | 73 of 128 | 1.00743 | 1.0102 | 0 of 128 |
| ogain | up | trace-weighted | 0.99729 | 0.99704 | 0.9997 | 73 of 128 | 1.00746 | 1.0102 | 0 of 128 |
| ogain | both | unweighted | 0.99797 | 0.99749 | 0.9995 | 84 of 128 | 1.00840 | 1.0105 | 0 of 128 |
| ogain | both | count-weighted | 0.99787 | 0.99752 | 0.9996 | 84 of 128 | 1.00913 | 1.0113 | 0 of 128 |
| ogain | both | trace-weighted | 0.99787 | 0.99752 | 0.9996 | 84 of 128 | 1.00912 | 1.0113 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -5.9706e-03 | [-6.2519e-03, -5.7130e-03] |
| up | -7.8664e-03 | [-8.1226e-03, -7.6023e-03] |
| both | -6.7439e-03 | [-6.9953e-03, -6.5100e-03] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.0778e-02 | [-1.1323e-02, -1.0268e-02] | -4.8072e-03 | [-5.1473e-03, -4.4948e-03] |
| full | up | -1.4769e-02 | [-1.5213e-02, -1.4329e-02] | -6.9023e-03 | [-7.2184e-03, -6.5961e-03] |
| full | both | -1.2414e-02 | [-1.2891e-02, -1.1973e-02] | -5.6701e-03 | [-5.9888e-03, -5.3744e-03] |
| reg | gate | -9.2922e-03 | [-9.7346e-03, -8.8774e-03] | -3.8688e-03 | [-4.1444e-03, -3.6141e-03] |
| reg | up | -1.2551e-02 | [-1.2905e-02, -1.2200e-02] | -5.5277e-03 | [-5.7854e-03, -5.2753e-03] |
| reg | both | -1.0647e-02 | [-1.1030e-02, -1.0289e-02] | -4.5616e-03 | [-4.8202e-03, -4.3183e-03] |
| bias | gate | -3.1992e-03 | [-3.3503e-03, -3.0531e-03] | -3.0633e-04 | [-3.6050e-04, -2.5346e-04] |
| bias | up | -4.6463e-03 | [-4.8089e-03, -4.4788e-03] | -7.9235e-04 | [-8.4653e-04, -7.3222e-04] |
| bias | both | -3.7910e-03 | [-3.9215e-03, -3.6588e-03] | -5.0585e-04 | [-5.5398e-04, -4.5706e-04] |
| rest | gate | -7.5787e-03 | [-8.0137e-03, -7.1850e-03] | -4.5009e-03 | [-4.7969e-03, -4.2288e-03] |
| rest | up | -1.0122e-02 | [-1.0479e-02, -9.7869e-03] | -6.1099e-03 | [-6.3974e-03, -5.8378e-03] |
| rest | both | -8.6230e-03 | [-9.0113e-03, -8.2709e-03] | -5.1642e-03 | [-5.4512e-03, -4.9013e-03] |
| diag | gate | +1.2848e-02 | [+1.2521e-02, +1.3219e-02] | +1.0113e-02 | [+9.8034e-03, +1.0462e-02] |
| diag | up | +1.3125e-02 | [+1.2793e-02, +1.3499e-02] | +1.0388e-02 | [+1.0072e-02, +1.0738e-02] |
| diag | both | +1.2987e-02 | [+1.2657e-02, +1.3359e-02] | +1.0251e-02 | [+9.9368e-03, +1.0599e-02] |

## Layer 23 (IQ3_XXS; UD `.scale` absent)

- Gram layout: expert 4: max |strict lower| = 0, max |strict upper| = 4.394e+05
- Gram layout: expert 122: max |strict lower| = 0, max |strict upper| = 3.759e+05
- Gram layout: expert 39: max |strict lower| = 0, max |strict upper| = 3.47e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 6.33e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.23.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.745889 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 1340, median 7285, max 39795

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.01927 | 0.01334 | 0.6921 | 128 of 128 | 0.00916 | 0.4753 | 121 of 128 |
| full | gate | count-weighted | 0.01785 | 0.01238 | 0.6937 | 128 of 128 | 0.00847 | 0.4743 | 121 of 128 |
| full | gate | trace-weighted | 0.01808 | 0.01253 | 0.6928 | 128 of 128 | 0.00854 | 0.4724 | 121 of 128 |
| full | up | unweighted | 0.02362 | 0.01633 | 0.6917 | 128 of 128 | 0.01079 | 0.4568 | 121 of 128 |
| full | up | count-weighted | 0.02322 | 0.01606 | 0.6918 | 128 of 128 | 0.01052 | 0.4529 | 121 of 128 |
| full | up | trace-weighted | 0.02325 | 0.01607 | 0.6912 | 128 of 128 | 0.01051 | 0.4521 | 121 of 128 |
| full | both | unweighted | 0.02059 | 0.01425 | 0.6920 | 128 of 128 | 0.00965 | 0.4686 | 121 of 128 |
| full | both | count-weighted | 0.01965 | 0.01362 | 0.6930 | 128 of 128 | 0.00915 | 0.4658 | 121 of 128 |
| full | both | trace-weighted | 0.01980 | 0.01371 | 0.6922 | 128 of 128 | 0.00920 | 0.4644 | 121 of 128 |
| reg | gate | unweighted | 0.01985 | 0.01446 | 0.7284 | 128 of 128 | 0.01114 | 0.5613 | 121 of 128 |
| reg | gate | count-weighted | 0.01848 | 0.01348 | 0.7297 | 128 of 128 | 0.01033 | 0.5590 | 121 of 128 |
| reg | gate | trace-weighted | 0.01870 | 0.01363 | 0.7289 | 128 of 128 | 0.01043 | 0.5575 | 121 of 128 |
| reg | up | unweighted | 0.02407 | 0.01752 | 0.7279 | 128 of 128 | 0.01312 | 0.5449 | 121 of 128 |
| reg | up | count-weighted | 0.02369 | 0.01724 | 0.7279 | 128 of 128 | 0.01279 | 0.5400 | 121 of 128 |
| reg | up | trace-weighted | 0.02372 | 0.01725 | 0.7274 | 128 of 128 | 0.01279 | 0.5394 | 121 of 128 |
| reg | both | unweighted | 0.02117 | 0.01542 | 0.7283 | 128 of 128 | 0.01175 | 0.5552 | 121 of 128 |
| reg | both | count-weighted | 0.02027 | 0.01477 | 0.7289 | 128 of 128 | 0.01117 | 0.5512 | 121 of 128 |
| reg | both | trace-weighted | 0.02041 | 0.01487 | 0.7283 | 128 of 128 | 0.01123 | 0.5502 | 121 of 128 |
| bias | gate | unweighted | 0.00344 | 8.323e-04 | 0.2418 | 128 of 128 | 6.560e-04 | 0.1906 | 121 of 128 |
| bias | gate | count-weighted | 0.00340 | 8.070e-04 | 0.2372 | 128 of 128 | 5.688e-04 | 0.1672 | 121 of 128 |
| bias | gate | trace-weighted | 0.00343 | 8.146e-04 | 0.2375 | 128 of 128 | 5.669e-04 | 0.1653 | 121 of 128 |
| bias | up | unweighted | 0.00436 | 0.00105 | 0.2407 | 128 of 128 | 5.721e-04 | 0.1313 | 121 of 128 |
| bias | up | count-weighted | 0.00458 | 0.00108 | 0.2368 | 128 of 128 | 4.473e-04 | 0.0977 | 121 of 128 |
| bias | up | trace-weighted | 0.00456 | 0.00108 | 0.2370 | 128 of 128 | 4.489e-04 | 0.0985 | 121 of 128 |
| bias | both | unweighted | 0.00372 | 8.975e-04 | 0.2412 | 128 of 128 | 6.247e-04 | 0.1679 | 121 of 128 |
| bias | both | count-weighted | 0.00380 | 9.000e-04 | 0.2370 | 128 of 128 | 5.223e-04 | 0.1375 | 121 of 128 |
| bias | both | trace-weighted | 0.00381 | 9.035e-04 | 0.2373 | 128 of 128 | 5.218e-04 | 0.1370 | 121 of 128 |
| rest | gate | unweighted | 0.01583 | 0.01250 | 0.7900 | 128 of 128 | 0.00850 | 0.5372 | 121 of 128 |
| rest | gate | count-weighted | 0.01445 | 0.01158 | 0.8012 | 128 of 128 | 0.00790 | 0.5466 | 121 of 128 |
| rest | gate | trace-weighted | 0.01465 | 0.01171 | 0.7994 | 128 of 128 | 0.00797 | 0.5443 | 121 of 128 |
| rest | up | unweighted | 0.01926 | 0.01529 | 0.7937 | 128 of 128 | 0.01022 | 0.5304 | 121 of 128 |
| rest | up | count-weighted | 0.01864 | 0.01498 | 0.8035 | 128 of 128 | 0.01007 | 0.5401 | 121 of 128 |
| rest | up | trace-weighted | 0.01869 | 0.01499 | 0.8020 | 128 of 128 | 0.01006 | 0.5384 | 121 of 128 |
| rest | both | unweighted | 0.01687 | 0.01335 | 0.7914 | 128 of 128 | 0.00902 | 0.5350 | 121 of 128 |
| rest | both | count-weighted | 0.01585 | 0.01272 | 0.8023 | 128 of 128 | 0.00863 | 0.5444 | 121 of 128 |
| rest | both | trace-weighted | 0.01599 | 0.01280 | 0.8006 | 128 of 128 | 0.00867 | 0.5423 | 121 of 128 |
| diag | gate | unweighted | 0.03083 | 0.03362 | 1.0903 | 0 of 128 | 0.04367 | 1.4164 | 0 of 128 |
| diag | gate | count-weighted | 0.03078 | 0.03350 | 1.0884 | 0 of 128 | 0.04305 | 1.3987 | 0 of 128 |
| diag | gate | trace-weighted | 0.03079 | 0.03352 | 1.0885 | 0 of 128 | 0.04313 | 1.4005 | 0 of 128 |
| diag | up | unweighted | 0.03081 | 0.03359 | 1.0901 | 0 of 128 | 0.04390 | 1.4246 | 0 of 128 |
| diag | up | count-weighted | 0.03074 | 0.03345 | 1.0881 | 0 of 128 | 0.04328 | 1.4076 | 0 of 128 |
| diag | up | trace-weighted | 0.03076 | 0.03348 | 1.0883 | 0 of 128 | 0.04335 | 1.4092 | 0 of 128 |
| diag | both | unweighted | 0.03082 | 0.03360 | 1.0902 | 0 of 128 | 0.04378 | 1.4205 | 0 of 128 |
| diag | both | count-weighted | 0.03076 | 0.03348 | 1.0882 | 0 of 128 | 0.04316 | 1.4031 | 0 of 128 |
| diag | both | trace-weighted | 0.03078 | 0.03350 | 1.0884 | 0 of 128 | 0.04324 | 1.4049 | 0 of 128 |
| plain | gate | unweighted | 0.18493 | 0.19082 | 1.0319 | 0 of 128 | 0.22208 | 1.2008 | 0 of 128 |
| plain | gate | count-weighted | 0.18498 | 0.19083 | 1.0316 | 0 of 128 | 0.22080 | 1.1937 | 0 of 128 |
| plain | gate | trace-weighted | 0.18497 | 0.19083 | 1.0317 | 0 of 128 | 0.22094 | 1.1944 | 0 of 128 |
| plain | up | unweighted | 0.18489 | 0.19077 | 1.0318 | 0 of 128 | 0.22256 | 1.2037 | 0 of 128 |
| plain | up | count-weighted | 0.18492 | 0.19076 | 1.0316 | 0 of 128 | 0.22131 | 1.1968 | 0 of 128 |
| plain | up | trace-weighted | 0.18492 | 0.19077 | 1.0316 | 0 of 128 | 0.22143 | 1.1974 | 0 of 128 |
| plain | both | unweighted | 0.18491 | 0.19080 | 1.0318 | 0 of 128 | 0.22232 | 1.2023 | 0 of 128 |
| plain | both | count-weighted | 0.18495 | 0.19080 | 1.0316 | 0 of 128 | 0.22106 | 1.1952 | 0 of 128 |
| plain | both | trace-weighted | 0.18495 | 0.19080 | 1.0317 | 0 of 128 | 0.22118 | 1.1959 | 0 of 128 |
| gain | gate | unweighted | 0.99443 | 0.99473 | 1.0003 | 14 of 128 | 0.98843 | 0.9940 | 121 of 128 |
| gain | gate | count-weighted | 0.99441 | 0.99474 | 1.0003 | 14 of 128 | 0.98856 | 0.9941 | 121 of 128 |
| gain | gate | trace-weighted | 0.99442 | 0.99473 | 1.0003 | 14 of 128 | 0.98855 | 0.9941 | 121 of 128 |
| gain | up | unweighted | 0.99442 | 0.99474 | 1.0003 | 12 of 128 | 0.98842 | 0.9940 | 121 of 128 |
| gain | up | count-weighted | 0.99439 | 0.99473 | 1.0003 | 12 of 128 | 0.98855 | 0.9941 | 121 of 128 |
| gain | up | trace-weighted | 0.99439 | 0.99473 | 1.0003 | 12 of 128 | 0.98854 | 0.9941 | 121 of 128 |
| gain | both | unweighted | 0.99443 | 0.99473 | 1.0003 | 11 of 128 | 0.98843 | 0.9940 | 121 of 128 |
| gain | both | count-weighted | 0.99440 | 0.99473 | 1.0003 | 11 of 128 | 0.98856 | 0.9941 | 121 of 128 |
| gain | both | trace-weighted | 0.99440 | 0.99473 | 1.0003 | 11 of 128 | 0.98855 | 0.9941 | 121 of 128 |
| energy | gate | unweighted | 1.02307 | 1.02588 | 1.0027 | 0 of 128 | 1.02630 | 1.0032 | 0 of 128 |
| energy | gate | count-weighted | 1.02305 | 1.02589 | 1.0028 | 0 of 128 | 1.02591 | 1.0028 | 0 of 128 |
| energy | gate | trace-weighted | 1.02305 | 1.02589 | 1.0028 | 0 of 128 | 1.02596 | 1.0028 | 0 of 128 |
| energy | up | unweighted | 1.02302 | 1.02586 | 1.0028 | 0 of 128 | 1.02649 | 1.0034 | 0 of 128 |
| energy | up | count-weighted | 1.02297 | 1.02586 | 1.0028 | 0 of 128 | 1.02613 | 1.0031 | 0 of 128 |
| energy | up | trace-weighted | 1.02298 | 1.02586 | 1.0028 | 0 of 128 | 1.02616 | 1.0031 | 0 of 128 |
| energy | both | unweighted | 1.02304 | 1.02587 | 1.0028 | 0 of 128 | 1.02639 | 1.0033 | 0 of 128 |
| energy | both | count-weighted | 1.02301 | 1.02587 | 1.0028 | 0 of 128 | 1.02602 | 1.0029 | 0 of 128 |
| energy | both | trace-weighted | 1.02301 | 1.02587 | 1.0028 | 0 of 128 | 1.02606 | 1.0030 | 0 of 128 |
| ogain | gate | unweighted | 0.99959 | 0.99806 | 0.9985 | 117 of 128 | 1.01003 | 1.0104 | 0 of 128 |
| ogain | gate | count-weighted | 0.99973 | 0.99816 | 0.9984 | 117 of 128 | 1.01096 | 1.0112 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99968 | 0.99813 | 0.9984 | 117 of 128 | 1.01089 | 1.0112 | 0 of 128 |
| ogain | up | unweighted | 0.99860 | 0.99767 | 0.9991 | 98 of 128 | 1.00838 | 1.0098 | 0 of 128 |
| ogain | up | count-weighted | 0.99855 | 0.99764 | 0.9991 | 98 of 128 | 1.00910 | 1.0106 | 0 of 128 |
| ogain | up | trace-weighted | 0.99856 | 0.99764 | 0.9991 | 98 of 128 | 1.00908 | 1.0105 | 0 of 128 |
| ogain | both | unweighted | 0.99923 | 0.99793 | 0.9987 | 118 of 128 | 1.00941 | 1.0102 | 0 of 128 |
| ogain | both | count-weighted | 0.99928 | 0.99797 | 0.9987 | 118 of 128 | 1.01024 | 1.0110 | 0 of 128 |
| ogain | both | trace-weighted | 0.99926 | 0.99795 | 0.9987 | 118 of 128 | 1.01019 | 1.0109 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -5.4678e-03 | [-5.8053e-03, -5.1602e-03] |
| up | -7.1562e-03 | [-7.4864e-03, -6.8415e-03] |
| both | -6.0323e-03 | [-6.3237e-03, -5.7592e-03] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -9.3847e-03 | [-1.0015e-02, -8.8092e-03] | -3.9169e-03 | [-4.2583e-03, -3.5995e-03] |
| full | up | -1.2703e-02 | [-1.3302e-02, -1.2146e-02] | -5.5464e-03 | [-5.8878e-03, -5.2427e-03] |
| full | both | -1.0497e-02 | [-1.1023e-02, -1.0013e-02] | -4.4649e-03 | [-4.7580e-03, -4.1851e-03] |
| reg | gate | -8.1494e-03 | [-8.6706e-03, -7.6684e-03] | -3.1537e-03 | [-3.4294e-03, -2.8962e-03] |
| reg | up | -1.0899e-02 | [-1.1382e-02, -1.0449e-02] | -4.4513e-03 | [-4.7188e-03, -4.2060e-03] |
| reg | both | -9.0952e-03 | [-9.5270e-03, -8.6946e-03] | -3.6021e-03 | [-3.8384e-03, -3.3744e-03] |
| bias | gate | -2.8337e-03 | [-3.0196e-03, -2.6481e-03] | -2.3814e-04 | [-3.0089e-04, -1.7472e-04] |
| bias | up | -4.1298e-03 | [-4.3669e-03, -3.8813e-03] | -6.3653e-04 | [-6.9698e-04, -5.7135e-04] |
| bias | both | -3.2756e-03 | [-3.4430e-03, -3.0981e-03] | -3.7773e-04 | [-4.3063e-04, -3.2289e-04] |
| rest | gate | -6.5510e-03 | [-7.0484e-03, -6.1140e-03] | -3.6788e-03 | [-3.9702e-03, -3.4113e-03] |
| rest | up | -8.5729e-03 | [-9.0227e-03, -8.1770e-03] | -4.9099e-03 | [-5.2078e-03, -4.6438e-03] |
| rest | both | -7.2217e-03 | [-7.6425e-03, -6.8431e-03] | -4.0872e-03 | [-4.3454e-03, -3.8449e-03] |
| diag | gate | +1.2270e-02 | [+1.1942e-02, +1.2652e-02] | +9.5508e-03 | [+9.2407e-03, +9.9039e-03] |
| diag | up | +1.2531e-02 | [+1.2200e-02, +1.2917e-02] | +9.8213e-03 | [+9.5136e-03, +1.0179e-02] |
| diag | both | +1.2400e-02 | [+1.2072e-02, +1.2782e-02] | +9.6860e-03 | [+9.3816e-03, +1.0040e-02] |

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

In layer 22 (both halves, count-weighted), UD's full error is 0.01657 against our 0.02331 (ratio 0.711; UD - ours -6.744e-03, 95% CI [-6.995e-03, -6.510e-03]) and its diagonal error 0.03432 against our 0.03158 (ratio 1.087); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 22 (added after registration, not part of the rule): the input-mean part is 0.00427 ours against 9.876e-04 UD (ratio 0.231), the centred rest 0.01904 against 0.01558 (ratio 0.818). In layer 23 (both halves, count-weighted), UD's full error is 0.01362 against our 0.01965 (ratio 0.693; UD - ours -6.032e-03, 95% CI [-6.324e-03, -5.759e-03]) and its diagonal error 0.03348 against our 0.03076 (ratio 1.088); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 23 (added after registration, not part of the rule): the input-mean part is 0.00380 ours against 9.000e-04 UD (ratio 0.237), the centred rest 0.01585 against 0.01272 (ratio 0.802). Under the registered rule this is "supported" for both halves together, count-weighted; across the three means and three halves the branches are: supported.

## refined against ours and UD

In layer 22 (both halves, count-weighted), refined's full error is 0.01090 against our 0.02331 and UD's 0.01657 (refined - ours -1.241e-02, 95% CI [-1.289e-02, -1.197e-02]; (ours - refined) / (ours - UD) = 1.841; lower than ours in 120 of 128 experts). Bias part 4.818e-04 (ours 0.00427, UD 9.876e-04), rest 0.01041 (ours 0.01904, UD 0.01558). Regularised (the search objective) 0.01319 (ours 0.02384, UD 0.01776). Diagonal 0.04457 (ours 0.03158, UD 0.03432), plain 0.22046 (ours 0.18439, UD 0.19058). Gains (ours / UD / refined): plain 0.9948 / 0.9951 / 0.9889, energy 1.0237 / 1.0264 / 1.0264, output 0.9979 / 0.9975 / 1.0091. In layer 23 (both halves, count-weighted), refined's full error is 0.00915 against our 0.01965 and UD's 0.01362 (refined - ours -1.050e-02, 95% CI [-1.102e-02, -1.001e-02]; (ours - refined) / (ours - UD) = 1.740; lower than ours in 121 of 128 experts). Bias part 5.223e-04 (ours 0.00380, UD 9.000e-04), rest 0.00863 (ours 0.01585, UD 0.01272). Regularised (the search objective) 0.01117 (ours 0.02027, UD 0.01477). Diagonal 0.04316 (ours 0.03076, UD 0.03348), plain 0.22106 (ours 0.18495, UD 0.19080). Gains (ours / UD / refined): plain 0.9944 / 0.9947 / 0.9886, energy 1.0230 / 1.0259 / 1.0260, output 0.9993 / 0.9980 / 1.0102.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined (count-w.) | refined / ours | refined / UD |
|---|---|---|---|---|---|---|---|---|
| 22 | < 2304 | 8 | 0.010 | 0.02332 | 0.01736 | 0.02332 | 1.000 | 1.344 |
| 22 | >= 2304 | 120 | 0.990 | 0.02331 | 0.01656 | 0.01076 | 0.462 | 0.650 |
| 23 | < 2304 | 7 | 0.011 | 0.02454 | 0.01711 | 0.02454 | 1.000 | 1.435 |
| 23 | >= 2304 | 121 | 0.989 | 0.01960 | 0.01358 | 0.00899 | 0.459 | 0.662 |

Refinement log `models/refined/gu-refined-mix-22-23.gguf.json` (scripts/refine-codes.py; wall time 4187 s for 256 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 22 | 120 | 8 | 4: 120 | 73290151 | 2487429 of 2488320 | 7100 | 2.89e-10 | 3.08e-10 | yes |
| 23 | 121 | 7 | 4: 121 | 71857758 | 2507711 of 2509056 | 9474 | 4.52e-10 | 2.77e-10 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 22: max abs(log - here) on reg 2.65e-10 before, 3.06e-10 after; layer 23: max abs(log - here) on reg 4.16e-10 before, 2.64e-10 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304').
