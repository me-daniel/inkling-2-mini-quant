# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined = `models/refined/gu-refined-mixc-1-2-3-5.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/v2c-offline-1-2-3-5.tsv`.
Grams: `models/grams/gu-1-2-3-5-mixc` (capture settings in its log, not restated here), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

## Layer 1 (IQ2_S; UD `.scale` present, min 3.958e-05, max 0.0001538)

- Gram layout: expert 84: max |strict lower| = 0, max |strict upper| = 1.282e+05
- Gram layout: expert 19: max |strict lower| = 0, max |strict upper| = 7.931e+04
- Gram layout: expert 9: max |strict lower| = 0, max |strict upper| = 1.037e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 1.99e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.1.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.743329 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 1052, median 10836, max 48176

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03527 | 0.02290 | 0.6492 | 128 of 128 | 0.01545 | 0.4380 | 127 of 128 |
| full | gate | count-weighted | 0.03524 | 0.02267 | 0.6432 | 128 of 128 | 0.01575 | 0.4470 | 127 of 128 |
| full | gate | trace-weighted | 0.03518 | 0.02265 | 0.6439 | 128 of 128 | 0.01581 | 0.4494 | 127 of 128 |
| full | up | unweighted | 0.04433 | 0.02884 | 0.6505 | 128 of 128 | 0.01886 | 0.4255 | 127 of 128 |
| full | up | count-weighted | 0.04443 | 0.02865 | 0.6447 | 128 of 128 | 0.01928 | 0.4340 | 127 of 128 |
| full | up | trace-weighted | 0.04446 | 0.02870 | 0.6455 | 128 of 128 | 0.01940 | 0.4364 | 127 of 128 |
| full | both | unweighted | 0.03921 | 0.02548 | 0.6499 | 128 of 128 | 0.01693 | 0.4318 | 127 of 128 |
| full | both | count-weighted | 0.03924 | 0.02527 | 0.6440 | 128 of 128 | 0.01729 | 0.4406 | 127 of 128 |
| full | both | trace-weighted | 0.03921 | 0.02528 | 0.6448 | 128 of 128 | 0.01737 | 0.4430 | 127 of 128 |
| reg | gate | unweighted | 0.03669 | 0.02524 | 0.6881 | 128 of 128 | 0.01956 | 0.5330 | 127 of 128 |
| reg | gate | count-weighted | 0.03667 | 0.02503 | 0.6826 | 128 of 128 | 0.01978 | 0.5393 | 127 of 128 |
| reg | gate | trace-weighted | 0.03661 | 0.02501 | 0.6832 | 128 of 128 | 0.01981 | 0.5412 | 127 of 128 |
| reg | up | unweighted | 0.04551 | 0.03137 | 0.6894 | 128 of 128 | 0.02377 | 0.5222 | 127 of 128 |
| reg | up | count-weighted | 0.04561 | 0.03121 | 0.6841 | 128 of 128 | 0.02409 | 0.5282 | 127 of 128 |
| reg | up | trace-weighted | 0.04564 | 0.03125 | 0.6848 | 128 of 128 | 0.02419 | 0.5301 | 127 of 128 |
| reg | both | unweighted | 0.04056 | 0.02794 | 0.6887 | 128 of 128 | 0.02140 | 0.5277 | 127 of 128 |
| reg | both | count-weighted | 0.04059 | 0.02774 | 0.6834 | 128 of 128 | 0.02167 | 0.5339 | 127 of 128 |
| reg | both | trace-weighted | 0.04056 | 0.02774 | 0.6840 | 128 of 128 | 0.02173 | 0.5358 | 127 of 128 |
| bias | gate | unweighted | 0.00515 | 9.861e-04 | 0.1916 | 128 of 128 | 4.307e-04 | 0.0837 | 127 of 128 |
| bias | gate | count-weighted | 0.00512 | 9.483e-04 | 0.1852 | 128 of 128 | 3.914e-04 | 0.0764 | 127 of 128 |
| bias | gate | trace-weighted | 0.00508 | 9.318e-04 | 0.1833 | 128 of 128 | 3.885e-04 | 0.0764 | 127 of 128 |
| bias | up | unweighted | 0.00637 | 0.00124 | 0.1946 | 128 of 128 | 2.233e-04 | 0.0350 | 127 of 128 |
| bias | up | count-weighted | 0.00637 | 0.00121 | 0.1893 | 128 of 128 | 1.743e-04 | 0.0273 | 127 of 128 |
| bias | up | trace-weighted | 0.00634 | 0.00119 | 0.1876 | 128 of 128 | 1.714e-04 | 0.0270 | 127 of 128 |
| bias | both | unweighted | 0.00568 | 0.00110 | 0.1932 | 128 of 128 | 3.405e-04 | 0.0600 | 127 of 128 |
| bias | both | count-weighted | 0.00566 | 0.00106 | 0.1872 | 128 of 128 | 2.971e-04 | 0.0525 | 127 of 128 |
| bias | both | trace-weighted | 0.00563 | 0.00104 | 0.1854 | 128 of 128 | 2.944e-04 | 0.0523 | 127 of 128 |
| rest | gate | unweighted | 0.03012 | 0.02191 | 0.7274 | 128 of 128 | 0.01502 | 0.4985 | 127 of 128 |
| rest | gate | count-weighted | 0.03012 | 0.02172 | 0.7211 | 128 of 128 | 0.01536 | 0.5099 | 127 of 128 |
| rest | gate | trace-weighted | 0.03010 | 0.02172 | 0.7217 | 128 of 128 | 0.01542 | 0.5124 | 127 of 128 |
| rest | up | unweighted | 0.03796 | 0.02760 | 0.7270 | 128 of 128 | 0.01864 | 0.4910 | 127 of 128 |
| rest | up | count-weighted | 0.03806 | 0.02744 | 0.7210 | 128 of 128 | 0.01911 | 0.5021 | 127 of 128 |
| rest | up | trace-weighted | 0.03811 | 0.02751 | 0.7217 | 128 of 128 | 0.01923 | 0.5045 | 127 of 128 |
| rest | both | unweighted | 0.03353 | 0.02439 | 0.7272 | 128 of 128 | 0.01659 | 0.4948 | 127 of 128 |
| rest | both | count-weighted | 0.03357 | 0.02421 | 0.7211 | 128 of 128 | 0.01699 | 0.5061 | 127 of 128 |
| rest | both | trace-weighted | 0.03358 | 0.02423 | 0.7218 | 128 of 128 | 0.01708 | 0.5085 | 127 of 128 |
| diag | gate | unweighted | 0.06243 | 0.06732 | 1.0782 | 0 of 128 | 0.09286 | 1.4874 | 0 of 128 |
| diag | gate | count-weighted | 0.06248 | 0.06735 | 1.0780 | 0 of 128 | 0.09171 | 1.4678 | 0 of 128 |
| diag | gate | trace-weighted | 0.06251 | 0.06738 | 1.0780 | 0 of 128 | 0.09153 | 1.4643 | 0 of 128 |
| diag | up | unweighted | 0.06239 | 0.06728 | 1.0783 | 0 of 128 | 0.09310 | 1.4922 | 0 of 128 |
| diag | up | count-weighted | 0.06244 | 0.06732 | 1.0781 | 0 of 128 | 0.09193 | 1.4723 | 0 of 128 |
| diag | up | trace-weighted | 0.06247 | 0.06735 | 1.0780 | 0 of 128 | 0.09175 | 1.4687 | 0 of 128 |
| diag | both | unweighted | 0.06241 | 0.06730 | 1.0783 | 0 of 128 | 0.09298 | 1.4898 | 0 of 128 |
| diag | both | count-weighted | 0.06246 | 0.06734 | 1.0781 | 0 of 128 | 0.09182 | 1.4701 | 0 of 128 |
| diag | both | trace-weighted | 0.06249 | 0.06736 | 1.0780 | 0 of 128 | 0.09164 | 1.4665 | 0 of 128 |
| plain | gate | unweighted | 0.26005 | 0.26790 | 1.0302 | 0 of 128 | 0.32639 | 1.2551 | 0 of 128 |
| plain | gate | count-weighted | 0.26002 | 0.26789 | 1.0302 | 0 of 128 | 0.32425 | 1.2470 | 0 of 128 |
| plain | gate | trace-weighted | 0.26000 | 0.26788 | 1.0303 | 0 of 128 | 0.32383 | 1.2455 | 0 of 128 |
| plain | up | unweighted | 0.25990 | 0.26775 | 1.0302 | 0 of 128 | 0.32700 | 1.2582 | 0 of 128 |
| plain | up | count-weighted | 0.25987 | 0.26774 | 1.0303 | 0 of 128 | 0.32484 | 1.2500 | 0 of 128 |
| plain | up | trace-weighted | 0.25984 | 0.26773 | 1.0303 | 0 of 128 | 0.32442 | 1.2485 | 0 of 128 |
| plain | both | unweighted | 0.25997 | 0.26783 | 1.0302 | 0 of 128 | 0.32670 | 1.2567 | 0 of 128 |
| plain | both | count-weighted | 0.25995 | 0.26781 | 1.0303 | 0 of 128 | 0.32454 | 1.2485 | 0 of 128 |
| plain | both | trace-weighted | 0.25992 | 0.26780 | 1.0303 | 0 of 128 | 0.32413 | 1.2470 | 0 of 128 |
| gain | gate | unweighted | 0.99668 | 0.99782 | 1.0011 | 6 of 128 | 0.98540 | 0.9887 | 127 of 128 |
| gain | gate | count-weighted | 0.99661 | 0.99774 | 1.0011 | 6 of 128 | 0.98570 | 0.9890 | 127 of 128 |
| gain | gate | trace-weighted | 0.99660 | 0.99774 | 1.0011 | 6 of 128 | 0.98573 | 0.9891 | 127 of 128 |
| gain | up | unweighted | 0.99666 | 0.99781 | 1.0012 | 5 of 128 | 0.98565 | 0.9890 | 127 of 128 |
| gain | up | count-weighted | 0.99659 | 0.99773 | 1.0011 | 5 of 128 | 0.98597 | 0.9893 | 127 of 128 |
| gain | up | trace-weighted | 0.99658 | 0.99773 | 1.0012 | 5 of 128 | 0.98601 | 0.9894 | 127 of 128 |
| gain | both | unweighted | 0.99667 | 0.99781 | 1.0011 | 5 of 128 | 0.98552 | 0.9888 | 127 of 128 |
| gain | both | count-weighted | 0.99660 | 0.99773 | 1.0011 | 5 of 128 | 0.98583 | 0.9892 | 127 of 128 |
| gain | both | trace-weighted | 0.99659 | 0.99774 | 1.0011 | 5 of 128 | 0.98587 | 0.9892 | 127 of 128 |
| energy | gate | unweighted | 1.06098 | 1.06741 | 1.0061 | 0 of 128 | 1.07744 | 1.0155 | 0 of 128 |
| energy | gate | count-weighted | 1.06084 | 1.06724 | 1.0060 | 0 of 128 | 1.07661 | 1.0149 | 0 of 128 |
| energy | gate | trace-weighted | 1.06080 | 1.06724 | 1.0061 | 0 of 128 | 1.07641 | 1.0147 | 0 of 128 |
| energy | up | unweighted | 1.06086 | 1.06731 | 1.0061 | 0 of 128 | 1.07834 | 1.0165 | 0 of 128 |
| energy | up | count-weighted | 1.06071 | 1.06714 | 1.0061 | 0 of 128 | 1.07753 | 1.0159 | 0 of 128 |
| energy | up | trace-weighted | 1.06068 | 1.06714 | 1.0061 | 0 of 128 | 1.07735 | 1.0157 | 0 of 128 |
| energy | both | unweighted | 1.06092 | 1.06736 | 1.0061 | 0 of 128 | 1.07789 | 1.0160 | 0 of 128 |
| energy | both | count-weighted | 1.06078 | 1.06719 | 1.0060 | 0 of 128 | 1.07707 | 1.0154 | 0 of 128 |
| energy | both | trace-weighted | 1.06074 | 1.06719 | 1.0061 | 0 of 128 | 1.07688 | 1.0152 | 0 of 128 |
| ogain | gate | unweighted | 0.99134 | 0.99914 | 1.0079 | 0 of 128 | 1.00781 | 1.0166 | 0 of 128 |
| ogain | gate | count-weighted | 0.99114 | 0.99914 | 1.0081 | 0 of 128 | 1.00773 | 1.0167 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99104 | 0.99913 | 1.0082 | 0 of 128 | 1.00763 | 1.0167 | 0 of 128 |
| ogain | up | unweighted | 0.99601 | 0.99879 | 1.0028 | 1 of 128 | 1.00591 | 1.0099 | 0 of 128 |
| ogain | up | count-weighted | 0.99594 | 0.99879 | 1.0029 | 1 of 128 | 1.00581 | 1.0099 | 0 of 128 |
| ogain | up | trace-weighted | 0.99592 | 0.99878 | 1.0029 | 1 of 128 | 1.00572 | 1.0098 | 0 of 128 |
| ogain | both | unweighted | 0.99341 | 0.99898 | 1.0056 | 0 of 128 | 1.00698 | 1.0137 | 0 of 128 |
| ogain | both | count-weighted | 0.99326 | 0.99898 | 1.0058 | 0 of 128 | 1.00689 | 1.0137 | 0 of 128 |
| ogain | both | trace-weighted | 0.99319 | 0.99898 | 1.0058 | 0 of 128 | 1.00680 | 1.0137 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.2575e-02 | [-1.2875e-02, -1.2247e-02] |
| up | -1.5785e-02 | [-1.6092e-02, -1.5464e-02] |
| both | -1.3968e-02 | [-1.4253e-02, -1.3659e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.9491e-02 | [-1.9939e-02, -1.9044e-02] | -6.9163e-03 | [-7.2981e-03, -6.5605e-03] |
| full | up | -2.5147e-02 | [-2.5530e-02, -2.4774e-02] | -9.3617e-03 | [-9.7834e-03, -8.9589e-03] |
| full | both | -2.1948e-02 | [-2.2347e-02, -2.1553e-02] | -7.9799e-03 | [-8.3699e-03, -7.6092e-03] |
| reg | gate | -1.6892e-02 | [-1.7259e-02, -1.6522e-02] | -5.2513e-03 | [-5.5713e-03, -4.9541e-03] |
| reg | up | -2.1520e-02 | [-2.1825e-02, -2.1224e-02] | -7.1107e-03 | [-7.4611e-03, -6.7767e-03] |
| reg | both | -1.8919e-02 | [-1.9239e-02, -1.8599e-02] | -6.0668e-03 | [-6.3958e-03, -5.7562e-03] |
| bias | gate | -4.7299e-03 | [-4.8494e-03, -4.6063e-03] | -5.5693e-04 | [-6.1220e-04, -5.0411e-04] |
| bias | up | -6.2002e-03 | [-6.3408e-03, -6.0623e-03] | -1.0323e-03 | [-1.0941e-03, -9.7255e-04] |
| bias | both | -5.3672e-03 | [-5.4816e-03, -5.2492e-03] | -7.6342e-04 | [-8.1887e-04, -7.1041e-04] |
| rest | gate | -1.4761e-02 | [-1.5132e-02, -1.4395e-02] | -6.3594e-03 | [-6.7014e-03, -6.0416e-03] |
| rest | up | -1.8947e-02 | [-1.9273e-02, -1.8632e-02] | -8.3294e-03 | [-8.7155e-03, -7.9632e-03] |
| rest | both | -1.6581e-02 | [-1.6924e-02, -1.6245e-02] | -7.2164e-03 | [-7.5766e-03, -6.8791e-03] |
| diag | gate | +2.9230e-02 | [+2.8474e-02, +3.0044e-02] | +2.4354e-02 | [+2.3598e-02, +2.5166e-02] |
| diag | up | +2.9491e-02 | [+2.8724e-02, +3.0315e-02] | +2.4613e-02 | [+2.3847e-02, +2.5432e-02] |
| diag | both | +2.9360e-02 | [+2.8598e-02, +3.0179e-02] | +2.4483e-02 | [+2.3721e-02, +2.5299e-02] |

## Layer 2 (IQ2_S; UD `.scale` present, min 3.92e-05, max 0.0001678)

- Gram layout: expert 119: max |strict lower| = 0, max |strict upper| = 1.135e+05
- Gram layout: expert 53: max |strict lower| = 0, max |strict upper| = 1.055e+05
- Gram layout: expert 7: max |strict lower| = 0, max |strict upper| = 8.248e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 2.65e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.2.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.783213 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 4125, median 11936, max 30005

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03501 | 0.02403 | 0.6864 | 128 of 128 | 0.01625 | 0.4640 | 128 of 128 |
| full | gate | count-weighted | 0.03487 | 0.02380 | 0.6825 | 128 of 128 | 0.01642 | 0.4710 | 128 of 128 |
| full | gate | trace-weighted | 0.03478 | 0.02373 | 0.6823 | 128 of 128 | 0.01641 | 0.4718 | 128 of 128 |
| full | up | unweighted | 0.04672 | 0.03203 | 0.6856 | 128 of 128 | 0.02058 | 0.4405 | 128 of 128 |
| full | up | count-weighted | 0.04649 | 0.03171 | 0.6820 | 128 of 128 | 0.02080 | 0.4474 | 128 of 128 |
| full | up | trace-weighted | 0.04647 | 0.03169 | 0.6820 | 128 of 128 | 0.02082 | 0.4480 | 128 of 128 |
| full | both | unweighted | 0.03998 | 0.02743 | 0.6860 | 128 of 128 | 0.01809 | 0.4524 | 128 of 128 |
| full | both | count-weighted | 0.03981 | 0.02716 | 0.6823 | 128 of 128 | 0.01828 | 0.4593 | 128 of 128 |
| full | both | trace-weighted | 0.03974 | 0.02711 | 0.6822 | 128 of 128 | 0.01828 | 0.4600 | 128 of 128 |
| reg | gate | unweighted | 0.03643 | 0.02629 | 0.7216 | 128 of 128 | 0.02019 | 0.5543 | 128 of 128 |
| reg | gate | count-weighted | 0.03629 | 0.02606 | 0.7179 | 128 of 128 | 0.02030 | 0.5594 | 128 of 128 |
| reg | gate | trace-weighted | 0.03620 | 0.02599 | 0.7178 | 128 of 128 | 0.02028 | 0.5600 | 128 of 128 |
| reg | up | unweighted | 0.04777 | 0.03443 | 0.7208 | 128 of 128 | 0.02548 | 0.5334 | 128 of 128 |
| reg | up | count-weighted | 0.04755 | 0.03412 | 0.7175 | 128 of 128 | 0.02561 | 0.5385 | 128 of 128 |
| reg | up | trace-weighted | 0.04753 | 0.03410 | 0.7175 | 128 of 128 | 0.02562 | 0.5390 | 128 of 128 |
| reg | both | unweighted | 0.04130 | 0.02978 | 0.7212 | 128 of 128 | 0.02246 | 0.5439 | 128 of 128 |
| reg | both | count-weighted | 0.04112 | 0.02952 | 0.7177 | 128 of 128 | 0.02258 | 0.5491 | 128 of 128 |
| reg | both | trace-weighted | 0.04106 | 0.02947 | 0.7177 | 128 of 128 | 0.02256 | 0.5496 | 128 of 128 |
| bias | gate | unweighted | 0.00482 | 0.00100 | 0.2078 | 128 of 128 | 4.748e-04 | 0.0985 | 128 of 128 |
| bias | gate | count-weighted | 0.00483 | 9.842e-04 | 0.2039 | 128 of 128 | 4.752e-04 | 0.0985 | 128 of 128 |
| bias | gate | trace-weighted | 0.00481 | 9.768e-04 | 0.2030 | 128 of 128 | 4.781e-04 | 0.0994 | 128 of 128 |
| bias | up | unweighted | 0.00642 | 0.00132 | 0.2053 | 128 of 128 | 1.543e-04 | 0.0240 | 128 of 128 |
| bias | up | count-weighted | 0.00641 | 0.00130 | 0.2022 | 128 of 128 | 1.567e-04 | 0.0245 | 128 of 128 |
| bias | up | trace-weighted | 0.00640 | 0.00129 | 0.2016 | 128 of 128 | 1.574e-04 | 0.0246 | 128 of 128 |
| bias | both | unweighted | 0.00550 | 0.00114 | 0.2066 | 128 of 128 | 3.388e-04 | 0.0616 | 128 of 128 |
| bias | both | count-weighted | 0.00550 | 0.00112 | 0.2031 | 128 of 128 | 3.400e-04 | 0.0618 | 128 of 128 |
| bias | both | trace-weighted | 0.00549 | 0.00111 | 0.2023 | 128 of 128 | 3.421e-04 | 0.0624 | 128 of 128 |
| rest | gate | unweighted | 0.03019 | 0.02303 | 0.7628 | 128 of 128 | 0.01577 | 0.5224 | 128 of 128 |
| rest | gate | count-weighted | 0.03004 | 0.02281 | 0.7594 | 128 of 128 | 0.01595 | 0.5308 | 128 of 128 |
| rest | gate | trace-weighted | 0.02997 | 0.02276 | 0.7593 | 128 of 128 | 0.01593 | 0.5316 | 128 of 128 |
| rest | up | unweighted | 0.04030 | 0.03072 | 0.7622 | 128 of 128 | 0.02043 | 0.5069 | 128 of 128 |
| rest | up | count-weighted | 0.04009 | 0.03041 | 0.7587 | 128 of 128 | 0.02064 | 0.5150 | 128 of 128 |
| rest | up | trace-weighted | 0.04007 | 0.03040 | 0.7587 | 128 of 128 | 0.02066 | 0.5156 | 128 of 128 |
| rest | both | unweighted | 0.03448 | 0.02629 | 0.7625 | 128 of 128 | 0.01775 | 0.5147 | 128 of 128 |
| rest | both | count-weighted | 0.03431 | 0.02604 | 0.7591 | 128 of 128 | 0.01794 | 0.5230 | 128 of 128 |
| rest | both | trace-weighted | 0.03425 | 0.02600 | 0.7590 | 128 of 128 | 0.01794 | 0.5237 | 128 of 128 |
| diag | gate | unweighted | 0.06185 | 0.06638 | 1.0732 | 0 of 128 | 0.09002 | 1.4556 | 0 of 128 |
| diag | gate | count-weighted | 0.06190 | 0.06642 | 1.0731 | 0 of 128 | 0.08937 | 1.4439 | 0 of 128 |
| diag | gate | trace-weighted | 0.06190 | 0.06642 | 1.0731 | 0 of 128 | 0.08931 | 1.4429 | 0 of 128 |
| diag | up | unweighted | 0.06182 | 0.06635 | 1.0732 | 0 of 128 | 0.09042 | 1.4627 | 0 of 128 |
| diag | up | count-weighted | 0.06187 | 0.06639 | 1.0731 | 0 of 128 | 0.08975 | 1.4507 | 0 of 128 |
| diag | up | trace-weighted | 0.06187 | 0.06639 | 1.0731 | 0 of 128 | 0.08969 | 1.4497 | 0 of 128 |
| diag | both | unweighted | 0.06183 | 0.06636 | 1.0732 | 0 of 128 | 0.09022 | 1.4591 | 0 of 128 |
| diag | both | count-weighted | 0.06188 | 0.06641 | 1.0731 | 0 of 128 | 0.08956 | 1.4473 | 0 of 128 |
| diag | both | trace-weighted | 0.06188 | 0.06641 | 1.0731 | 0 of 128 | 0.08950 | 1.4463 | 0 of 128 |
| plain | gate | unweighted | 0.26189 | 0.26942 | 1.0288 | 0 of 128 | 0.33021 | 1.2609 | 0 of 128 |
| plain | gate | count-weighted | 0.26191 | 0.26945 | 1.0288 | 0 of 128 | 0.32896 | 1.2560 | 0 of 128 |
| plain | gate | trace-weighted | 0.26191 | 0.26946 | 1.0288 | 0 of 128 | 0.32887 | 1.2557 | 0 of 128 |
| plain | up | unweighted | 0.26177 | 0.26930 | 1.0288 | 0 of 128 | 0.33148 | 1.2663 | 0 of 128 |
| plain | up | count-weighted | 0.26177 | 0.26931 | 1.0288 | 0 of 128 | 0.33018 | 1.2613 | 0 of 128 |
| plain | up | trace-weighted | 0.26177 | 0.26932 | 1.0288 | 0 of 128 | 0.33009 | 1.2610 | 0 of 128 |
| plain | both | unweighted | 0.26183 | 0.26936 | 1.0288 | 0 of 128 | 0.33085 | 1.2636 | 0 of 128 |
| plain | both | count-weighted | 0.26184 | 0.26938 | 1.0288 | 0 of 128 | 0.32957 | 1.2587 | 0 of 128 |
| plain | both | trace-weighted | 0.26184 | 0.26939 | 1.0288 | 0 of 128 | 0.32948 | 1.2583 | 0 of 128 |
| gain | gate | unweighted | 0.99768 | 0.99868 | 1.0010 | 4 of 128 | 0.98682 | 0.9891 | 128 of 128 |
| gain | gate | count-weighted | 0.99762 | 0.99862 | 1.0010 | 4 of 128 | 0.98702 | 0.9894 | 128 of 128 |
| gain | gate | trace-weighted | 0.99762 | 0.99861 | 1.0010 | 4 of 128 | 0.98704 | 0.9894 | 128 of 128 |
| gain | up | unweighted | 0.99768 | 0.99869 | 1.0010 | 3 of 128 | 0.98683 | 0.9891 | 128 of 128 |
| gain | up | count-weighted | 0.99762 | 0.99864 | 1.0010 | 3 of 128 | 0.98702 | 0.9894 | 128 of 128 |
| gain | up | trace-weighted | 0.99762 | 0.99863 | 1.0010 | 3 of 128 | 0.98704 | 0.9894 | 128 of 128 |
| gain | both | unweighted | 0.99768 | 0.99869 | 1.0010 | 4 of 128 | 0.98683 | 0.9891 | 128 of 128 |
| gain | both | count-weighted | 0.99762 | 0.99863 | 1.0010 | 4 of 128 | 0.98702 | 0.9894 | 128 of 128 |
| gain | both | trace-weighted | 0.99762 | 0.99862 | 1.0010 | 4 of 128 | 0.98704 | 0.9894 | 128 of 128 |
| energy | gate | unweighted | 1.06394 | 1.06995 | 1.0057 | 0 of 128 | 1.08273 | 1.0177 | 0 of 128 |
| energy | gate | count-weighted | 1.06385 | 1.06985 | 1.0056 | 0 of 128 | 1.08229 | 1.0173 | 0 of 128 |
| energy | gate | trace-weighted | 1.06384 | 1.06984 | 1.0056 | 0 of 128 | 1.08227 | 1.0173 | 0 of 128 |
| energy | up | unweighted | 1.06388 | 1.06991 | 1.0057 | 0 of 128 | 1.08358 | 1.0185 | 0 of 128 |
| energy | up | count-weighted | 1.06377 | 1.06980 | 1.0057 | 0 of 128 | 1.08310 | 1.0182 | 0 of 128 |
| energy | up | trace-weighted | 1.06377 | 1.06979 | 1.0057 | 0 of 128 | 1.08307 | 1.0181 | 0 of 128 |
| energy | both | unweighted | 1.06391 | 1.06993 | 1.0057 | 0 of 128 | 1.08316 | 1.0181 | 0 of 128 |
| energy | both | count-weighted | 1.06381 | 1.06982 | 1.0057 | 0 of 128 | 1.08270 | 1.0178 | 0 of 128 |
| energy | both | trace-weighted | 1.06380 | 1.06981 | 1.0056 | 0 of 128 | 1.08267 | 1.0177 | 0 of 128 |
| ogain | gate | unweighted | 0.99376 | 0.99950 | 1.0058 | 0 of 128 | 1.00949 | 1.0158 | 0 of 128 |
| ogain | gate | count-weighted | 0.99369 | 0.99943 | 1.0058 | 0 of 128 | 1.00942 | 1.0158 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99370 | 0.99944 | 1.0058 | 0 of 128 | 1.00945 | 1.0158 | 0 of 128 |
| ogain | up | unweighted | 0.99677 | 0.99913 | 1.0024 | 3 of 128 | 1.00592 | 1.0092 | 0 of 128 |
| ogain | up | count-weighted | 0.99670 | 0.99906 | 1.0024 | 3 of 128 | 1.00585 | 1.0092 | 0 of 128 |
| ogain | up | trace-weighted | 0.99669 | 0.99906 | 1.0024 | 3 of 128 | 1.00586 | 1.0092 | 0 of 128 |
| ogain | both | unweighted | 0.99507 | 0.99934 | 1.0043 | 0 of 128 | 1.00797 | 1.0130 | 0 of 128 |
| ogain | both | count-weighted | 0.99500 | 0.99928 | 1.0043 | 0 of 128 | 1.00789 | 1.0130 | 0 of 128 |
| ogain | both | trace-weighted | 0.99500 | 0.99927 | 1.0043 | 0 of 128 | 1.00791 | 1.0130 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.1072e-02 | [-1.1322e-02, -1.0844e-02] |
| up | -1.4783e-02 | [-1.5001e-02, -1.4580e-02] |
| both | -1.2647e-02 | [-1.2873e-02, -1.2437e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.8447e-02 | [-1.8869e-02, -1.8038e-02] | -7.3751e-03 | [-7.7233e-03, -7.0459e-03] |
| full | up | -2.5692e-02 | [-2.6059e-02, -2.5339e-02] | -1.0909e-02 | [-1.1304e-02, -1.0518e-02] |
| full | both | -2.1523e-02 | [-2.1910e-02, -2.1147e-02] | -8.8761e-03 | [-9.2409e-03, -8.5239e-03] |
| reg | gate | -1.5989e-02 | [-1.6337e-02, -1.5649e-02] | -5.7529e-03 | [-6.0464e-03, -5.4744e-03] |
| reg | up | -2.1944e-02 | [-2.2234e-02, -2.1665e-02] | -8.5117e-03 | [-8.8421e-03, -8.1823e-03] |
| reg | both | -1.8544e-02 | [-1.8857e-02, -1.8237e-02] | -6.9368e-03 | [-7.2442e-03, -6.6410e-03] |
| bias | gate | -4.3512e-03 | [-4.4769e-03, -4.2243e-03] | -5.0891e-04 | [-5.7233e-04, -4.5309e-04] |
| bias | up | -6.2511e-03 | [-6.3832e-03, -6.1206e-03] | -1.1390e-03 | [-1.2124e-03, -1.0753e-03] |
| bias | both | -5.1583e-03 | [-5.2781e-03, -5.0362e-03] | -7.7662e-04 | [-8.4169e-04, -7.1941e-04] |
| rest | gate | -1.4096e-02 | [-1.4432e-02, -1.3767e-02] | -6.8661e-03 | [-7.1706e-03, -6.5760e-03] |
| rest | up | -1.9441e-02 | [-1.9774e-02, -1.9117e-02] | -9.7700e-03 | [-1.0128e-02, -9.4150e-03] |
| rest | both | -1.6365e-02 | [-1.6695e-02, -1.6047e-02] | -8.0994e-03 | [-8.4223e-03, -7.7855e-03] |
| diag | gate | +2.7477e-02 | [+2.6940e-02, +2.8033e-02] | +2.2951e-02 | [+2.2420e-02, +2.3508e-02] |
| diag | up | +2.7884e-02 | [+2.7340e-02, +2.8449e-02] | +2.3360e-02 | [+2.2820e-02, +2.3919e-02] |
| diag | both | +2.7680e-02 | [+2.7141e-02, +2.8242e-02] | +2.3155e-02 | [+2.2620e-02, +2.3709e-02] |

## Layer 3 (IQ2_S; UD `.scale` present, min 4.028e-05, max 0.0002603)

- Gram layout: expert 62: max |strict lower| = 0, max |strict upper| = 8.781e+04
- Gram layout: expert 44: max |strict lower| = 0, max |strict upper| = 9.274e+04
- Gram layout: expert 103: max |strict lower| = 0, max |strict upper| = 7.975e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 5.61e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.3.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.775444 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 799, median 12339, max 25839

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03534 | 0.02460 | 0.6961 | 128 of 128 | 0.01717 | 0.4859 | 127 of 128 |
| full | gate | count-weighted | 0.03488 | 0.02415 | 0.6923 | 128 of 128 | 0.01697 | 0.4865 | 127 of 128 |
| full | gate | trace-weighted | 0.03474 | 0.02402 | 0.6915 | 128 of 128 | 0.01695 | 0.4880 | 127 of 128 |
| full | up | unweighted | 0.04844 | 0.03372 | 0.6962 | 128 of 128 | 0.02214 | 0.4572 | 127 of 128 |
| full | up | count-weighted | 0.04791 | 0.03315 | 0.6920 | 128 of 128 | 0.02191 | 0.4572 | 127 of 128 |
| full | up | trace-weighted | 0.04780 | 0.03304 | 0.6912 | 128 of 128 | 0.02190 | 0.4581 | 127 of 128 |
| full | both | unweighted | 0.04081 | 0.02841 | 0.6962 | 128 of 128 | 0.01925 | 0.4717 | 127 of 128 |
| full | both | count-weighted | 0.04032 | 0.02791 | 0.6922 | 128 of 128 | 0.01903 | 0.4720 | 127 of 128 |
| full | both | trace-weighted | 0.04018 | 0.02778 | 0.6914 | 128 of 128 | 0.01902 | 0.4732 | 127 of 128 |
| reg | gate | unweighted | 0.03676 | 0.02684 | 0.7300 | 128 of 128 | 0.02102 | 0.5719 | 127 of 128 |
| reg | gate | count-weighted | 0.03631 | 0.02638 | 0.7265 | 128 of 128 | 0.02077 | 0.5720 | 127 of 128 |
| reg | gate | trace-weighted | 0.03617 | 0.02625 | 0.7257 | 128 of 128 | 0.02073 | 0.5732 | 127 of 128 |
| reg | up | unweighted | 0.04941 | 0.03608 | 0.7301 | 128 of 128 | 0.02700 | 0.5464 | 127 of 128 |
| reg | up | count-weighted | 0.04892 | 0.03552 | 0.7262 | 128 of 128 | 0.02671 | 0.5460 | 127 of 128 |
| reg | up | trace-weighted | 0.04881 | 0.03541 | 0.7255 | 128 of 128 | 0.02668 | 0.5467 | 127 of 128 |
| reg | both | unweighted | 0.04211 | 0.03074 | 0.7300 | 128 of 128 | 0.02355 | 0.5593 | 127 of 128 |
| reg | both | count-weighted | 0.04164 | 0.03024 | 0.7264 | 128 of 128 | 0.02328 | 0.5591 | 127 of 128 |
| reg | both | trace-weighted | 0.04150 | 0.03012 | 0.7256 | 128 of 128 | 0.02325 | 0.5601 | 127 of 128 |
| bias | gate | unweighted | 0.00467 | 9.288e-04 | 0.1988 | 128 of 128 | 5.462e-04 | 0.1169 | 127 of 128 |
| bias | gate | count-weighted | 0.00463 | 9.062e-04 | 0.1957 | 128 of 128 | 5.135e-04 | 0.1109 | 127 of 128 |
| bias | gate | trace-weighted | 0.00462 | 8.962e-04 | 0.1942 | 128 of 128 | 5.192e-04 | 0.1125 | 127 of 128 |
| bias | up | unweighted | 0.00640 | 0.00127 | 0.1987 | 128 of 128 | 1.934e-04 | 0.0302 | 127 of 128 |
| bias | up | count-weighted | 0.00637 | 0.00124 | 0.1947 | 128 of 128 | 1.370e-04 | 0.0215 | 127 of 128 |
| bias | up | trace-weighted | 0.00635 | 0.00123 | 0.1935 | 128 of 128 | 1.365e-04 | 0.0215 | 127 of 128 |
| bias | both | unweighted | 0.00539 | 0.00107 | 0.1987 | 128 of 128 | 3.998e-04 | 0.0741 | 127 of 128 |
| bias | both | count-weighted | 0.00536 | 0.00104 | 0.1951 | 128 of 128 | 3.568e-04 | 0.0666 | 127 of 128 |
| bias | both | trace-weighted | 0.00534 | 0.00103 | 0.1938 | 128 of 128 | 3.602e-04 | 0.0675 | 127 of 128 |
| rest | gate | unweighted | 0.03067 | 0.02367 | 0.7719 | 128 of 128 | 0.01663 | 0.5421 | 127 of 128 |
| rest | gate | count-weighted | 0.03025 | 0.02324 | 0.7683 | 128 of 128 | 0.01646 | 0.5440 | 127 of 128 |
| rest | gate | trace-weighted | 0.03012 | 0.02312 | 0.7677 | 128 of 128 | 0.01643 | 0.5456 | 127 of 128 |
| rest | up | unweighted | 0.04204 | 0.03245 | 0.7719 | 128 of 128 | 0.02195 | 0.5221 | 127 of 128 |
| rest | up | count-weighted | 0.04155 | 0.03192 | 0.7682 | 128 of 128 | 0.02177 | 0.5240 | 127 of 128 |
| rest | up | trace-weighted | 0.04145 | 0.03181 | 0.7675 | 128 of 128 | 0.02176 | 0.5250 | 127 of 128 |
| rest | both | unweighted | 0.03542 | 0.02734 | 0.7719 | 128 of 128 | 0.01885 | 0.5323 | 127 of 128 |
| rest | both | count-weighted | 0.03497 | 0.02686 | 0.7683 | 128 of 128 | 0.01868 | 0.5341 | 127 of 128 |
| rest | both | trace-weighted | 0.03484 | 0.02675 | 0.7676 | 128 of 128 | 0.01866 | 0.5354 | 127 of 128 |
| diag | gate | unweighted | 0.06224 | 0.06655 | 1.0694 | 0 of 128 | 0.08920 | 1.4332 | 0 of 128 |
| diag | gate | count-weighted | 0.06224 | 0.06655 | 1.0694 | 0 of 128 | 0.08884 | 1.4275 | 0 of 128 |
| diag | gate | trace-weighted | 0.06225 | 0.06657 | 1.0694 | 0 of 128 | 0.08875 | 1.4257 | 0 of 128 |
| diag | up | unweighted | 0.06220 | 0.06652 | 1.0694 | 0 of 128 | 0.08959 | 1.4404 | 0 of 128 |
| diag | up | count-weighted | 0.06220 | 0.06652 | 1.0694 | 0 of 128 | 0.08923 | 1.4346 | 0 of 128 |
| diag | up | trace-weighted | 0.06221 | 0.06653 | 1.0694 | 0 of 128 | 0.08914 | 1.4328 | 0 of 128 |
| diag | both | unweighted | 0.06222 | 0.06654 | 1.0694 | 0 of 128 | 0.08940 | 1.4368 | 0 of 128 |
| diag | both | count-weighted | 0.06222 | 0.06654 | 1.0694 | 0 of 128 | 0.08904 | 1.4310 | 0 of 128 |
| diag | both | trace-weighted | 0.06223 | 0.06655 | 1.0694 | 0 of 128 | 0.08894 | 1.4293 | 0 of 128 |
| plain | gate | unweighted | 0.26148 | 0.26857 | 1.0271 | 0 of 128 | 0.32496 | 1.2428 | 0 of 128 |
| plain | gate | count-weighted | 0.26154 | 0.26863 | 1.0271 | 0 of 128 | 0.32458 | 1.2410 | 0 of 128 |
| plain | gate | trace-weighted | 0.26158 | 0.26867 | 1.0271 | 0 of 128 | 0.32450 | 1.2405 | 0 of 128 |
| plain | up | unweighted | 0.26135 | 0.26845 | 1.0272 | 0 of 128 | 0.32633 | 1.2486 | 0 of 128 |
| plain | up | count-weighted | 0.26141 | 0.26850 | 1.0271 | 0 of 128 | 0.32596 | 1.2469 | 0 of 128 |
| plain | up | trace-weighted | 0.26144 | 0.26854 | 1.0271 | 0 of 128 | 0.32589 | 1.2465 | 0 of 128 |
| plain | both | unweighted | 0.26141 | 0.26851 | 1.0271 | 0 of 128 | 0.32565 | 1.2457 | 0 of 128 |
| plain | both | count-weighted | 0.26148 | 0.26857 | 1.0271 | 0 of 128 | 0.32527 | 1.2440 | 0 of 128 |
| plain | both | trace-weighted | 0.26151 | 0.26860 | 1.0271 | 0 of 128 | 0.32520 | 1.2435 | 0 of 128 |
| gain | gate | unweighted | 0.99750 | 0.99843 | 1.0009 | 3 of 128 | 0.98729 | 0.9898 | 127 of 128 |
| gain | gate | count-weighted | 0.99749 | 0.99841 | 1.0009 | 3 of 128 | 0.98741 | 0.9899 | 127 of 128 |
| gain | gate | trace-weighted | 0.99748 | 0.99840 | 1.0009 | 3 of 128 | 0.98745 | 0.9899 | 127 of 128 |
| gain | up | unweighted | 0.99749 | 0.99846 | 1.0010 | 3 of 128 | 0.98713 | 0.9896 | 127 of 128 |
| gain | up | count-weighted | 0.99748 | 0.99844 | 1.0010 | 3 of 128 | 0.98722 | 0.9897 | 127 of 128 |
| gain | up | trace-weighted | 0.99747 | 0.99843 | 1.0010 | 3 of 128 | 0.98724 | 0.9897 | 127 of 128 |
| gain | both | unweighted | 0.99750 | 0.99844 | 1.0009 | 3 of 128 | 0.98721 | 0.9897 | 127 of 128 |
| gain | both | count-weighted | 0.99748 | 0.99842 | 1.0009 | 3 of 128 | 0.98731 | 0.9898 | 127 of 128 |
| gain | both | trace-weighted | 0.99748 | 0.99841 | 1.0009 | 3 of 128 | 0.98734 | 0.9898 | 127 of 128 |
| energy | gate | unweighted | 1.06338 | 1.06900 | 1.0053 | 0 of 128 | 1.08025 | 1.0159 | 0 of 128 |
| energy | gate | count-weighted | 1.06338 | 1.06898 | 1.0053 | 0 of 128 | 1.08020 | 1.0158 | 0 of 128 |
| energy | gate | trace-weighted | 1.06339 | 1.06898 | 1.0053 | 0 of 128 | 1.08022 | 1.0158 | 0 of 128 |
| energy | up | unweighted | 1.06329 | 1.06898 | 1.0053 | 0 of 128 | 1.08081 | 1.0165 | 0 of 128 |
| energy | up | count-weighted | 1.06329 | 1.06897 | 1.0053 | 0 of 128 | 1.08072 | 1.0164 | 0 of 128 |
| energy | up | trace-weighted | 1.06329 | 1.06897 | 1.0053 | 0 of 128 | 1.08071 | 1.0164 | 0 of 128 |
| energy | both | unweighted | 1.06333 | 1.06899 | 1.0053 | 0 of 128 | 1.08053 | 1.0162 | 0 of 128 |
| energy | both | count-weighted | 1.06334 | 1.06898 | 1.0053 | 0 of 128 | 1.08046 | 1.0161 | 0 of 128 |
| energy | both | trace-weighted | 1.06334 | 1.06898 | 1.0053 | 0 of 128 | 1.08047 | 1.0161 | 0 of 128 |
| ogain | gate | unweighted | 0.99408 | 0.99934 | 1.0053 | 9 of 128 | 1.00955 | 1.0156 | 0 of 128 |
| ogain | gate | count-weighted | 0.99427 | 0.99932 | 1.0051 | 9 of 128 | 1.00973 | 1.0156 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99439 | 0.99931 | 1.0050 | 9 of 128 | 1.00982 | 1.0155 | 0 of 128 |
| ogain | up | unweighted | 0.99645 | 0.99896 | 1.0025 | 1 of 128 | 1.00501 | 1.0086 | 0 of 128 |
| ogain | up | count-weighted | 0.99641 | 0.99897 | 1.0026 | 1 of 128 | 1.00506 | 1.0087 | 0 of 128 |
| ogain | up | trace-weighted | 0.99638 | 0.99897 | 1.0026 | 1 of 128 | 1.00504 | 1.0087 | 0 of 128 |
| ogain | both | unweighted | 0.99510 | 0.99918 | 1.0041 | 3 of 128 | 1.00764 | 1.0126 | 0 of 128 |
| ogain | both | count-weighted | 0.99520 | 0.99917 | 1.0040 | 3 of 128 | 1.00777 | 1.0126 | 0 of 128 |
| ogain | both | trace-weighted | 0.99525 | 0.99916 | 1.0039 | 3 of 128 | 1.00781 | 1.0126 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.0733e-02 | [-1.0906e-02, -1.0566e-02] |
| up | -1.4758e-02 | [-1.4949e-02, -1.4564e-02] |
| both | -1.2413e-02 | [-1.2580e-02, -1.2250e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.7911e-02 | [-1.8327e-02, -1.7509e-02] | -7.1780e-03 | [-7.5453e-03, -6.8280e-03] |
| full | up | -2.6007e-02 | [-2.6407e-02, -2.5620e-02] | -1.1249e-02 | [-1.1666e-02, -1.0851e-02] |
| full | both | -2.1290e-02 | [-2.1679e-02, -2.0915e-02] | -8.8770e-03 | [-9.2616e-03, -8.5095e-03] |
| reg | gate | -1.5541e-02 | [-1.5886e-02, -1.5210e-02] | -5.6092e-03 | [-5.9224e-03, -5.3091e-03] |
| reg | up | -2.2209e-02 | [-2.2523e-02, -2.1903e-02] | -8.8157e-03 | [-9.1646e-03, -8.4858e-03] |
| reg | both | -1.8357e-02 | [-1.8671e-02, -1.8053e-02] | -6.9631e-03 | [-7.2874e-03, -6.6530e-03] |
| bias | gate | -4.1176e-03 | [-4.2144e-03, -4.0241e-03] | -3.9269e-04 | [-4.4311e-04, -3.4390e-04] |
| bias | up | -6.2284e-03 | [-6.3365e-03, -6.1210e-03] | -1.1021e-03 | [-1.1552e-03, -1.0516e-03] |
| bias | both | -4.9982e-03 | [-5.0895e-03, -4.9112e-03] | -6.8805e-04 | [-7.3604e-04, -6.4236e-04] |
| rest | gate | -1.3793e-02 | [-1.4139e-02, -1.3467e-02] | -6.7853e-03 | [-7.1132e-03, -6.4755e-03] |
| rest | up | -1.9779e-02 | [-2.0140e-02, -1.9429e-02] | -1.0147e-02 | [-1.0535e-02, -9.7784e-03] |
| rest | both | -1.6291e-02 | [-1.6637e-02, -1.5964e-02] | -8.1890e-03 | [-8.5380e-03, -7.8573e-03] |
| diag | gate | +2.6603e-02 | [+2.6124e-02, +2.7106e-02] | +2.2286e-02 | [+2.1808e-02, +2.2785e-02] |
| diag | up | +2.7032e-02 | [+2.6556e-02, +2.7532e-02] | +2.2712e-02 | [+2.2236e-02, +2.3211e-02] |
| diag | both | +2.6817e-02 | [+2.6340e-02, +2.7319e-02] | +2.2499e-02 | [+2.2023e-02, +2.2998e-02] |

## Layer 5 (IQ2_S; UD `.scale` present, min 4.09e-05, max 0.0001975)

- Gram layout: expert 32: max |strict lower| = 0, max |strict upper| = 3e+05
- Gram layout: expert 20: max |strict lower| = 0, max |strict upper| = 1.825e+05
- Gram layout: expert 28: max |strict lower| = 0, max |strict upper| = 1.833e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 4.12e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.5.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.769924 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 1253, median 12296, max 35709

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03452 | 0.02240 | 0.6489 | 128 of 128 | 0.01584 | 0.4590 | 127 of 128 |
| full | gate | count-weighted | 0.03492 | 0.02258 | 0.6465 | 128 of 128 | 0.01608 | 0.4603 | 127 of 128 |
| full | gate | trace-weighted | 0.03509 | 0.02270 | 0.6468 | 128 of 128 | 0.01624 | 0.4628 | 127 of 128 |
| full | up | unweighted | 0.04535 | 0.02957 | 0.6520 | 128 of 128 | 0.02020 | 0.4454 | 127 of 128 |
| full | up | count-weighted | 0.04526 | 0.02939 | 0.6494 | 128 of 128 | 0.02025 | 0.4473 | 127 of 128 |
| full | up | trace-weighted | 0.04511 | 0.02930 | 0.6496 | 128 of 128 | 0.02027 | 0.4492 | 127 of 128 |
| full | both | unweighted | 0.03873 | 0.02518 | 0.6500 | 128 of 128 | 0.01751 | 0.4522 | 127 of 128 |
| full | both | count-weighted | 0.03896 | 0.02523 | 0.6475 | 128 of 128 | 0.01768 | 0.4538 | 127 of 128 |
| full | both | trace-weighted | 0.03901 | 0.02527 | 0.6477 | 128 of 128 | 0.01779 | 0.4560 | 127 of 128 |
| reg | gate | unweighted | 0.03596 | 0.02472 | 0.6875 | 128 of 128 | 0.01973 | 0.5487 | 127 of 128 |
| reg | gate | count-weighted | 0.03637 | 0.02492 | 0.6853 | 128 of 128 | 0.01998 | 0.5493 | 127 of 128 |
| reg | gate | trace-weighted | 0.03653 | 0.02505 | 0.6856 | 128 of 128 | 0.02014 | 0.5512 | 127 of 128 |
| reg | up | unweighted | 0.04655 | 0.03213 | 0.6902 | 128 of 128 | 0.02497 | 0.5365 | 127 of 128 |
| reg | up | count-weighted | 0.04647 | 0.03196 | 0.6879 | 128 of 128 | 0.02498 | 0.5376 | 127 of 128 |
| reg | up | trace-weighted | 0.04632 | 0.03187 | 0.6880 | 128 of 128 | 0.02497 | 0.5391 | 127 of 128 |
| reg | both | unweighted | 0.04013 | 0.02763 | 0.6885 | 128 of 128 | 0.02178 | 0.5426 | 127 of 128 |
| reg | both | count-weighted | 0.04036 | 0.02770 | 0.6862 | 128 of 128 | 0.02193 | 0.5434 | 127 of 128 |
| reg | both | trace-weighted | 0.04041 | 0.02774 | 0.6864 | 128 of 128 | 0.02203 | 0.5451 | 127 of 128 |
| bias | gate | unweighted | 0.00549 | 9.959e-04 | 0.1815 | 128 of 128 | 5.460e-04 | 0.0995 | 127 of 128 |
| bias | gate | count-weighted | 0.00553 | 9.842e-04 | 0.1778 | 128 of 128 | 4.917e-04 | 0.0888 | 127 of 128 |
| bias | gate | trace-weighted | 0.00553 | 9.751e-04 | 0.1762 | 128 of 128 | 4.921e-04 | 0.0889 | 127 of 128 |
| bias | up | unweighted | 0.00708 | 0.00128 | 0.1815 | 128 of 128 | 2.977e-04 | 0.0421 | 127 of 128 |
| bias | up | count-weighted | 0.00705 | 0.00125 | 0.1780 | 128 of 128 | 2.398e-04 | 0.0340 | 127 of 128 |
| bias | up | trace-weighted | 0.00701 | 0.00123 | 0.1763 | 128 of 128 | 2.395e-04 | 0.0342 | 127 of 128 |
| bias | both | unweighted | 0.00611 | 0.00111 | 0.1817 | 128 of 128 | 4.452e-04 | 0.0728 | 127 of 128 |
| bias | both | count-weighted | 0.00613 | 0.00109 | 0.1780 | 128 of 128 | 3.877e-04 | 0.0632 | 127 of 128 |
| bias | both | trace-weighted | 0.00612 | 0.00108 | 0.1763 | 128 of 128 | 3.872e-04 | 0.0633 | 127 of 128 |
| rest | gate | unweighted | 0.02903 | 0.02140 | 0.7373 | 128 of 128 | 0.01530 | 0.5269 | 127 of 128 |
| rest | gate | count-weighted | 0.02939 | 0.02159 | 0.7348 | 128 of 128 | 0.01558 | 0.5303 | 127 of 128 |
| rest | gate | trace-weighted | 0.02955 | 0.02172 | 0.7350 | 128 of 128 | 0.01575 | 0.5328 | 127 of 128 |
| rest | up | unweighted | 0.03827 | 0.02828 | 0.7390 | 128 of 128 | 0.01990 | 0.5199 | 127 of 128 |
| rest | up | count-weighted | 0.03821 | 0.02814 | 0.7364 | 128 of 128 | 0.02001 | 0.5236 | 127 of 128 |
| rest | up | trace-weighted | 0.03811 | 0.02807 | 0.7366 | 128 of 128 | 0.02003 | 0.5255 | 127 of 128 |
| rest | both | unweighted | 0.03262 | 0.02406 | 0.7378 | 128 of 128 | 0.01707 | 0.5233 | 127 of 128 |
| rest | both | count-weighted | 0.03283 | 0.02413 | 0.7352 | 128 of 128 | 0.01729 | 0.5267 | 127 of 128 |
| rest | both | trace-weighted | 0.03289 | 0.02419 | 0.7354 | 128 of 128 | 0.01740 | 0.5290 | 127 of 128 |
| diag | gate | unweighted | 0.06385 | 0.06854 | 1.0734 | 0 of 128 | 0.09240 | 1.4471 | 0 of 128 |
| diag | gate | count-weighted | 0.06392 | 0.06860 | 1.0732 | 0 of 128 | 0.09192 | 1.4380 | 0 of 128 |
| diag | gate | trace-weighted | 0.06394 | 0.06862 | 1.0732 | 0 of 128 | 0.09176 | 1.4350 | 0 of 128 |
| diag | up | unweighted | 0.06385 | 0.06853 | 1.0732 | 0 of 128 | 0.09281 | 1.4536 | 0 of 128 |
| diag | up | count-weighted | 0.06391 | 0.06859 | 1.0731 | 0 of 128 | 0.09230 | 1.4441 | 0 of 128 |
| diag | up | trace-weighted | 0.06393 | 0.06860 | 1.0731 | 0 of 128 | 0.09213 | 1.4410 | 0 of 128 |
| diag | both | unweighted | 0.06385 | 0.06853 | 1.0733 | 0 of 128 | 0.09261 | 1.4503 | 0 of 128 |
| diag | both | count-weighted | 0.06392 | 0.06859 | 1.0732 | 0 of 128 | 0.09211 | 1.4411 | 0 of 128 |
| diag | both | trace-weighted | 0.06394 | 0.06861 | 1.0731 | 0 of 128 | 0.09194 | 1.4380 | 0 of 128 |
| plain | gate | unweighted | 0.25844 | 0.26665 | 1.0318 | 0 of 128 | 0.31623 | 1.2236 | 0 of 128 |
| plain | gate | count-weighted | 0.25844 | 0.26664 | 1.0317 | 0 of 128 | 0.31528 | 1.2199 | 0 of 128 |
| plain | gate | trace-weighted | 0.25847 | 0.26668 | 1.0318 | 0 of 128 | 0.31504 | 1.2189 | 0 of 128 |
| plain | up | unweighted | 0.25833 | 0.26652 | 1.0317 | 0 of 128 | 0.31691 | 1.2267 | 0 of 128 |
| plain | up | count-weighted | 0.25833 | 0.26651 | 1.0317 | 0 of 128 | 0.31592 | 1.2230 | 0 of 128 |
| plain | up | trace-weighted | 0.25835 | 0.26655 | 1.0317 | 0 of 128 | 0.31567 | 1.2219 | 0 of 128 |
| plain | both | unweighted | 0.25839 | 0.26658 | 1.0317 | 0 of 128 | 0.31657 | 1.2252 | 0 of 128 |
| plain | both | count-weighted | 0.25838 | 0.26658 | 1.0317 | 0 of 128 | 0.31560 | 1.2214 | 0 of 128 |
| plain | both | trace-weighted | 0.25841 | 0.26661 | 1.0318 | 0 of 128 | 0.31535 | 1.2204 | 0 of 128 |
| gain | gate | unweighted | 0.99566 | 0.99687 | 1.0012 | 2 of 128 | 0.98547 | 0.9898 | 127 of 128 |
| gain | gate | count-weighted | 0.99561 | 0.99683 | 1.0012 | 2 of 128 | 0.98562 | 0.9900 | 127 of 128 |
| gain | gate | trace-weighted | 0.99561 | 0.99682 | 1.0012 | 2 of 128 | 0.98570 | 0.9900 | 127 of 128 |
| gain | up | unweighted | 0.99566 | 0.99686 | 1.0012 | 2 of 128 | 0.98581 | 0.9901 | 127 of 128 |
| gain | up | count-weighted | 0.99562 | 0.99681 | 1.0012 | 2 of 128 | 0.98595 | 0.9903 | 127 of 128 |
| gain | up | trace-weighted | 0.99561 | 0.99680 | 1.0012 | 2 of 128 | 0.98599 | 0.9903 | 127 of 128 |
| gain | both | unweighted | 0.99566 | 0.99687 | 1.0012 | 2 of 128 | 0.98564 | 0.9899 | 127 of 128 |
| gain | both | count-weighted | 0.99561 | 0.99682 | 1.0012 | 2 of 128 | 0.98579 | 0.9901 | 127 of 128 |
| gain | both | trace-weighted | 0.99561 | 0.99681 | 1.0012 | 2 of 128 | 0.98585 | 0.9902 | 127 of 128 |
| energy | gate | unweighted | 1.05811 | 1.06485 | 1.0064 | 0 of 128 | 1.07103 | 1.0122 | 0 of 128 |
| energy | gate | count-weighted | 1.05802 | 1.06475 | 1.0064 | 0 of 128 | 1.07070 | 1.0120 | 0 of 128 |
| energy | gate | trace-weighted | 1.05803 | 1.06476 | 1.0064 | 0 of 128 | 1.07069 | 1.0120 | 0 of 128 |
| energy | up | unweighted | 1.05805 | 1.06475 | 1.0063 | 0 of 128 | 1.07212 | 1.0133 | 0 of 128 |
| energy | up | count-weighted | 1.05796 | 1.06465 | 1.0063 | 0 of 128 | 1.07174 | 1.0130 | 0 of 128 |
| energy | up | trace-weighted | 1.05797 | 1.06465 | 1.0063 | 0 of 128 | 1.07167 | 1.0130 | 0 of 128 |
| energy | both | unweighted | 1.05808 | 1.06480 | 1.0064 | 0 of 128 | 1.07157 | 1.0128 | 0 of 128 |
| energy | both | count-weighted | 1.05799 | 1.06470 | 1.0063 | 0 of 128 | 1.07122 | 1.0125 | 0 of 128 |
| energy | both | trace-weighted | 1.05800 | 1.06471 | 1.0063 | 0 of 128 | 1.07118 | 1.0125 | 0 of 128 |
| ogain | gate | unweighted | 0.99081 | 0.99856 | 1.0078 | 0 of 128 | 1.00863 | 1.0180 | 0 of 128 |
| ogain | gate | count-weighted | 0.99094 | 0.99851 | 1.0076 | 0 of 128 | 1.00851 | 1.0177 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99118 | 0.99850 | 1.0074 | 0 of 128 | 1.00851 | 1.0175 | 0 of 128 |
| ogain | up | unweighted | 0.99474 | 0.99825 | 1.0035 | 1 of 128 | 1.00613 | 1.0115 | 0 of 128 |
| ogain | up | count-weighted | 0.99464 | 0.99819 | 1.0036 | 1 of 128 | 1.00602 | 1.0114 | 0 of 128 |
| ogain | up | trace-weighted | 0.99462 | 0.99820 | 1.0036 | 1 of 128 | 1.00600 | 1.0114 | 0 of 128 |
| ogain | both | unweighted | 0.99242 | 0.99844 | 1.0061 | 0 of 128 | 1.00762 | 1.0153 | 0 of 128 |
| ogain | both | count-weighted | 0.99245 | 0.99839 | 1.0060 | 0 of 128 | 1.00749 | 1.0152 | 0 of 128 |
| ogain | both | trace-weighted | 0.99258 | 0.99839 | 1.0059 | 0 of 128 | 1.00747 | 1.0150 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.2346e-02 | [-1.2790e-02, -1.1877e-02] |
| up | -1.5867e-02 | [-1.6172e-02, -1.5558e-02] |
| both | -1.3733e-02 | [-1.4100e-02, -1.3342e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.8848e-02 | [-1.9521e-02, -1.8134e-02] | -6.5016e-03 | [-6.8532e-03, -6.1574e-03] |
| full | up | -2.5014e-02 | [-2.5432e-02, -2.4597e-02] | -9.1473e-03 | [-9.4774e-03, -8.8320e-03] |
| full | both | -2.1281e-02 | [-2.1796e-02, -2.0734e-02] | -7.5476e-03 | [-7.8705e-03, -7.2401e-03] |
| reg | gate | -1.6394e-02 | [-1.6958e-02, -1.5795e-02] | -4.9471e-03 | [-5.2315e-03, -4.6738e-03] |
| reg | up | -2.1485e-02 | [-2.1821e-02, -2.1149e-02] | -6.9823e-03 | [-7.2546e-03, -6.7204e-03] |
| reg | both | -1.8429e-02 | [-1.8855e-02, -1.7973e-02] | -5.7626e-03 | [-6.0258e-03, -5.5132e-03] |
| bias | gate | -5.0432e-03 | [-5.2651e-03, -4.8116e-03] | -4.9251e-04 | [-5.6073e-04, -4.2192e-04] |
| bias | up | -6.8081e-03 | [-6.9333e-03, -6.6814e-03] | -1.0149e-03 | [-1.0699e-03, -9.6091e-04] |
| bias | both | -5.7454e-03 | [-5.9209e-03, -5.5593e-03] | -7.0414e-04 | [-7.6526e-04, -6.4263e-04] |
| rest | gate | -1.3804e-02 | [-1.4269e-02, -1.3314e-02] | -6.0091e-03 | [-6.3043e-03, -5.7265e-03] |
| rest | up | -1.8206e-02 | [-1.8545e-02, -1.7869e-02] | -8.1323e-03 | [-8.4289e-03, -7.8508e-03] |
| rest | both | -1.5535e-02 | [-1.5901e-02, -1.5151e-02] | -6.8435e-03 | [-7.1156e-03, -6.5865e-03] |
| diag | gate | +2.7998e-02 | [+2.7364e-02, +2.8646e-02] | +2.3318e-02 | [+2.2694e-02, +2.3960e-02] |
| diag | up | +2.8384e-02 | [+2.7771e-02, +2.9015e-02] | +2.3711e-02 | [+2.3106e-02, +2.4336e-02] |
| diag | both | +2.8191e-02 | [+2.7572e-02, +2.8828e-02] | +2.3515e-02 | [+2.2898e-02, +2.4148e-02] |

## Reading (amendment 7 rule, applied literally)

Rule: supported if UD is lower than ours on the full error and higher or equal on the diagonal one in both layers; if UD is also worse on the full error, layer-local output error does not explain UD's edge.

| mean | half | layer 1 full UD < ours | layer 1 diag UD >= ours | layer 2 full UD < ours | layer 2 diag UD >= ours | layer 3 full UD < ours | layer 3 diag UD >= ours | layer 5 full UD < ours | layer 5 diag UD >= ours | branch |
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

In layer 1 (both halves, count-weighted), UD's full error is 0.02527 against our 0.03924 (ratio 0.644; UD - ours -1.397e-02, 95% CI [-1.425e-02, -1.366e-02]) and its diagonal error 0.06734 against our 0.06246 (ratio 1.078); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 1 (added after registration, not part of the rule): the input-mean part is 0.00566 ours against 0.00106 UD (ratio 0.187), the centred rest 0.03357 against 0.02421 (ratio 0.721). In layer 2 (both halves, count-weighted), UD's full error is 0.02716 against our 0.03981 (ratio 0.682; UD - ours -1.265e-02, 95% CI [-1.287e-02, -1.244e-02]) and its diagonal error 0.06641 against our 0.06188 (ratio 1.073); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 2 (added after registration, not part of the rule): the input-mean part is 0.00550 ours against 0.00112 UD (ratio 0.203), the centred rest 0.03431 against 0.02604 (ratio 0.759). In layer 3 (both halves, count-weighted), UD's full error is 0.02791 against our 0.04032 (ratio 0.692; UD - ours -1.241e-02, 95% CI [-1.258e-02, -1.225e-02]) and its diagonal error 0.06654 against our 0.06222 (ratio 1.069); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 3 (added after registration, not part of the rule): the input-mean part is 0.00536 ours against 0.00104 UD (ratio 0.195), the centred rest 0.03497 against 0.02686 (ratio 0.768). In layer 5 (both halves, count-weighted), UD's full error is 0.02523 against our 0.03896 (ratio 0.647; UD - ours -1.373e-02, 95% CI [-1.410e-02, -1.334e-02]) and its diagonal error 0.06859 against our 0.06392 (ratio 1.073); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 5 (added after registration, not part of the rule): the input-mean part is 0.00613 ours against 0.00109 UD (ratio 0.178), the centred rest 0.03283 against 0.02413 (ratio 0.735). Under the registered rule this is "supported" for both halves together, count-weighted; across the three means and three halves the branches are: supported.

## refined against ours and UD

In layer 1 (both halves, count-weighted), refined's full error is 0.01729 against our 0.03924 and UD's 0.02527 (refined - ours -2.195e-02, 95% CI [-2.235e-02, -2.155e-02]; (ours - refined) / (ours - UD) = 1.571; lower than ours in 127 of 128 experts). Bias part 2.971e-04 (ours 0.00566, UD 0.00106), rest 0.01699 (ours 0.03357, UD 0.02421). Regularised (the search objective) 0.02167 (ours 0.04059, UD 0.02774). Diagonal 0.09182 (ours 0.06246, UD 0.06734), plain 0.32454 (ours 0.25995, UD 0.26781). Gains (ours / UD / refined): plain 0.9966 / 0.9977 / 0.9858, energy 1.0608 / 1.0672 / 1.0771, output 0.9933 / 0.9990 / 1.0069. In layer 2 (both halves, count-weighted), refined's full error is 0.01828 against our 0.03981 and UD's 0.02716 (refined - ours -2.152e-02, 95% CI [-2.191e-02, -2.115e-02]; (ours - refined) / (ours - UD) = 1.702; lower than ours in 128 of 128 experts). Bias part 3.400e-04 (ours 0.00550, UD 0.00112), rest 0.01794 (ours 0.03431, UD 0.02604). Regularised (the search objective) 0.02258 (ours 0.04112, UD 0.02952). Diagonal 0.08956 (ours 0.06188, UD 0.06641), plain 0.32957 (ours 0.26184, UD 0.26938). Gains (ours / UD / refined): plain 0.9976 / 0.9986 / 0.9870, energy 1.0638 / 1.0698 / 1.0827, output 0.9950 / 0.9993 / 1.0079. In layer 3 (both halves, count-weighted), refined's full error is 0.01903 against our 0.04032 and UD's 0.02791 (refined - ours -2.129e-02, 95% CI [-2.168e-02, -2.091e-02]; (ours - refined) / (ours - UD) = 1.715; lower than ours in 127 of 128 experts). Bias part 3.568e-04 (ours 0.00536, UD 0.00104), rest 0.01868 (ours 0.03497, UD 0.02686). Regularised (the search objective) 0.02328 (ours 0.04164, UD 0.03024). Diagonal 0.08904 (ours 0.06222, UD 0.06654), plain 0.32527 (ours 0.26148, UD 0.26857). Gains (ours / UD / refined): plain 0.9975 / 0.9984 / 0.9873, energy 1.0633 / 1.0690 / 1.0805, output 0.9952 / 0.9992 / 1.0078. In layer 5 (both halves, count-weighted), refined's full error is 0.01768 against our 0.03896 and UD's 0.02523 (refined - ours -2.128e-02, 95% CI [-2.180e-02, -2.073e-02]; (ours - refined) / (ours - UD) = 1.550; lower than ours in 127 of 128 experts). Bias part 3.877e-04 (ours 0.00613, UD 0.00109), rest 0.01729 (ours 0.03283, UD 0.02413). Regularised (the search objective) 0.02193 (ours 0.04036, UD 0.02770). Diagonal 0.09211 (ours 0.06392, UD 0.06859), plain 0.31560 (ours 0.25838, UD 0.26658). Gains (ours / UD / refined): plain 0.9956 / 0.9968 / 0.9858, energy 1.0580 / 1.0647 / 1.0712, output 0.9925 / 0.9984 / 1.0075.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined (count-w.) | refined / ours | refined / UD |
|---|---|---|---|---|---|---|---|---|
| 1 | < 2304 | 1 | 0.001 | 0.04645 | 0.03076 | 0.04645 | 1.000 | 1.510 |
| 1 | >= 2304 | 127 | 0.999 | 0.03923 | 0.02526 | 0.01727 | 0.440 | 0.684 |
| 2 | >= 2304 | 128 | 1.000 | 0.03981 | 0.02716 | 0.01828 | 0.459 | 0.673 |
| 3 | < 2304 | 1 | 0.000 | 0.04886 | 0.03140 | 0.04886 | 1.000 | 1.556 |
| 3 | >= 2304 | 127 | 1.000 | 0.04032 | 0.02791 | 0.01902 | 0.472 | 0.681 |
| 5 | < 2304 | 1 | 0.001 | 0.05530 | 0.03563 | 0.05530 | 1.000 | 1.552 |
| 5 | >= 2304 | 127 | 0.999 | 0.03895 | 0.02522 | 0.01765 | 0.453 | 0.700 |

Refinement log `models/refined/gu-refined-mixc-1-2-3-5.gguf.json` (scripts/refine-codes.py; wall time 8717 s for 512 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 127 | 1 | 4: 127 | 61088099 | 2632885 of 2633472 | 8997 | 1.25e-16 | 6.25e-17 | yes |
| 2 | 128 | 0 | 4: 128 | 62055188 | 2653555 of 2654208 | 8179 | 1.18e-16 | 5.55e-17 | yes |
| 3 | 127 | 1 | 4: 127 | 61258947 | 2632785 of 2633472 | 11371 | 1.18e-16 | 5.90e-17 | yes |
| 5 | 127 | 1 | 4: 127 | 58888520 | 2632609 of 2633472 | 6132 | 9.71e-17 | 7.98e-17 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 1: max abs(log - here) on reg 1.18e-16 before, 5.90e-17 after; layer 2: max abs(log - here) on reg 1.39e-16 before, 7.63e-17 after; layer 3: max abs(log - here) on reg 1.25e-16 before, 6.59e-17 after; layer 5: max abs(log - here) on reg 1.46e-16 before, 5.90e-17 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304').
