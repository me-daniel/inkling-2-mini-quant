# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined = `models/refined/gu-refined-1-2-3-5.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/v2r-offline-1-2-3-5.tsv`.
Grams: `models/grams/gu-1-2-3-5` (capture settings in its log, not restated here), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

## Layer 1 (IQ2_S; UD `.scale` present, min 3.958e-05, max 0.0001538)

- Gram layout: expert 19: max |strict lower| = 0, max |strict upper| = 4.279e+04
- Gram layout: expert 79: max |strict lower| = 0, max |strict upper| = 9.28e+04
- Gram layout: expert 87: max |strict lower| = 0, max |strict upper| = 2.077e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 1.58e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.1.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.547873 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 53, median 2330, max 19552

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03668 | 0.02495 | 0.6802 | 128 of 128 | 0.02544 | 0.6936 | 68 of 128 |
| full | gate | count-weighted | 0.03476 | 0.02317 | 0.6666 | 128 of 128 | 0.01703 | 0.4898 | 68 of 128 |
| full | gate | trace-weighted | 0.03493 | 0.02335 | 0.6686 | 128 of 128 | 0.01732 | 0.4958 | 68 of 128 |
| full | up | unweighted | 0.04626 | 0.03146 | 0.6800 | 128 of 128 | 0.03100 | 0.6700 | 68 of 128 |
| full | up | count-weighted | 0.04527 | 0.03022 | 0.6675 | 128 of 128 | 0.02120 | 0.4683 | 68 of 128 |
| full | up | trace-weighted | 0.04532 | 0.03035 | 0.6695 | 128 of 128 | 0.02151 | 0.4746 | 68 of 128 |
| full | both | unweighted | 0.04081 | 0.02776 | 0.6802 | 128 of 128 | 0.02788 | 0.6831 | 68 of 128 |
| full | both | count-weighted | 0.03922 | 0.02616 | 0.6671 | 128 of 128 | 0.01882 | 0.4798 | 68 of 128 |
| full | both | trace-weighted | 0.03935 | 0.02633 | 0.6691 | 128 of 128 | 0.01912 | 0.4859 | 68 of 128 |
| reg | gate | unweighted | 0.03807 | 0.02724 | 0.7156 | 128 of 128 | 0.02841 | 0.7462 | 68 of 128 |
| reg | gate | count-weighted | 0.03619 | 0.02546 | 0.7034 | 128 of 128 | 0.02088 | 0.5771 | 68 of 128 |
| reg | gate | trace-weighted | 0.03635 | 0.02564 | 0.7052 | 128 of 128 | 0.02116 | 0.5820 | 68 of 128 |
| reg | up | unweighted | 0.04738 | 0.03391 | 0.7157 | 128 of 128 | 0.03444 | 0.7270 | 68 of 128 |
| reg | up | count-weighted | 0.04643 | 0.03271 | 0.7045 | 128 of 128 | 0.02597 | 0.5593 | 68 of 128 |
| reg | up | trace-weighted | 0.04648 | 0.03283 | 0.7063 | 128 of 128 | 0.02624 | 0.5644 | 68 of 128 |
| reg | both | unweighted | 0.04212 | 0.03015 | 0.7157 | 128 of 128 | 0.03107 | 0.7376 | 68 of 128 |
| reg | both | count-weighted | 0.04058 | 0.02856 | 0.7040 | 128 of 128 | 0.02308 | 0.5687 | 68 of 128 |
| reg | both | trace-weighted | 0.04070 | 0.02873 | 0.7058 | 128 of 128 | 0.02335 | 0.5738 | 68 of 128 |
| bias | gate | unweighted | 0.00712 | 0.00217 | 0.3054 | 128 of 128 | 0.00392 | 0.5507 | 68 of 128 |
| bias | gate | count-weighted | 0.00675 | 0.00186 | 0.2758 | 128 of 128 | 0.00164 | 0.2427 | 68 of 128 |
| bias | gate | trace-weighted | 0.00677 | 0.00190 | 0.2797 | 128 of 128 | 0.00170 | 0.2505 | 68 of 128 |
| bias | up | unweighted | 0.00894 | 0.00277 | 0.3103 | 128 of 128 | 0.00446 | 0.4992 | 68 of 128 |
| bias | up | count-weighted | 0.00877 | 0.00250 | 0.2847 | 128 of 128 | 0.00160 | 0.1819 | 68 of 128 |
| bias | up | trace-weighted | 0.00877 | 0.00253 | 0.2889 | 128 of 128 | 0.00167 | 0.1909 | 68 of 128 |
| bias | both | unweighted | 0.00790 | 0.00243 | 0.3078 | 128 of 128 | 0.00417 | 0.5274 | 68 of 128 |
| bias | both | count-weighted | 0.00760 | 0.00213 | 0.2801 | 128 of 128 | 0.00163 | 0.2138 | 68 of 128 |
| bias | both | trace-weighted | 0.00762 | 0.00216 | 0.2841 | 128 of 128 | 0.00169 | 0.2222 | 68 of 128 |
| rest | gate | unweighted | 0.02956 | 0.02277 | 0.7705 | 128 of 128 | 0.02152 | 0.7281 | 68 of 128 |
| rest | gate | count-weighted | 0.02801 | 0.02131 | 0.7608 | 128 of 128 | 0.01539 | 0.5494 | 68 of 128 |
| rest | gate | trace-weighted | 0.02815 | 0.02146 | 0.7622 | 128 of 128 | 0.01562 | 0.5548 | 68 of 128 |
| rest | up | unweighted | 0.03732 | 0.02868 | 0.7685 | 128 of 128 | 0.02653 | 0.7110 | 68 of 128 |
| rest | up | count-weighted | 0.03650 | 0.02772 | 0.7595 | 128 of 128 | 0.01960 | 0.5371 | 68 of 128 |
| rest | up | trace-weighted | 0.03655 | 0.02781 | 0.7609 | 128 of 128 | 0.01984 | 0.5427 | 68 of 128 |
| rest | both | unweighted | 0.03291 | 0.02533 | 0.7696 | 128 of 128 | 0.02371 | 0.7205 | 68 of 128 |
| rest | both | count-weighted | 0.03161 | 0.02403 | 0.7602 | 128 of 128 | 0.01719 | 0.5437 | 68 of 128 |
| rest | both | trace-weighted | 0.03173 | 0.02416 | 0.7616 | 128 of 128 | 0.01743 | 0.5493 | 68 of 128 |
| diag | gate | unweighted | 0.06287 | 0.06742 | 1.0723 | 0 of 128 | 0.08200 | 1.3043 | 0 of 128 |
| diag | gate | count-weighted | 0.06272 | 0.06734 | 1.0738 | 0 of 128 | 0.09194 | 1.4660 | 0 of 128 |
| diag | gate | trace-weighted | 0.06275 | 0.06736 | 1.0736 | 0 of 128 | 0.09162 | 1.4601 | 0 of 128 |
| diag | up | unweighted | 0.06286 | 0.06740 | 1.0724 | 0 of 128 | 0.08221 | 1.3079 | 0 of 128 |
| diag | up | count-weighted | 0.06269 | 0.06733 | 1.0739 | 0 of 128 | 0.09228 | 1.4720 | 0 of 128 |
| diag | up | trace-weighted | 0.06272 | 0.06735 | 1.0737 | 0 of 128 | 0.09195 | 1.4659 | 0 of 128 |
| diag | both | unweighted | 0.06286 | 0.06741 | 1.0723 | 0 of 128 | 0.08210 | 1.3061 | 0 of 128 |
| diag | both | count-weighted | 0.06270 | 0.06733 | 1.0738 | 0 of 128 | 0.09211 | 1.4690 | 0 of 128 |
| diag | both | trace-weighted | 0.06274 | 0.06736 | 1.0737 | 0 of 128 | 0.09178 | 1.4630 | 0 of 128 |
| plain | gate | unweighted | 0.26005 | 0.26790 | 1.0302 | 0 of 128 | 0.30147 | 1.1593 | 0 of 128 |
| plain | gate | count-weighted | 0.26009 | 0.26801 | 1.0305 | 0 of 128 | 0.32398 | 1.2456 | 0 of 128 |
| plain | gate | trace-weighted | 0.26007 | 0.26799 | 1.0305 | 0 of 128 | 0.32319 | 1.2427 | 0 of 128 |
| plain | up | unweighted | 0.25990 | 0.26775 | 1.0302 | 0 of 128 | 0.30186 | 1.1615 | 0 of 128 |
| plain | up | count-weighted | 0.25992 | 0.26785 | 1.0305 | 0 of 128 | 0.32468 | 1.2491 | 0 of 128 |
| plain | up | trace-weighted | 0.25991 | 0.26784 | 1.0305 | 0 of 128 | 0.32387 | 1.2461 | 0 of 128 |
| plain | both | unweighted | 0.25997 | 0.26783 | 1.0302 | 0 of 128 | 0.30167 | 1.1604 | 0 of 128 |
| plain | both | count-weighted | 0.26001 | 0.26793 | 1.0305 | 0 of 128 | 0.32433 | 1.2474 | 0 of 128 |
| plain | both | trace-weighted | 0.25999 | 0.26792 | 1.0305 | 0 of 128 | 0.32353 | 1.2444 | 0 of 128 |
| gain | gate | unweighted | 0.99668 | 0.99782 | 1.0011 | 6 of 128 | 0.98991 | 0.9932 | 68 of 128 |
| gain | gate | count-weighted | 0.99663 | 0.99785 | 1.0012 | 6 of 128 | 0.98621 | 0.9895 | 68 of 128 |
| gain | gate | trace-weighted | 0.99662 | 0.99784 | 1.0012 | 6 of 128 | 0.98631 | 0.9897 | 68 of 128 |
| gain | up | unweighted | 0.99666 | 0.99781 | 1.0012 | 5 of 128 | 0.98995 | 0.9933 | 68 of 128 |
| gain | up | count-weighted | 0.99660 | 0.99784 | 1.0012 | 5 of 128 | 0.98633 | 0.9897 | 68 of 128 |
| gain | up | trace-weighted | 0.99660 | 0.99783 | 1.0012 | 5 of 128 | 0.98645 | 0.9898 | 68 of 128 |
| gain | both | unweighted | 0.99667 | 0.99781 | 1.0011 | 5 of 128 | 0.98993 | 0.9932 | 68 of 128 |
| gain | both | count-weighted | 0.99662 | 0.99784 | 1.0012 | 5 of 128 | 0.98627 | 0.9896 | 68 of 128 |
| gain | both | trace-weighted | 0.99661 | 0.99783 | 1.0012 | 5 of 128 | 0.98638 | 0.9897 | 68 of 128 |
| energy | gate | unweighted | 1.06098 | 1.06741 | 1.0061 | 0 of 128 | 1.07226 | 1.0106 | 0 of 128 |
| energy | gate | count-weighted | 1.06091 | 1.06752 | 1.0062 | 0 of 128 | 1.07809 | 1.0162 | 0 of 128 |
| energy | gate | trace-weighted | 1.06088 | 1.06749 | 1.0062 | 0 of 128 | 1.07783 | 1.0160 | 0 of 128 |
| energy | up | unweighted | 1.06086 | 1.06731 | 1.0061 | 0 of 128 | 1.07263 | 1.0111 | 0 of 128 |
| energy | up | count-weighted | 1.06077 | 1.06743 | 1.0063 | 0 of 128 | 1.07882 | 1.0170 | 0 of 128 |
| energy | up | trace-weighted | 1.06075 | 1.06740 | 1.0063 | 0 of 128 | 1.07857 | 1.0168 | 0 of 128 |
| energy | both | unweighted | 1.06092 | 1.06736 | 1.0061 | 0 of 128 | 1.07244 | 1.0109 | 0 of 128 |
| energy | both | count-weighted | 1.06084 | 1.06747 | 1.0063 | 0 of 128 | 1.07845 | 1.0166 | 0 of 128 |
| energy | both | trace-weighted | 1.06082 | 1.06745 | 1.0062 | 0 of 128 | 1.07820 | 1.0164 | 0 of 128 |
| ogain | gate | unweighted | 0.99196 | 0.99914 | 1.0072 | 0 of 128 | 1.00280 | 1.0109 | 0 of 128 |
| ogain | gate | count-weighted | 0.99204 | 0.99926 | 1.0073 | 0 of 128 | 1.00931 | 1.0174 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99195 | 0.99923 | 1.0073 | 0 of 128 | 1.00906 | 1.0172 | 0 of 128 |
| ogain | up | unweighted | 0.99603 | 0.99870 | 1.0027 | 7 of 128 | 1.00365 | 1.0076 | 0 of 128 |
| ogain | up | count-weighted | 0.99608 | 0.99881 | 1.0027 | 7 of 128 | 1.00803 | 1.0120 | 0 of 128 |
| ogain | up | trace-weighted | 0.99604 | 0.99879 | 1.0028 | 7 of 128 | 1.00791 | 1.0119 | 0 of 128 |
| ogain | both | unweighted | 0.99376 | 0.99895 | 1.0052 | 0 of 128 | 1.00322 | 1.0095 | 0 of 128 |
| ogain | both | count-weighted | 0.99380 | 0.99907 | 1.0053 | 0 of 128 | 1.00878 | 1.0151 | 0 of 128 |
| ogain | both | trace-weighted | 0.99373 | 0.99905 | 1.0054 | 0 of 128 | 1.00859 | 1.0150 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.1589e-02 | [-1.1922e-02, -1.1251e-02] |
| up | -1.5054e-02 | [-1.5435e-02, -1.4670e-02] |
| both | -1.3056e-02 | [-1.3379e-02, -1.2730e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.7736e-02 | [-1.8743e-02, -1.6485e-02] | -6.1469e-03 | [-7.0831e-03, -5.0141e-03] |
| full | up | -2.4071e-02 | [-2.5324e-02, -2.2542e-02] | -9.0179e-03 | [-1.0204e-02, -7.5575e-03] |
| full | both | -2.0403e-02 | [-2.1473e-02, -1.9061e-02] | -7.3474e-03 | [-8.3797e-03, -6.0774e-03] |
| reg | gate | -1.5305e-02 | [-1.6169e-02, -1.4226e-02] | -4.5724e-03 | [-5.3713e-03, -3.6026e-03] |
| reg | up | -2.0463e-02 | [-2.1521e-02, -1.9155e-02] | -6.7415e-03 | [-7.7527e-03, -5.5052e-03] |
| reg | both | -1.7499e-02 | [-1.8406e-02, -1.6352e-02] | -5.4882e-03 | [-6.3688e-03, -4.4064e-03] |
| bias | gate | -5.1147e-03 | [-5.4652e-03, -4.7047e-03] | -2.2390e-04 | [-5.1321e-04, +1.4501e-04] |
| bias | up | -7.1745e-03 | [-7.6959e-03, -6.5846e-03] | -9.0105e-04 | [-1.3314e-03, -3.9251e-04] |
| bias | both | -5.9785e-03 | [-6.3862e-03, -5.5036e-03] | -5.0390e-04 | [-8.4963e-04, -7.8479e-05] |
| rest | gate | -1.2621e-02 | [-1.3392e-02, -1.1695e-02] | -5.9230e-03 | [-6.6194e-03, -5.0951e-03] |
| rest | up | -1.6897e-02 | [-1.7800e-02, -1.5805e-02] | -8.1168e-03 | [-8.9936e-03, -7.0844e-03] |
| rest | both | -1.4425e-02 | [-1.5231e-02, -1.3447e-02] | -6.8435e-03 | [-7.6085e-03, -5.9398e-03] |
| diag | gate | +2.9228e-02 | [+2.7333e-02, +3.0880e-02] | +2.4602e-02 | [+2.2724e-02, +2.6240e-02] |
| diag | up | +2.9588e-02 | [+2.7648e-02, +3.1253e-02] | +2.4953e-02 | [+2.3035e-02, +2.6614e-02] |
| diag | both | +2.9408e-02 | [+2.7496e-02, +3.1066e-02] | +2.4778e-02 | [+2.2882e-02, +2.6417e-02] |

## Layer 2 (IQ2_S; UD `.scale` present, min 3.92e-05, max 0.0001678)

- Gram layout: expert 49: max |strict lower| = 0, max |strict upper| = 5.719e+04
- Gram layout: expert 15: max |strict lower| = 0, max |strict upper| = 5.721e+04
- Gram layout: expert 112: max |strict lower| = 0, max |strict upper| = 2.789e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 2.44e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.2.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.566838 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 70, median 2702, max 15117

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03571 | 0.02556 | 0.7158 | 128 of 128 | 0.02327 | 0.6516 | 76 of 128 |
| full | gate | count-weighted | 0.03444 | 0.02423 | 0.7034 | 128 of 128 | 0.01715 | 0.4979 | 76 of 128 |
| full | gate | trace-weighted | 0.03454 | 0.02436 | 0.7053 | 128 of 128 | 0.01737 | 0.5029 | 76 of 128 |
| full | up | unweighted | 0.04750 | 0.03388 | 0.7132 | 128 of 128 | 0.02978 | 0.6269 | 76 of 128 |
| full | up | count-weighted | 0.04645 | 0.03260 | 0.7017 | 128 of 128 | 0.02182 | 0.4698 | 76 of 128 |
| full | up | trace-weighted | 0.04652 | 0.03274 | 0.7038 | 128 of 128 | 0.02210 | 0.4751 | 76 of 128 |
| full | both | unweighted | 0.04071 | 0.02909 | 0.7145 | 128 of 128 | 0.02606 | 0.6401 | 76 of 128 |
| full | both | count-weighted | 0.03950 | 0.02775 | 0.7026 | 128 of 128 | 0.01913 | 0.4843 | 76 of 128 |
| full | both | trace-weighted | 0.03959 | 0.02789 | 0.7046 | 128 of 128 | 0.01938 | 0.4894 | 76 of 128 |
| reg | gate | unweighted | 0.03712 | 0.02775 | 0.7478 | 128 of 128 | 0.02644 | 0.7124 | 76 of 128 |
| reg | gate | count-weighted | 0.03587 | 0.02642 | 0.7366 | 128 of 128 | 0.02098 | 0.5850 | 76 of 128 |
| reg | gate | trace-weighted | 0.03596 | 0.02655 | 0.7384 | 128 of 128 | 0.02118 | 0.5890 | 76 of 128 |
| reg | up | unweighted | 0.04852 | 0.03617 | 0.7454 | 128 of 128 | 0.03354 | 0.6913 | 76 of 128 |
| reg | up | count-weighted | 0.04752 | 0.03493 | 0.7351 | 128 of 128 | 0.02663 | 0.5604 | 76 of 128 |
| reg | up | trace-weighted | 0.04759 | 0.03507 | 0.7369 | 128 of 128 | 0.02688 | 0.5648 | 76 of 128 |
| reg | both | unweighted | 0.04200 | 0.03136 | 0.7466 | 128 of 128 | 0.02951 | 0.7025 | 76 of 128 |
| reg | both | count-weighted | 0.04083 | 0.03004 | 0.7359 | 128 of 128 | 0.02339 | 0.5730 | 76 of 128 |
| reg | both | trace-weighted | 0.04091 | 0.03018 | 0.7377 | 128 of 128 | 0.02362 | 0.5773 | 76 of 128 |
| bias | gate | unweighted | 0.00652 | 0.00206 | 0.3163 | 128 of 128 | 0.00322 | 0.4931 | 76 of 128 |
| bias | gate | count-weighted | 0.00636 | 0.00192 | 0.3017 | 128 of 128 | 0.00164 | 0.2585 | 76 of 128 |
| bias | gate | trace-weighted | 0.00645 | 0.00201 | 0.3114 | 128 of 128 | 0.00174 | 0.2701 | 76 of 128 |
| bias | up | unweighted | 0.00876 | 0.00277 | 0.3155 | 128 of 128 | 0.00378 | 0.4308 | 76 of 128 |
| bias | up | count-weighted | 0.00866 | 0.00263 | 0.3032 | 128 of 128 | 0.00157 | 0.1817 | 76 of 128 |
| bias | up | trace-weighted | 0.00877 | 0.00275 | 0.3132 | 128 of 128 | 0.00171 | 0.1944 | 76 of 128 |
| bias | both | unweighted | 0.00747 | 0.00236 | 0.3159 | 128 of 128 | 0.00346 | 0.4633 | 76 of 128 |
| bias | both | count-weighted | 0.00733 | 0.00222 | 0.3024 | 128 of 128 | 0.00162 | 0.2207 | 76 of 128 |
| bias | both | trace-weighted | 0.00743 | 0.00232 | 0.3122 | 128 of 128 | 0.00173 | 0.2329 | 76 of 128 |
| rest | gate | unweighted | 0.02919 | 0.02350 | 0.8050 | 128 of 128 | 0.02005 | 0.6870 | 76 of 128 |
| rest | gate | count-weighted | 0.02808 | 0.02231 | 0.7943 | 128 of 128 | 0.01551 | 0.5521 | 76 of 128 |
| rest | gate | trace-weighted | 0.02809 | 0.02235 | 0.7957 | 128 of 128 | 0.01563 | 0.5563 | 76 of 128 |
| rest | up | unweighted | 0.03874 | 0.03111 | 0.8032 | 128 of 128 | 0.02600 | 0.6712 | 76 of 128 |
| rest | up | count-weighted | 0.03779 | 0.02997 | 0.7931 | 128 of 128 | 0.02025 | 0.5359 | 76 of 128 |
| rest | up | trace-weighted | 0.03775 | 0.02999 | 0.7946 | 128 of 128 | 0.02040 | 0.5403 | 76 of 128 |
| rest | both | unweighted | 0.03324 | 0.02673 | 0.8041 | 128 of 128 | 0.02260 | 0.6798 | 76 of 128 |
| rest | both | count-weighted | 0.03217 | 0.02554 | 0.7938 | 128 of 128 | 0.01751 | 0.5443 | 76 of 128 |
| rest | both | trace-weighted | 0.03216 | 0.02557 | 0.7952 | 128 of 128 | 0.01765 | 0.5487 | 76 of 128 |
| diag | gate | unweighted | 0.06204 | 0.06629 | 1.0684 | 0 of 128 | 0.08298 | 1.3374 | 0 of 128 |
| diag | gate | count-weighted | 0.06197 | 0.06627 | 1.0693 | 0 of 128 | 0.09052 | 1.4607 | 0 of 128 |
| diag | gate | trace-weighted | 0.06197 | 0.06626 | 1.0692 | 0 of 128 | 0.09029 | 1.4568 | 0 of 128 |
| diag | up | unweighted | 0.06202 | 0.06627 | 1.0684 | 0 of 128 | 0.08332 | 1.3434 | 0 of 128 |
| diag | up | count-weighted | 0.06195 | 0.06624 | 1.0693 | 0 of 128 | 0.09100 | 1.4689 | 0 of 128 |
| diag | up | trace-weighted | 0.06195 | 0.06623 | 1.0692 | 0 of 128 | 0.09075 | 1.4649 | 0 of 128 |
| diag | both | unweighted | 0.06203 | 0.06628 | 1.0684 | 0 of 128 | 0.08315 | 1.3404 | 0 of 128 |
| diag | both | count-weighted | 0.06196 | 0.06625 | 1.0693 | 0 of 128 | 0.09076 | 1.4648 | 0 of 128 |
| diag | both | trace-weighted | 0.06196 | 0.06625 | 1.0692 | 0 of 128 | 0.09052 | 1.4609 | 0 of 128 |
| plain | gate | unweighted | 0.26189 | 0.26942 | 1.0288 | 0 of 128 | 0.31186 | 1.1908 | 0 of 128 |
| plain | gate | count-weighted | 0.26193 | 0.26950 | 1.0289 | 0 of 128 | 0.33085 | 1.2631 | 0 of 128 |
| plain | gate | trace-weighted | 0.26195 | 0.26953 | 1.0289 | 0 of 128 | 0.33030 | 1.2609 | 0 of 128 |
| plain | up | unweighted | 0.26177 | 0.26930 | 1.0288 | 0 of 128 | 0.31274 | 1.1947 | 0 of 128 |
| plain | up | count-weighted | 0.26179 | 0.26935 | 1.0289 | 0 of 128 | 0.33213 | 1.2687 | 0 of 128 |
| plain | up | trace-weighted | 0.26181 | 0.26938 | 1.0289 | 0 of 128 | 0.33155 | 1.2664 | 0 of 128 |
| plain | both | unweighted | 0.26183 | 0.26936 | 1.0288 | 0 of 128 | 0.31230 | 1.1928 | 0 of 128 |
| plain | both | count-weighted | 0.26186 | 0.26942 | 1.0289 | 0 of 128 | 0.33149 | 1.2659 | 0 of 128 |
| plain | both | trace-weighted | 0.26188 | 0.26945 | 1.0289 | 0 of 128 | 0.33093 | 1.2636 | 0 of 128 |
| gain | gate | unweighted | 0.99768 | 0.99868 | 1.0010 | 4 of 128 | 0.98997 | 0.9923 | 76 of 128 |
| gain | gate | count-weighted | 0.99757 | 0.99854 | 1.0010 | 4 of 128 | 0.98703 | 0.9894 | 76 of 128 |
| gain | gate | trace-weighted | 0.99759 | 0.99855 | 1.0010 | 4 of 128 | 0.98714 | 0.9895 | 76 of 128 |
| gain | up | unweighted | 0.99768 | 0.99869 | 1.0010 | 3 of 128 | 0.98989 | 0.9922 | 76 of 128 |
| gain | up | count-weighted | 0.99758 | 0.99856 | 1.0010 | 3 of 128 | 0.98690 | 0.9893 | 76 of 128 |
| gain | up | trace-weighted | 0.99760 | 0.99858 | 1.0010 | 3 of 128 | 0.98702 | 0.9894 | 76 of 128 |
| gain | both | unweighted | 0.99768 | 0.99869 | 1.0010 | 4 of 128 | 0.98993 | 0.9922 | 76 of 128 |
| gain | both | count-weighted | 0.99758 | 0.99855 | 1.0010 | 4 of 128 | 0.98697 | 0.9894 | 76 of 128 |
| gain | both | trace-weighted | 0.99760 | 0.99856 | 1.0010 | 4 of 128 | 0.98708 | 0.9895 | 76 of 128 |
| energy | gate | unweighted | 1.06394 | 1.06995 | 1.0057 | 0 of 128 | 1.07894 | 1.0141 | 0 of 128 |
| energy | gate | count-weighted | 1.06375 | 1.06971 | 1.0056 | 0 of 128 | 1.08440 | 1.0194 | 0 of 128 |
| energy | gate | trace-weighted | 1.06381 | 1.06975 | 1.0056 | 0 of 128 | 1.08431 | 1.0193 | 0 of 128 |
| energy | up | unweighted | 1.06388 | 1.06991 | 1.0057 | 0 of 128 | 1.07939 | 1.0146 | 0 of 128 |
| energy | up | count-weighted | 1.06369 | 1.06968 | 1.0056 | 0 of 128 | 1.08501 | 1.0200 | 0 of 128 |
| energy | up | trace-weighted | 1.06375 | 1.06971 | 1.0056 | 0 of 128 | 1.08494 | 1.0199 | 0 of 128 |
| energy | both | unweighted | 1.06391 | 1.06993 | 1.0057 | 0 of 128 | 1.07917 | 1.0143 | 0 of 128 |
| energy | both | count-weighted | 1.06372 | 1.06969 | 1.0056 | 0 of 128 | 1.08470 | 1.0197 | 0 of 128 |
| energy | both | trace-weighted | 1.06378 | 1.06973 | 1.0056 | 0 of 128 | 1.08462 | 1.0196 | 0 of 128 |
| ogain | gate | unweighted | 0.99555 | 0.99943 | 1.0039 | 8 of 128 | 1.00653 | 1.0110 | 0 of 128 |
| ogain | gate | count-weighted | 0.99583 | 0.99945 | 1.0036 | 8 of 128 | 1.01136 | 1.0156 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99580 | 0.99946 | 1.0037 | 8 of 128 | 1.01130 | 1.0156 | 0 of 128 |
| ogain | up | unweighted | 0.99687 | 0.99908 | 1.0022 | 6 of 128 | 1.00517 | 1.0083 | 0 of 128 |
| ogain | up | count-weighted | 0.99659 | 0.99896 | 1.0024 | 6 of 128 | 1.00827 | 1.0117 | 0 of 128 |
| ogain | up | trace-weighted | 0.99662 | 0.99896 | 1.0023 | 6 of 128 | 1.00835 | 1.0118 | 0 of 128 |
| ogain | both | unweighted | 0.99613 | 0.99928 | 1.0032 | 3 of 128 | 1.00597 | 1.0099 | 0 of 128 |
| ogain | both | count-weighted | 0.99616 | 0.99924 | 1.0031 | 3 of 128 | 1.01005 | 1.0139 | 0 of 128 |
| ogain | both | trace-weighted | 0.99616 | 0.99925 | 1.0031 | 3 of 128 | 1.01006 | 1.0139 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.0216e-02 | [-1.0454e-02, -9.9790e-03] |
| up | -1.3855e-02 | [-1.4102e-02, -1.3622e-02] |
| both | -1.1747e-02 | [-1.1970e-02, -1.1529e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.7292e-02 | [-1.8290e-02, -1.6168e-02] | -7.0764e-03 | [-8.0345e-03, -5.9836e-03] |
| full | up | -2.4629e-02 | [-2.5930e-02, -2.3133e-02] | -1.0774e-02 | [-1.2071e-02, -9.2847e-03] |
| full | both | -2.0369e-02 | [-2.1470e-02, -1.9111e-02] | -8.6227e-03 | [-9.7186e-03, -7.3728e-03] |
| reg | gate | -1.4886e-02 | [-1.5734e-02, -1.3921e-02] | -5.4393e-03 | [-6.2585e-03, -4.5016e-03] |
| reg | up | -2.0889e-02 | [-2.1981e-02, -1.9629e-02] | -8.3014e-03 | [-9.3901e-03, -7.0416e-03] |
| reg | both | -1.7432e-02 | [-1.8368e-02, -1.6360e-02] | -6.6492e-03 | [-7.5809e-03, -5.5876e-03] |
| bias | gate | -4.7147e-03 | [-5.0320e-03, -4.3500e-03] | -2.7464e-04 | [-5.5124e-04, +5.4007e-05] |
| bias | up | -7.0901e-03 | [-7.5662e-03, -6.5518e-03] | -1.0526e-03 | [-1.4676e-03, -5.6326e-04] |
| bias | both | -5.7105e-03 | [-6.0857e-03, -5.2794e-03] | -5.9858e-04 | [-9.3051e-04, -2.0365e-04] |
| rest | gate | -1.2578e-02 | [-1.3388e-02, -1.1708e-02] | -6.8018e-03 | [-7.5571e-03, -5.9796e-03] |
| rest | up | -1.7539e-02 | [-1.8575e-02, -1.6416e-02] | -9.7217e-03 | [-1.0723e-02, -8.6211e-03] |
| rest | both | -1.4659e-02 | [-1.5556e-02, -1.3690e-02] | -8.0242e-03 | [-8.8806e-03, -7.0814e-03] |
| diag | gate | +2.8550e-02 | [+2.6778e-02, +3.0153e-02] | +2.4255e-02 | [+2.2494e-02, +2.5859e-02] |
| diag | up | +2.9049e-02 | [+2.7246e-02, +3.0685e-02] | +2.4757e-02 | [+2.2960e-02, +2.6382e-02] |
| diag | both | +2.8799e-02 | [+2.7012e-02, +3.0425e-02] | +2.4506e-02 | [+2.2726e-02, +2.6124e-02] |

## Layer 3 (IQ2_S; UD `.scale` present, min 4.028e-05, max 0.0002603)

- Gram layout: expert 94: max |strict lower| = 0, max |strict upper| = 3.094e+04
- Gram layout: expert 6: max |strict lower| = 0, max |strict upper| = 4.2e+04
- Gram layout: expert 86: max |strict lower| = 0, max |strict upper| = 2.687e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 4.14e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.3.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.641106 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 45, median 2563, max 10005

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03633 | 0.02664 | 0.7333 | 128 of 128 | 0.02456 | 0.6760 | 74 of 128 |
| full | gate | count-weighted | 0.03457 | 0.02481 | 0.7176 | 128 of 128 | 0.01831 | 0.5296 | 74 of 128 |
| full | gate | trace-weighted | 0.03463 | 0.02488 | 0.7186 | 128 of 128 | 0.01847 | 0.5335 | 74 of 128 |
| full | up | unweighted | 0.04998 | 0.03655 | 0.7313 | 128 of 128 | 0.03244 | 0.6490 | 74 of 128 |
| full | up | count-weighted | 0.04818 | 0.03449 | 0.7159 | 128 of 128 | 0.02381 | 0.4943 | 74 of 128 |
| full | up | trace-weighted | 0.04824 | 0.03459 | 0.7169 | 128 of 128 | 0.02403 | 0.4981 | 74 of 128 |
| full | both | unweighted | 0.04200 | 0.03076 | 0.7324 | 128 of 128 | 0.02786 | 0.6633 | 74 of 128 |
| full | both | count-weighted | 0.04018 | 0.02880 | 0.7169 | 128 of 128 | 0.02059 | 0.5125 | 74 of 128 |
| full | both | trace-weighted | 0.04024 | 0.02889 | 0.7179 | 128 of 128 | 0.02078 | 0.5164 | 74 of 128 |
| reg | gate | unweighted | 0.03773 | 0.02880 | 0.7632 | 128 of 128 | 0.02764 | 0.7324 | 74 of 128 |
| reg | gate | count-weighted | 0.03600 | 0.02697 | 0.7491 | 128 of 128 | 0.02201 | 0.6112 | 74 of 128 |
| reg | gate | trace-weighted | 0.03606 | 0.02704 | 0.7499 | 128 of 128 | 0.02216 | 0.6144 | 74 of 128 |
| reg | up | unweighted | 0.05089 | 0.03875 | 0.7615 | 128 of 128 | 0.03608 | 0.7090 | 74 of 128 |
| reg | up | count-weighted | 0.04917 | 0.03676 | 0.7476 | 128 of 128 | 0.02853 | 0.5802 | 74 of 128 |
| reg | up | trace-weighted | 0.04924 | 0.03685 | 0.7485 | 128 of 128 | 0.02872 | 0.5833 | 74 of 128 |
| reg | both | unweighted | 0.04327 | 0.03299 | 0.7624 | 128 of 128 | 0.03121 | 0.7213 | 74 of 128 |
| reg | both | count-weighted | 0.04150 | 0.03106 | 0.7484 | 128 of 128 | 0.02474 | 0.5962 | 74 of 128 |
| reg | both | trace-weighted | 0.04156 | 0.03114 | 0.7493 | 128 of 128 | 0.02491 | 0.5994 | 74 of 128 |
| bias | gate | unweighted | 0.00612 | 0.00182 | 0.2980 | 128 of 128 | 0.00312 | 0.5096 | 74 of 128 |
| bias | gate | count-weighted | 0.00601 | 0.00172 | 0.2868 | 128 of 128 | 0.00173 | 0.2878 | 74 of 128 |
| bias | gate | trace-weighted | 0.00605 | 0.00176 | 0.2914 | 128 of 128 | 0.00178 | 0.2942 | 74 of 128 |
| bias | up | unweighted | 0.00850 | 0.00253 | 0.2980 | 128 of 128 | 0.00372 | 0.4376 | 74 of 128 |
| bias | up | count-weighted | 0.00848 | 0.00245 | 0.2886 | 128 of 128 | 0.00162 | 0.1913 | 74 of 128 |
| bias | up | trace-weighted | 0.00853 | 0.00250 | 0.2934 | 128 of 128 | 0.00169 | 0.1977 | 74 of 128 |
| bias | both | unweighted | 0.00711 | 0.00212 | 0.2979 | 128 of 128 | 0.00338 | 0.4751 | 74 of 128 |
| bias | both | count-weighted | 0.00702 | 0.00202 | 0.2876 | 128 of 128 | 0.00169 | 0.2406 | 74 of 128 |
| bias | both | trace-weighted | 0.00707 | 0.00207 | 0.2922 | 128 of 128 | 0.00175 | 0.2470 | 74 of 128 |
| rest | gate | unweighted | 0.03021 | 0.02481 | 0.8214 | 128 of 128 | 0.02144 | 0.7097 | 74 of 128 |
| rest | gate | count-weighted | 0.02856 | 0.02308 | 0.8083 | 128 of 128 | 0.01658 | 0.5804 | 74 of 128 |
| rest | gate | trace-weighted | 0.02858 | 0.02312 | 0.8091 | 128 of 128 | 0.01669 | 0.5842 | 74 of 128 |
| rest | up | unweighted | 0.04148 | 0.03402 | 0.8201 | 128 of 128 | 0.02872 | 0.6924 | 74 of 128 |
| rest | up | count-weighted | 0.03970 | 0.03204 | 0.8072 | 128 of 128 | 0.02219 | 0.5590 | 74 of 128 |
| rest | up | trace-weighted | 0.03971 | 0.03208 | 0.8079 | 128 of 128 | 0.02235 | 0.5627 | 74 of 128 |
| rest | both | unweighted | 0.03490 | 0.02865 | 0.8208 | 128 of 128 | 0.02449 | 0.7016 | 74 of 128 |
| rest | both | count-weighted | 0.03316 | 0.02678 | 0.8078 | 128 of 128 | 0.01890 | 0.5701 | 74 of 128 |
| rest | both | trace-weighted | 0.03317 | 0.02682 | 0.8086 | 128 of 128 | 0.01903 | 0.5738 | 74 of 128 |
| diag | gate | unweighted | 0.06254 | 0.06657 | 1.0643 | 0 of 128 | 0.08255 | 1.3198 | 0 of 128 |
| diag | gate | count-weighted | 0.06237 | 0.06646 | 1.0656 | 0 of 128 | 0.08988 | 1.4411 | 0 of 128 |
| diag | gate | trace-weighted | 0.06240 | 0.06649 | 1.0655 | 0 of 128 | 0.08971 | 1.4375 | 0 of 128 |
| diag | up | unweighted | 0.06251 | 0.06654 | 1.0643 | 0 of 128 | 0.08291 | 1.3263 | 0 of 128 |
| diag | up | count-weighted | 0.06234 | 0.06643 | 1.0657 | 0 of 128 | 0.09039 | 1.4501 | 0 of 128 |
| diag | up | trace-weighted | 0.06237 | 0.06646 | 1.0656 | 0 of 128 | 0.09022 | 1.4465 | 0 of 128 |
| diag | both | unweighted | 0.06253 | 0.06655 | 1.0643 | 0 of 128 | 0.08273 | 1.3231 | 0 of 128 |
| diag | both | count-weighted | 0.06235 | 0.06645 | 1.0657 | 0 of 128 | 0.09014 | 1.4456 | 0 of 128 |
| diag | both | trace-weighted | 0.06239 | 0.06648 | 1.0656 | 0 of 128 | 0.08996 | 1.4420 | 0 of 128 |
| plain | gate | unweighted | 0.26148 | 0.26857 | 1.0271 | 0 of 128 | 0.30792 | 1.1776 | 0 of 128 |
| plain | gate | count-weighted | 0.26170 | 0.26880 | 1.0271 | 0 of 128 | 0.32646 | 1.2474 | 0 of 128 |
| plain | gate | trace-weighted | 0.26172 | 0.26881 | 1.0271 | 0 of 128 | 0.32599 | 1.2456 | 0 of 128 |
| plain | up | unweighted | 0.26135 | 0.26845 | 1.0272 | 0 of 128 | 0.30892 | 1.1820 | 0 of 128 |
| plain | up | count-weighted | 0.26157 | 0.26866 | 1.0271 | 0 of 128 | 0.32794 | 1.2538 | 0 of 128 |
| plain | up | trace-weighted | 0.26157 | 0.26867 | 1.0271 | 0 of 128 | 0.32745 | 1.2519 | 0 of 128 |
| plain | both | unweighted | 0.26141 | 0.26851 | 1.0271 | 0 of 128 | 0.30842 | 1.1798 | 0 of 128 |
| plain | both | count-weighted | 0.26163 | 0.26873 | 1.0271 | 0 of 128 | 0.32720 | 1.2506 | 0 of 128 |
| plain | both | trace-weighted | 0.26164 | 0.26874 | 1.0271 | 0 of 128 | 0.32672 | 1.2487 | 0 of 128 |
| gain | gate | unweighted | 0.99750 | 0.99843 | 1.0009 | 3 of 128 | 0.99039 | 0.9929 | 74 of 128 |
| gain | gate | count-weighted | 0.99749 | 0.99841 | 1.0009 | 3 of 128 | 0.98769 | 0.9902 | 74 of 128 |
| gain | gate | trace-weighted | 0.99748 | 0.99840 | 1.0009 | 3 of 128 | 0.98777 | 0.9903 | 74 of 128 |
| gain | up | unweighted | 0.99749 | 0.99846 | 1.0010 | 3 of 128 | 0.99010 | 0.9926 | 74 of 128 |
| gain | up | count-weighted | 0.99747 | 0.99844 | 1.0010 | 3 of 128 | 0.98724 | 0.9897 | 74 of 128 |
| gain | up | trace-weighted | 0.99747 | 0.99844 | 1.0010 | 3 of 128 | 0.98733 | 0.9898 | 74 of 128 |
| gain | both | unweighted | 0.99750 | 0.99844 | 1.0009 | 3 of 128 | 0.99024 | 0.9927 | 74 of 128 |
| gain | both | count-weighted | 0.99748 | 0.99842 | 1.0009 | 3 of 128 | 0.98747 | 0.9900 | 74 of 128 |
| gain | both | trace-weighted | 0.99748 | 0.99842 | 1.0009 | 3 of 128 | 0.98755 | 0.9900 | 74 of 128 |
| energy | gate | unweighted | 1.06338 | 1.06900 | 1.0053 | 0 of 128 | 1.07722 | 1.0130 | 0 of 128 |
| energy | gate | count-weighted | 1.06347 | 1.06906 | 1.0053 | 0 of 128 | 1.08285 | 1.0182 | 0 of 128 |
| energy | gate | trace-weighted | 1.06346 | 1.06906 | 1.0053 | 0 of 128 | 1.08273 | 1.0181 | 0 of 128 |
| energy | up | unweighted | 1.06329 | 1.06898 | 1.0053 | 0 of 128 | 1.07732 | 1.0132 | 0 of 128 |
| energy | up | count-weighted | 1.06336 | 1.06906 | 1.0054 | 0 of 128 | 1.08295 | 1.0184 | 0 of 128 |
| energy | up | trace-weighted | 1.06336 | 1.06906 | 1.0054 | 0 of 128 | 1.08284 | 1.0183 | 0 of 128 |
| energy | both | unweighted | 1.06333 | 1.06899 | 1.0053 | 0 of 128 | 1.07727 | 1.0131 | 0 of 128 |
| energy | both | count-weighted | 1.06342 | 1.06906 | 1.0053 | 0 of 128 | 1.08290 | 1.0183 | 0 of 128 |
| energy | both | trace-weighted | 1.06341 | 1.06906 | 1.0053 | 0 of 128 | 1.08279 | 1.0182 | 0 of 128 |
| ogain | gate | unweighted | 0.99679 | 0.99930 | 1.0025 | 28 of 128 | 1.00682 | 1.0101 | 0 of 128 |
| ogain | gate | count-weighted | 0.99762 | 0.99928 | 1.0017 | 28 of 128 | 1.01180 | 1.0142 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99763 | 0.99927 | 1.0016 | 28 of 128 | 1.01179 | 1.0142 | 0 of 128 |
| ogain | up | unweighted | 0.99640 | 0.99883 | 1.0024 | 4 of 128 | 1.00392 | 1.0075 | 0 of 128 |
| ogain | up | count-weighted | 0.99624 | 0.99887 | 1.0026 | 4 of 128 | 1.00692 | 1.0107 | 0 of 128 |
| ogain | up | trace-weighted | 0.99622 | 0.99887 | 1.0027 | 4 of 128 | 1.00695 | 1.0108 | 0 of 128 |
| ogain | both | unweighted | 0.99664 | 0.99910 | 1.0025 | 13 of 128 | 1.00562 | 1.0090 | 0 of 128 |
| ogain | both | count-weighted | 0.99707 | 0.99911 | 1.0020 | 13 of 128 | 1.00978 | 1.0127 | 0 of 128 |
| ogain | both | trace-weighted | 0.99707 | 0.99910 | 1.0020 | 13 of 128 | 1.00978 | 1.0127 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -9.7623e-03 | [-9.9560e-03, -9.5836e-03] |
| up | -1.3685e-02 | [-1.3931e-02, -1.3430e-02] |
| both | -1.1375e-02 | [-1.1554e-02, -1.1200e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.6263e-02 | [-1.7333e-02, -1.5090e-02] | -6.5008e-03 | [-7.5501e-03, -5.3454e-03] |
| full | up | -2.4364e-02 | [-2.5839e-02, -2.2658e-02] | -1.0679e-02 | [-1.2153e-02, -9.0111e-03] |
| full | both | -1.9588e-02 | [-2.0801e-02, -1.8216e-02] | -8.2122e-03 | [-9.4263e-03, -6.8496e-03] |
| reg | gate | -1.3996e-02 | [-1.4904e-02, -1.2995e-02] | -4.9625e-03 | [-5.8547e-03, -3.9726e-03] |
| reg | up | -2.0643e-02 | [-2.1889e-02, -1.9202e-02] | -8.2324e-03 | [-9.4663e-03, -6.8192e-03] |
| reg | both | -1.6758e-02 | [-1.7790e-02, -1.5582e-02] | -6.3182e-03 | [-7.3416e-03, -5.1615e-03] |
| bias | gate | -4.2814e-03 | [-4.5740e-03, -3.9504e-03] | +5.6053e-06 | [-2.5844e-04, +3.1040e-04] |
| bias | up | -6.8541e-03 | [-7.3623e-03, -6.2896e-03] | -8.2448e-04 | [-1.2770e-03, -3.1576e-04] |
| bias | both | -5.3325e-03 | [-5.7025e-03, -4.9215e-03] | -3.2970e-04 | [-6.6474e-04, +5.0322e-05] |
| rest | gate | -1.1982e-02 | [-1.2831e-02, -1.1080e-02] | -6.5064e-03 | [-7.3335e-03, -5.6164e-03] |
| rest | up | -1.7510e-02 | [-1.8613e-02, -1.6272e-02] | -9.8545e-03 | [-1.0960e-02, -8.6341e-03] |
| rest | both | -1.4255e-02 | [-1.5199e-02, -1.3222e-02] | -7.8825e-03 | [-8.8167e-03, -6.8532e-03] |
| diag | gate | +2.7509e-02 | [+2.5621e-02, +2.9156e-02] | +2.3415e-02 | [+2.1538e-02, +2.5059e-02] |
| diag | up | +2.8057e-02 | [+2.6138e-02, +2.9739e-02] | +2.3962e-02 | [+2.2052e-02, +2.5633e-02] |
| diag | both | +2.7783e-02 | [+2.5877e-02, +2.9447e-02] | +2.3688e-02 | [+2.1797e-02, +2.5342e-02] |

## Layer 5 (IQ2_S; UD `.scale` present, min 4.09e-05, max 0.0001975)

- Gram layout: expert 120: max |strict lower| = 0, max |strict upper| = 1.323e+05
- Gram layout: expert 114: max |strict lower| = 0, max |strict upper| = 1.016e+05
- Gram layout: expert 112: max |strict lower| = 0, max |strict upper| = 8.736e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 1.21e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.5.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.433437 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 44, median 2502, max 12722

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03531 | 0.02442 | 0.6916 | 128 of 128 | 0.02460 | 0.6967 | 68 of 128 |
| full | gate | count-weighted | 0.03507 | 0.02393 | 0.6824 | 128 of 128 | 0.01785 | 0.5092 | 68 of 128 |
| full | gate | trace-weighted | 0.03544 | 0.02420 | 0.6828 | 128 of 128 | 0.01803 | 0.5087 | 68 of 128 |
| full | up | unweighted | 0.04714 | 0.03273 | 0.6942 | 128 of 128 | 0.03227 | 0.6845 | 68 of 128 |
| full | up | count-weighted | 0.04575 | 0.03135 | 0.6853 | 128 of 128 | 0.02271 | 0.4964 | 68 of 128 |
| full | up | trace-weighted | 0.04565 | 0.03129 | 0.6854 | 128 of 128 | 0.02265 | 0.4962 | 68 of 128 |
| full | both | unweighted | 0.03988 | 0.02763 | 0.6927 | 128 of 128 | 0.02759 | 0.6918 | 68 of 128 |
| full | both | count-weighted | 0.03912 | 0.02673 | 0.6833 | 128 of 128 | 0.01968 | 0.5029 | 68 of 128 |
| full | both | trace-weighted | 0.03935 | 0.02690 | 0.6835 | 128 of 128 | 0.01978 | 0.5026 | 68 of 128 |
| reg | gate | unweighted | 0.03674 | 0.02666 | 0.7258 | 128 of 128 | 0.02750 | 0.7486 | 68 of 128 |
| reg | gate | count-weighted | 0.03651 | 0.02620 | 0.7176 | 128 of 128 | 0.02165 | 0.5930 | 68 of 128 |
| reg | gate | trace-weighted | 0.03688 | 0.02648 | 0.7180 | 128 of 128 | 0.02185 | 0.5925 | 68 of 128 |
| reg | up | unweighted | 0.04827 | 0.03514 | 0.7281 | 128 of 128 | 0.03562 | 0.7380 | 68 of 128 |
| reg | up | count-weighted | 0.04694 | 0.03380 | 0.7201 | 128 of 128 | 0.02729 | 0.5814 | 68 of 128 |
| reg | up | trace-weighted | 0.04684 | 0.03373 | 0.7201 | 128 of 128 | 0.02722 | 0.5811 | 68 of 128 |
| reg | both | unweighted | 0.04126 | 0.02998 | 0.7268 | 128 of 128 | 0.03071 | 0.7443 | 68 of 128 |
| reg | both | count-weighted | 0.04052 | 0.02911 | 0.7183 | 128 of 128 | 0.02380 | 0.5873 | 68 of 128 |
| reg | both | trace-weighted | 0.04075 | 0.02928 | 0.7186 | 128 of 128 | 0.02391 | 0.5869 | 68 of 128 |
| bias | gate | unweighted | 0.00732 | 0.00227 | 0.3098 | 128 of 128 | 0.00424 | 0.5794 | 68 of 128 |
| bias | gate | count-weighted | 0.00680 | 0.00181 | 0.2662 | 128 of 128 | 0.00177 | 0.2604 | 68 of 128 |
| bias | gate | trace-weighted | 0.00696 | 0.00190 | 0.2736 | 128 of 128 | 0.00182 | 0.2614 | 68 of 128 |
| bias | up | unweighted | 0.00947 | 0.00286 | 0.3017 | 128 of 128 | 0.00511 | 0.5398 | 68 of 128 |
| bias | up | count-weighted | 0.00873 | 0.00226 | 0.2590 | 128 of 128 | 0.00185 | 0.2117 | 68 of 128 |
| bias | up | trace-weighted | 0.00883 | 0.00235 | 0.2658 | 128 of 128 | 0.00189 | 0.2139 | 68 of 128 |
| bias | both | unweighted | 0.00817 | 0.00251 | 0.3068 | 128 of 128 | 0.00459 | 0.5621 | 68 of 128 |
| bias | both | count-weighted | 0.00755 | 0.00199 | 0.2631 | 128 of 128 | 0.00179 | 0.2374 | 68 of 128 |
| bias | both | trace-weighted | 0.00769 | 0.00208 | 0.2703 | 128 of 128 | 0.00184 | 0.2389 | 68 of 128 |
| rest | gate | unweighted | 0.02798 | 0.02215 | 0.7915 | 128 of 128 | 0.02036 | 0.7274 | 68 of 128 |
| rest | gate | count-weighted | 0.02827 | 0.02212 | 0.7825 | 128 of 128 | 0.01608 | 0.5690 | 68 of 128 |
| rest | gate | trace-weighted | 0.02848 | 0.02230 | 0.7828 | 128 of 128 | 0.01621 | 0.5691 | 68 of 128 |
| rest | up | unweighted | 0.03767 | 0.02987 | 0.7929 | 128 of 128 | 0.02715 | 0.7209 | 68 of 128 |
| rest | up | count-weighted | 0.03702 | 0.02909 | 0.7859 | 128 of 128 | 0.02086 | 0.5636 | 68 of 128 |
| rest | up | trace-weighted | 0.03682 | 0.02894 | 0.7860 | 128 of 128 | 0.02076 | 0.5639 | 68 of 128 |
| rest | both | unweighted | 0.03171 | 0.02512 | 0.7921 | 128 of 128 | 0.02300 | 0.7252 | 68 of 128 |
| rest | both | count-weighted | 0.03157 | 0.02474 | 0.7838 | 128 of 128 | 0.01788 | 0.5664 | 68 of 128 |
| rest | both | trace-weighted | 0.03166 | 0.02482 | 0.7839 | 128 of 128 | 0.01794 | 0.5666 | 68 of 128 |
| diag | gate | unweighted | 0.06402 | 0.06840 | 1.0683 | 0 of 128 | 0.08222 | 1.2843 | 0 of 128 |
| diag | gate | count-weighted | 0.06405 | 0.06848 | 1.0692 | 0 of 128 | 0.09176 | 1.4327 | 0 of 128 |
| diag | gate | trace-weighted | 0.06407 | 0.06850 | 1.0691 | 0 of 128 | 0.09174 | 1.4320 | 0 of 128 |
| diag | up | unweighted | 0.06403 | 0.06839 | 1.0682 | 0 of 128 | 0.08256 | 1.2894 | 0 of 128 |
| diag | up | count-weighted | 0.06402 | 0.06845 | 1.0691 | 0 of 128 | 0.09220 | 1.4401 | 0 of 128 |
| diag | up | trace-weighted | 0.06404 | 0.06846 | 1.0690 | 0 of 128 | 0.09217 | 1.4392 | 0 of 128 |
| diag | both | unweighted | 0.06402 | 0.06839 | 1.0683 | 0 of 128 | 0.08239 | 1.2869 | 0 of 128 |
| diag | both | count-weighted | 0.06404 | 0.06846 | 1.0691 | 0 of 128 | 0.09198 | 1.4364 | 0 of 128 |
| diag | both | trace-weighted | 0.06406 | 0.06848 | 1.0691 | 0 of 128 | 0.09196 | 1.4356 | 0 of 128 |
| plain | gate | unweighted | 0.25844 | 0.26665 | 1.0318 | 0 of 128 | 0.29610 | 1.1457 | 0 of 128 |
| plain | gate | count-weighted | 0.25851 | 0.26676 | 1.0319 | 0 of 128 | 0.31619 | 1.2231 | 0 of 128 |
| plain | gate | trace-weighted | 0.25854 | 0.26679 | 1.0319 | 0 of 128 | 0.31619 | 1.2230 | 0 of 128 |
| plain | up | unweighted | 0.25833 | 0.26652 | 1.0317 | 0 of 128 | 0.29662 | 1.1482 | 0 of 128 |
| plain | up | count-weighted | 0.25838 | 0.26661 | 1.0319 | 0 of 128 | 0.31695 | 1.2267 | 0 of 128 |
| plain | up | trace-weighted | 0.25840 | 0.26663 | 1.0318 | 0 of 128 | 0.31692 | 1.2265 | 0 of 128 |
| plain | both | unweighted | 0.25839 | 0.26658 | 1.0317 | 0 of 128 | 0.29636 | 1.1469 | 0 of 128 |
| plain | both | count-weighted | 0.25845 | 0.26668 | 1.0319 | 0 of 128 | 0.31657 | 1.2249 | 0 of 128 |
| plain | both | trace-weighted | 0.25847 | 0.26671 | 1.0319 | 0 of 128 | 0.31656 | 1.2247 | 0 of 128 |
| gain | gate | unweighted | 0.99566 | 0.99687 | 1.0012 | 2 of 128 | 0.98936 | 0.9937 | 68 of 128 |
| gain | gate | count-weighted | 0.99563 | 0.99684 | 1.0012 | 2 of 128 | 0.98606 | 0.9904 | 68 of 128 |
| gain | gate | trace-weighted | 0.99563 | 0.99684 | 1.0012 | 2 of 128 | 0.98611 | 0.9904 | 68 of 128 |
| gain | up | unweighted | 0.99566 | 0.99686 | 1.0012 | 2 of 128 | 0.98939 | 0.9937 | 68 of 128 |
| gain | up | count-weighted | 0.99564 | 0.99682 | 1.0012 | 2 of 128 | 0.98611 | 0.9904 | 68 of 128 |
| gain | up | trace-weighted | 0.99563 | 0.99681 | 1.0012 | 2 of 128 | 0.98614 | 0.9905 | 68 of 128 |
| gain | both | unweighted | 0.99566 | 0.99687 | 1.0012 | 2 of 128 | 0.98938 | 0.9937 | 68 of 128 |
| gain | both | count-weighted | 0.99563 | 0.99683 | 1.0012 | 2 of 128 | 0.98609 | 0.9904 | 68 of 128 |
| gain | both | trace-weighted | 0.99563 | 0.99682 | 1.0012 | 2 of 128 | 0.98613 | 0.9905 | 68 of 128 |
| energy | gate | unweighted | 1.05811 | 1.06485 | 1.0064 | 0 of 128 | 1.06769 | 1.0091 | 0 of 128 |
| energy | gate | count-weighted | 1.05809 | 1.06484 | 1.0064 | 0 of 128 | 1.07275 | 1.0138 | 0 of 128 |
| energy | gate | trace-weighted | 1.05810 | 1.06485 | 1.0064 | 0 of 128 | 1.07285 | 1.0139 | 0 of 128 |
| energy | up | unweighted | 1.05805 | 1.06475 | 1.0063 | 0 of 128 | 1.06809 | 1.0095 | 0 of 128 |
| energy | up | count-weighted | 1.05803 | 1.06471 | 1.0063 | 0 of 128 | 1.07334 | 1.0145 | 0 of 128 |
| energy | up | trace-weighted | 1.05803 | 1.06471 | 1.0063 | 0 of 128 | 1.07338 | 1.0145 | 0 of 128 |
| energy | both | unweighted | 1.05808 | 1.06480 | 1.0064 | 0 of 128 | 1.06789 | 1.0093 | 0 of 128 |
| energy | both | count-weighted | 1.05806 | 1.06478 | 1.0063 | 0 of 128 | 1.07304 | 1.0142 | 0 of 128 |
| energy | both | trace-weighted | 1.05807 | 1.06478 | 1.0063 | 0 of 128 | 1.07311 | 1.0142 | 0 of 128 |
| ogain | gate | unweighted | 0.99337 | 0.99843 | 1.0051 | 12 of 128 | 1.00345 | 1.0102 | 0 of 128 |
| ogain | gate | count-weighted | 0.99356 | 0.99836 | 1.0048 | 12 of 128 | 1.00902 | 1.0156 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99378 | 0.99833 | 1.0046 | 12 of 128 | 1.00922 | 1.0155 | 0 of 128 |
| ogain | up | unweighted | 0.99480 | 0.99818 | 1.0034 | 7 of 128 | 1.00242 | 1.0077 | 0 of 128 |
| ogain | up | count-weighted | 0.99473 | 0.99815 | 1.0034 | 7 of 128 | 1.00672 | 1.0121 | 0 of 128 |
| ogain | up | trace-weighted | 0.99473 | 0.99817 | 1.0035 | 7 of 128 | 1.00688 | 1.0122 | 0 of 128 |
| ogain | both | unweighted | 0.99386 | 0.99834 | 1.0045 | 3 of 128 | 1.00298 | 1.0092 | 0 of 128 |
| ogain | both | count-weighted | 0.99394 | 0.99830 | 1.0044 | 3 of 128 | 1.00804 | 1.0142 | 0 of 128 |
| ogain | both | trace-weighted | 0.99407 | 0.99829 | 1.0042 | 3 of 128 | 1.00821 | 1.0142 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.1137e-02 | [-1.1697e-02, -1.0536e-02] |
| up | -1.4396e-02 | [-1.4651e-02, -1.4133e-02] |
| both | -1.2391e-02 | [-1.2827e-02, -1.1914e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.7213e-02 | [-1.8500e-02, -1.5700e-02] | -6.0753e-03 | [-7.0419e-03, -4.9178e-03] |
| full | up | -2.3038e-02 | [-2.4358e-02, -2.1471e-02] | -8.6421e-03 | [-9.9365e-03, -7.1204e-03] |
| full | both | -1.9445e-02 | [-2.0666e-02, -1.7970e-02] | -7.0546e-03 | [-8.1024e-03, -5.7817e-03] |
| reg | gate | -1.4859e-02 | [-1.5961e-02, -1.3565e-02] | -4.5493e-03 | [-5.3713e-03, -3.5719e-03] |
| reg | up | -1.9648e-02 | [-2.0766e-02, -1.8316e-02] | -6.5091e-03 | [-7.5995e-03, -5.2147e-03] |
| reg | both | -1.6723e-02 | [-1.7770e-02, -1.5464e-02] | -5.3095e-03 | [-6.2085e-03, -4.2236e-03] |
| bias | gate | -5.0283e-03 | [-5.5265e-03, -4.4941e-03] | -3.9517e-05 | [-3.6455e-04, +3.4027e-04] |
| bias | up | -6.8836e-03 | [-7.3343e-03, -6.3593e-03] | -4.1230e-04 | [-8.3151e-04, +8.9841e-05] |
| bias | both | -5.7572e-03 | [-6.2338e-03, -5.2338e-03] | -1.9362e-04 | [-5.5393e-04, +2.3186e-04] |
| rest | gate | -1.2184e-02 | [-1.3059e-02, -1.1169e-02] | -6.0358e-03 | [-6.7156e-03, -5.2174e-03] |
| rest | up | -1.6155e-02 | [-1.7132e-02, -1.5037e-02] | -8.2297e-03 | [-9.1860e-03, -7.1307e-03] |
| rest | both | -1.3688e-02 | [-1.4525e-02, -1.2675e-02] | -6.8609e-03 | [-7.6023e-03, -5.9733e-03] |
| diag | gate | +2.7711e-02 | [+2.5868e-02, +2.9282e-02] | +2.3281e-02 | [+2.1445e-02, +2.4849e-02] |
| diag | up | +2.8177e-02 | [+2.6311e-02, +2.9777e-02] | +2.3755e-02 | [+2.1897e-02, +2.5352e-02] |
| diag | both | +2.7944e-02 | [+2.6089e-02, +2.9524e-02] | +2.3518e-02 | [+2.1674e-02, +2.5102e-02] |

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

In layer 1 (both halves, count-weighted), UD's full error is 0.02616 against our 0.03922 (ratio 0.667; UD - ours -1.306e-02, 95% CI [-1.338e-02, -1.273e-02]) and its diagonal error 0.06733 against our 0.06270 (ratio 1.074); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 1 (added after registration, not part of the rule): the input-mean part is 0.00760 ours against 0.00213 UD (ratio 0.280), the centred rest 0.03161 against 0.02403 (ratio 0.760). In layer 2 (both halves, count-weighted), UD's full error is 0.02775 against our 0.03950 (ratio 0.703; UD - ours -1.175e-02, 95% CI [-1.197e-02, -1.153e-02]) and its diagonal error 0.06625 against our 0.06196 (ratio 1.069); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 2 (added after registration, not part of the rule): the input-mean part is 0.00733 ours against 0.00222 UD (ratio 0.302), the centred rest 0.03217 against 0.02554 (ratio 0.794). In layer 3 (both halves, count-weighted), UD's full error is 0.02880 against our 0.04018 (ratio 0.717; UD - ours -1.138e-02, 95% CI [-1.155e-02, -1.120e-02]) and its diagonal error 0.06645 against our 0.06235 (ratio 1.066); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 3 (added after registration, not part of the rule): the input-mean part is 0.00702 ours against 0.00202 UD (ratio 0.288), the centred rest 0.03316 against 0.02678 (ratio 0.808). In layer 5 (both halves, count-weighted), UD's full error is 0.02673 against our 0.03912 (ratio 0.683; UD - ours -1.239e-02, 95% CI [-1.283e-02, -1.191e-02]) and its diagonal error 0.06846 against our 0.06404 (ratio 1.069); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 5 (added after registration, not part of the rule): the input-mean part is 0.00755 ours against 0.00199 UD (ratio 0.263), the centred rest 0.03157 against 0.02474 (ratio 0.784). Under the registered rule this is "supported" for both halves together, count-weighted; across the three means and three halves the branches are: supported.

## refined against ours and UD

In layer 1 (both halves, count-weighted), refined's full error is 0.01882 against our 0.03922 and UD's 0.02616 (refined - ours -2.040e-02, 95% CI [-2.147e-02, -1.906e-02]; (ours - refined) / (ours - UD) = 1.563; lower than ours in 68 of 128 experts). Bias part 0.00163 (ours 0.00760, UD 0.00213), rest 0.01719 (ours 0.03161, UD 0.02403). Regularised (the search objective) 0.02308 (ours 0.04058, UD 0.02856). Diagonal 0.09211 (ours 0.06270, UD 0.06733), plain 0.32433 (ours 0.26001, UD 0.26793). Gains (ours / UD / refined): plain 0.9966 / 0.9978 / 0.9863, energy 1.0608 / 1.0675 / 1.0785, output 0.9938 / 0.9991 / 1.0088. In layer 2 (both halves, count-weighted), refined's full error is 0.01913 against our 0.03950 and UD's 0.02775 (refined - ours -2.037e-02, 95% CI [-2.147e-02, -1.911e-02]; (ours - refined) / (ours - UD) = 1.734; lower than ours in 76 of 128 experts). Bias part 0.00162 (ours 0.00733, UD 0.00222), rest 0.01751 (ours 0.03217, UD 0.02554). Regularised (the search objective) 0.02339 (ours 0.04083, UD 0.03004). Diagonal 0.09076 (ours 0.06196, UD 0.06625), plain 0.33149 (ours 0.26186, UD 0.26942). Gains (ours / UD / refined): plain 0.9976 / 0.9986 / 0.9870, energy 1.0637 / 1.0697 / 1.0847, output 0.9962 / 0.9992 / 1.0101. In layer 3 (both halves, count-weighted), refined's full error is 0.02059 against our 0.04018 and UD's 0.02880 (refined - ours -1.959e-02, 95% CI [-2.080e-02, -1.822e-02]; (ours - refined) / (ours - UD) = 1.722; lower than ours in 74 of 128 experts). Bias part 0.00169 (ours 0.00702, UD 0.00202), rest 0.01890 (ours 0.03316, UD 0.02678). Regularised (the search objective) 0.02474 (ours 0.04150, UD 0.03106). Diagonal 0.09014 (ours 0.06235, UD 0.06645), plain 0.32720 (ours 0.26163, UD 0.26873). Gains (ours / UD / refined): plain 0.9975 / 0.9984 / 0.9875, energy 1.0634 / 1.0691 / 1.0829, output 0.9971 / 0.9991 / 1.0098. In layer 5 (both halves, count-weighted), refined's full error is 0.01968 against our 0.03912 and UD's 0.02673 (refined - ours -1.945e-02, 95% CI [-2.067e-02, -1.797e-02]; (ours - refined) / (ours - UD) = 1.569; lower than ours in 68 of 128 experts). Bias part 0.00179 (ours 0.00755, UD 0.00199), rest 0.01788 (ours 0.03157, UD 0.02474). Regularised (the search objective) 0.02380 (ours 0.04052, UD 0.02911). Diagonal 0.09198 (ours 0.06404, UD 0.06846), plain 0.31657 (ours 0.25845, UD 0.26668). Gains (ours / UD / refined): plain 0.9956 / 0.9968 / 0.9861, energy 1.0581 / 1.0648 / 1.0730, output 0.9939 / 0.9983 / 1.0080.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined (count-w.) | refined / ours | refined / UD |
|---|---|---|---|---|---|---|---|---|
| 1 | < 2304 | 60 | 0.135 | 0.04218 | 0.02913 | 0.04218 | 1.000 | 1.448 |
| 1 | >= 2304 | 68 | 0.865 | 0.03876 | 0.02570 | 0.01517 | 0.391 | 0.590 |
| 2 | < 2304 | 52 | 0.144 | 0.04188 | 0.03030 | 0.04188 | 1.000 | 1.382 |
| 2 | >= 2304 | 76 | 0.856 | 0.03910 | 0.02732 | 0.01530 | 0.391 | 0.560 |
| 3 | < 2304 | 54 | 0.166 | 0.04452 | 0.03345 | 0.04452 | 1.000 | 1.331 |
| 3 | >= 2304 | 74 | 0.834 | 0.03931 | 0.02788 | 0.01582 | 0.402 | 0.568 |
| 5 | < 2304 | 60 | 0.152 | 0.03999 | 0.02820 | 0.03999 | 1.000 | 1.418 |
| 5 | >= 2304 | 68 | 0.848 | 0.03896 | 0.02647 | 0.01602 | 0.411 | 0.605 |

Refinement log `models/refined/gu-refined-1-2-3-5.gguf.json` (scripts/refine-codes.py; wall time 4392 s for 512 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 68 | 60 | 4: 68 | 31293886 | 1409413 of 1410048 | 4656 | 5.12e-10 | 4.19e-10 | yes |
| 2 | 76 | 52 | 4: 76 | 35577599 | 1575034 of 1575936 | 5083 | 4.61e-10 | 7.13e-10 | yes |
| 3 | 74 | 54 | 4: 74 | 34683413 | 1533827 of 1534464 | 4393 | 8.85e-10 | 5.87e-10 | yes |
| 5 | 68 | 60 | 4: 68 | 31527676 | 1409434 of 1410048 | 3032 | 5.29e-10 | 6.86e-10 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 1: max abs(log - here) on reg 4.75e-10 before, 3.97e-10 after; layer 2: max abs(log - here) on reg 4.39e-10 before, 6.30e-10 after; layer 3: max abs(log - here) on reg 8.47e-10 before, 5.46e-10 after; layer 5: max abs(log - here) on reg 5.11e-10 before, 6.61e-10 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304').
