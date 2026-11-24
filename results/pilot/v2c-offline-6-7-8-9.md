# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined = `models/refined/gu-refined-mixc-6-7-8-9.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/v2c-offline-6-7-8-9.tsv`.
Grams: `models/grams/gu-6-7-8-9-mixc` (capture settings in its log, not restated here), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

## Layer 6 (IQ2_S; UD `.scale` present, min 3.867e-05, max 0.0001495)

- Gram layout: expert 91: max |strict lower| = 0, max |strict upper| = 2.615e+05
- Gram layout: expert 23: max |strict lower| = 0, max |strict upper| = 4.207e+05
- Gram layout: expert 73: max |strict lower| = 0, max |strict upper| = 1.553e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 2.04e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.6.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.692047 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 1120, median 35578, max 141389

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.02996 | 0.01917 | 0.6397 | 128 of 128 | 0.01614 | 0.5387 | 126 of 128 |
| full | gate | count-weighted | 0.03062 | 0.01963 | 0.6410 | 128 of 128 | 0.01688 | 0.5511 | 126 of 128 |
| full | gate | trace-weighted | 0.03053 | 0.01958 | 0.6413 | 128 of 128 | 0.01686 | 0.5522 | 126 of 128 |
| full | up | unweighted | 0.04095 | 0.02610 | 0.6373 | 128 of 128 | 0.02104 | 0.5139 | 126 of 128 |
| full | up | count-weighted | 0.04106 | 0.02625 | 0.6395 | 128 of 128 | 0.02158 | 0.5257 | 126 of 128 |
| full | up | trace-weighted | 0.04108 | 0.02628 | 0.6398 | 128 of 128 | 0.02163 | 0.5266 | 126 of 128 |
| full | both | unweighted | 0.03438 | 0.02196 | 0.6387 | 128 of 128 | 0.01810 | 0.5265 | 126 of 128 |
| full | both | count-weighted | 0.03492 | 0.02236 | 0.6403 | 128 of 128 | 0.01881 | 0.5387 | 126 of 128 |
| full | both | trace-weighted | 0.03485 | 0.02233 | 0.6407 | 128 of 128 | 0.01881 | 0.5397 | 126 of 128 |
| reg | gate | unweighted | 0.03137 | 0.02129 | 0.6787 | 128 of 128 | 0.01921 | 0.6126 | 126 of 128 |
| reg | gate | count-weighted | 0.03205 | 0.02179 | 0.6799 | 128 of 128 | 0.01992 | 0.6216 | 126 of 128 |
| reg | gate | trace-weighted | 0.03195 | 0.02173 | 0.6801 | 128 of 128 | 0.01989 | 0.6225 | 126 of 128 |
| reg | up | unweighted | 0.04222 | 0.02856 | 0.6765 | 128 of 128 | 0.02492 | 0.5902 | 126 of 128 |
| reg | up | count-weighted | 0.04234 | 0.02872 | 0.6784 | 128 of 128 | 0.02536 | 0.5989 | 126 of 128 |
| reg | up | trace-weighted | 0.04236 | 0.02875 | 0.6788 | 128 of 128 | 0.02540 | 0.5997 | 126 of 128 |
| reg | both | unweighted | 0.03577 | 0.02425 | 0.6778 | 128 of 128 | 0.02152 | 0.6016 | 126 of 128 |
| reg | both | count-weighted | 0.03632 | 0.02467 | 0.6792 | 128 of 128 | 0.02218 | 0.6105 | 126 of 128 |
| reg | both | trace-weighted | 0.03626 | 0.02464 | 0.6795 | 128 of 128 | 0.02217 | 0.6114 | 126 of 128 |
| bias | gate | unweighted | 0.00485 | 8.752e-04 | 0.1803 | 128 of 128 | 8.072e-04 | 0.1663 | 126 of 128 |
| bias | gate | count-weighted | 0.00474 | 8.126e-04 | 0.1713 | 128 of 128 | 6.913e-04 | 0.1458 | 126 of 128 |
| bias | gate | trace-weighted | 0.00472 | 8.014e-04 | 0.1699 | 128 of 128 | 6.914e-04 | 0.1466 | 126 of 128 |
| bias | up | unweighted | 0.00679 | 0.00120 | 0.1773 | 128 of 128 | 5.683e-04 | 0.0837 | 126 of 128 |
| bias | up | count-weighted | 0.00647 | 0.00109 | 0.1682 | 128 of 128 | 3.978e-04 | 0.0615 | 126 of 128 |
| bias | up | trace-weighted | 0.00646 | 0.00108 | 0.1670 | 128 of 128 | 3.959e-04 | 0.0613 | 126 of 128 |
| bias | both | unweighted | 0.00563 | 0.00101 | 0.1789 | 128 of 128 | 7.053e-04 | 0.1254 | 126 of 128 |
| bias | both | count-weighted | 0.00545 | 9.260e-04 | 0.1699 | 128 of 128 | 5.679e-04 | 0.1042 | 126 of 128 |
| bias | both | trace-weighted | 0.00543 | 9.150e-04 | 0.1686 | 128 of 128 | 5.672e-04 | 0.1045 | 126 of 128 |
| rest | gate | unweighted | 0.02511 | 0.01829 | 0.7285 | 128 of 128 | 0.01533 | 0.6106 | 126 of 128 |
| rest | gate | count-weighted | 0.02588 | 0.01882 | 0.7271 | 128 of 128 | 0.01618 | 0.6253 | 126 of 128 |
| rest | gate | trace-weighted | 0.02581 | 0.01877 | 0.7274 | 128 of 128 | 0.01617 | 0.6263 | 126 of 128 |
| rest | up | unweighted | 0.03416 | 0.02490 | 0.7288 | 128 of 128 | 0.02048 | 0.5994 | 126 of 128 |
| rest | up | count-weighted | 0.03458 | 0.02516 | 0.7277 | 128 of 128 | 0.02118 | 0.6126 | 126 of 128 |
| rest | up | trace-weighted | 0.03461 | 0.02520 | 0.7281 | 128 of 128 | 0.02123 | 0.6134 | 126 of 128 |
| rest | both | unweighted | 0.02875 | 0.02095 | 0.7287 | 128 of 128 | 0.01740 | 0.6050 | 126 of 128 |
| rest | both | count-weighted | 0.02946 | 0.02143 | 0.7274 | 128 of 128 | 0.01824 | 0.6191 | 126 of 128 |
| rest | both | trace-weighted | 0.02942 | 0.02141 | 0.7278 | 128 of 128 | 0.01824 | 0.6200 | 126 of 128 |
| diag | gate | unweighted | 0.06145 | 0.06565 | 1.0683 | 0 of 128 | 0.08265 | 1.3450 | 0 of 128 |
| diag | gate | count-weighted | 0.06170 | 0.06589 | 1.0679 | 0 of 128 | 0.08176 | 1.3251 | 0 of 128 |
| diag | gate | trace-weighted | 0.06171 | 0.06590 | 1.0679 | 0 of 128 | 0.08171 | 1.3241 | 0 of 128 |
| diag | up | unweighted | 0.06161 | 0.06581 | 1.0683 | 0 of 128 | 0.08331 | 1.3523 | 0 of 128 |
| diag | up | count-weighted | 0.06182 | 0.06602 | 1.0680 | 0 of 128 | 0.08230 | 1.3314 | 0 of 128 |
| diag | up | trace-weighted | 0.06183 | 0.06603 | 1.0679 | 0 of 128 | 0.08226 | 1.3304 | 0 of 128 |
| diag | both | unweighted | 0.06153 | 0.06573 | 1.0683 | 0 of 128 | 0.08298 | 1.3486 | 0 of 128 |
| diag | both | count-weighted | 0.06176 | 0.06595 | 1.0680 | 0 of 128 | 0.08203 | 1.3282 | 0 of 128 |
| diag | both | trace-weighted | 0.06177 | 0.06596 | 1.0679 | 0 of 128 | 0.08199 | 1.3272 | 0 of 128 |
| plain | gate | unweighted | 0.26149 | 0.26828 | 1.0259 | 0 of 128 | 0.31860 | 1.2184 | 0 of 128 |
| plain | gate | count-weighted | 0.26137 | 0.26813 | 1.0258 | 0 of 128 | 0.31611 | 1.2094 | 0 of 128 |
| plain | gate | trace-weighted | 0.26142 | 0.26817 | 1.0258 | 0 of 128 | 0.31623 | 1.2097 | 0 of 128 |
| plain | up | unweighted | 0.26148 | 0.26826 | 1.0259 | 0 of 128 | 0.31990 | 1.2234 | 0 of 128 |
| plain | up | count-weighted | 0.26133 | 0.26808 | 1.0258 | 0 of 128 | 0.31725 | 1.2140 | 0 of 128 |
| plain | up | trace-weighted | 0.26138 | 0.26812 | 1.0258 | 0 of 128 | 0.31738 | 1.2143 | 0 of 128 |
| plain | both | unweighted | 0.26149 | 0.26827 | 1.0259 | 0 of 128 | 0.31925 | 1.2209 | 0 of 128 |
| plain | both | count-weighted | 0.26135 | 0.26810 | 1.0258 | 0 of 128 | 0.31668 | 1.2117 | 0 of 128 |
| plain | both | trace-weighted | 0.26140 | 0.26814 | 1.0258 | 0 of 128 | 0.31681 | 1.2120 | 0 of 128 |
| gain | gate | unweighted | 0.99629 | 0.99714 | 1.0009 | 9 of 128 | 0.98723 | 0.9909 | 126 of 128 |
| gain | gate | count-weighted | 0.99633 | 0.99726 | 1.0009 | 9 of 128 | 0.98769 | 0.9913 | 126 of 128 |
| gain | gate | trace-weighted | 0.99632 | 0.99725 | 1.0009 | 9 of 128 | 0.98766 | 0.9913 | 126 of 128 |
| gain | up | unweighted | 0.99630 | 0.99717 | 1.0009 | 11 of 128 | 0.98707 | 0.9907 | 126 of 128 |
| gain | up | count-weighted | 0.99635 | 0.99729 | 1.0009 | 11 of 128 | 0.98755 | 0.9912 | 126 of 128 |
| gain | up | trace-weighted | 0.99635 | 0.99729 | 1.0009 | 11 of 128 | 0.98753 | 0.9912 | 126 of 128 |
| gain | both | unweighted | 0.99629 | 0.99716 | 1.0009 | 11 of 128 | 0.98715 | 0.9908 | 126 of 128 |
| gain | both | count-weighted | 0.99634 | 0.99727 | 1.0009 | 11 of 128 | 0.98762 | 0.9912 | 126 of 128 |
| gain | both | trace-weighted | 0.99634 | 0.99727 | 1.0009 | 11 of 128 | 0.98760 | 0.9912 | 126 of 128 |
| energy | gate | unweighted | 1.06096 | 1.06626 | 1.0050 | 0 of 128 | 1.07608 | 1.0143 | 0 of 128 |
| energy | gate | count-weighted | 1.06098 | 1.06640 | 1.0051 | 0 of 128 | 1.07532 | 1.0135 | 0 of 128 |
| energy | gate | trace-weighted | 1.06099 | 1.06641 | 1.0051 | 0 of 128 | 1.07534 | 1.0135 | 0 of 128 |
| energy | up | unweighted | 1.06098 | 1.06631 | 1.0050 | 0 of 128 | 1.07659 | 1.0147 | 0 of 128 |
| energy | up | count-weighted | 1.06100 | 1.06644 | 1.0051 | 0 of 128 | 1.07577 | 1.0139 | 0 of 128 |
| energy | up | trace-weighted | 1.06102 | 1.06646 | 1.0051 | 0 of 128 | 1.07583 | 1.0140 | 0 of 128 |
| energy | both | unweighted | 1.06097 | 1.06629 | 1.0050 | 0 of 128 | 1.07633 | 1.0145 | 0 of 128 |
| energy | both | count-weighted | 1.06099 | 1.06642 | 1.0051 | 0 of 128 | 1.07555 | 1.0137 | 0 of 128 |
| energy | both | trace-weighted | 1.06101 | 1.06644 | 1.0051 | 0 of 128 | 1.07559 | 1.0137 | 0 of 128 |
| ogain | gate | unweighted | 0.99843 | 0.99911 | 1.0007 | 48 of 128 | 1.01273 | 1.0143 | 0 of 128 |
| ogain | gate | count-weighted | 0.99814 | 0.99905 | 1.0009 | 48 of 128 | 1.01213 | 1.0140 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99810 | 0.99906 | 1.0010 | 48 of 128 | 1.01210 | 1.0140 | 0 of 128 |
| ogain | up | unweighted | 0.99575 | 0.99862 | 1.0029 | 0 of 128 | 1.00877 | 1.0131 | 0 of 128 |
| ogain | up | count-weighted | 0.99568 | 0.99861 | 1.0029 | 0 of 128 | 1.00821 | 1.0126 | 0 of 128 |
| ogain | up | trace-weighted | 0.99564 | 0.99861 | 1.0030 | 0 of 128 | 1.00816 | 1.0126 | 0 of 128 |
| ogain | both | unweighted | 0.99730 | 0.99891 | 1.0016 | 20 of 128 | 1.01108 | 1.0138 | 0 of 128 |
| ogain | both | count-weighted | 0.99709 | 0.99887 | 1.0018 | 20 of 128 | 1.01047 | 1.0134 | 0 of 128 |
| ogain | both | trace-weighted | 0.99705 | 0.99887 | 1.0018 | 20 of 128 | 1.01043 | 1.0134 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.0994e-02 | [-1.1414e-02, -1.0570e-02] |
| up | -1.4802e-02 | [-1.5081e-02, -1.4525e-02] |
| both | -1.2558e-02 | [-1.2913e-02, -1.2200e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.3748e-02 | [-1.4243e-02, -1.3243e-02] | -2.7543e-03 | [-2.9791e-03, -2.5442e-03] |
| full | up | -1.9475e-02 | [-1.9821e-02, -1.9152e-02] | -4.6723e-03 | [-4.9492e-03, -4.4110e-03] |
| full | both | -1.6108e-02 | [-1.6515e-02, -1.5692e-02] | -3.5494e-03 | [-3.7866e-03, -3.3260e-03] |
| reg | gate | -1.2129e-02 | [-1.2550e-02, -1.1697e-02] | -1.8689e-03 | [-2.0635e-03, -1.6865e-03] |
| reg | up | -1.6980e-02 | [-1.7267e-02, -1.6712e-02] | -3.3653e-03 | [-3.6060e-03, -3.1377e-03] |
| reg | both | -1.4148e-02 | [-1.4490e-02, -1.3798e-02] | -2.4966e-03 | [-2.7027e-03, -2.3008e-03] |
| bias | gate | -4.0510e-03 | [-4.2082e-03, -3.8926e-03] | -1.2128e-04 | [-1.7784e-04, -6.3938e-05] |
| bias | up | -6.0757e-03 | [-6.2173e-03, -5.9437e-03] | -6.9101e-04 | [-7.3716e-04, -6.4313e-04] |
| bias | both | -4.8833e-03 | [-5.0165e-03, -4.7544e-03] | -3.5810e-04 | [-4.0721e-04, -3.0578e-04] |
| rest | gate | -9.6970e-03 | [-1.0050e-02, -9.3352e-03] | -2.6330e-03 | [-2.8219e-03, -2.4621e-03] |
| rest | up | -1.3399e-02 | [-1.3652e-02, -1.3158e-02] | -3.9813e-03 | [-4.2251e-03, -3.7568e-03] |
| rest | both | -1.1224e-02 | [-1.1530e-02, -1.0914e-02] | -3.1913e-03 | [-3.3962e-03, -3.0047e-03] |
| diag | gate | +2.0059e-02 | [+1.9719e-02, +2.0425e-02] | +1.5867e-02 | [+1.5526e-02, +1.6235e-02] |
| diag | up | +2.0483e-02 | [+2.0114e-02, +2.0879e-02] | +1.6282e-02 | [+1.5914e-02, +1.6677e-02] |
| diag | both | +2.0271e-02 | [+1.9917e-02, +2.0649e-02] | +1.6074e-02 | [+1.5722e-02, +1.6454e-02] |

## Layer 7 (IQ2_S; UD `.scale` present, min 3.885e-05, max 9.079e-05)

- Gram layout: expert 65: max |strict lower| = 0, max |strict upper| = 4.31e+05
- Gram layout: expert 71: max |strict lower| = 0, max |strict upper| = 3.538e+05
- Gram layout: expert 43: max |strict lower| = 0, max |strict upper| = 3.526e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 1.39e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.7.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.832467 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 5087, median 35300, max 127914

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03351 | 0.02278 | 0.6796 | 128 of 128 | 0.01795 | 0.5355 | 128 of 128 |
| full | gate | count-weighted | 0.03395 | 0.02306 | 0.6791 | 128 of 128 | 0.01845 | 0.5436 | 128 of 128 |
| full | gate | trace-weighted | 0.03395 | 0.02307 | 0.6795 | 128 of 128 | 0.01848 | 0.5443 | 128 of 128 |
| full | up | unweighted | 0.04198 | 0.02857 | 0.6807 | 128 of 128 | 0.02192 | 0.5221 | 128 of 128 |
| full | up | count-weighted | 0.04205 | 0.02862 | 0.6805 | 128 of 128 | 0.02235 | 0.5314 | 128 of 128 |
| full | up | trace-weighted | 0.04209 | 0.02866 | 0.6809 | 128 of 128 | 0.02239 | 0.5319 | 128 of 128 |
| full | both | unweighted | 0.03719 | 0.02529 | 0.6802 | 128 of 128 | 0.01967 | 0.5290 | 128 of 128 |
| full | both | count-weighted | 0.03751 | 0.02550 | 0.6798 | 128 of 128 | 0.02016 | 0.5376 | 128 of 128 |
| full | both | trace-weighted | 0.03752 | 0.02552 | 0.6802 | 128 of 128 | 0.02019 | 0.5382 | 128 of 128 |
| reg | gate | unweighted | 0.03495 | 0.02499 | 0.7151 | 128 of 128 | 0.02127 | 0.6086 | 128 of 128 |
| reg | gate | count-weighted | 0.03538 | 0.02529 | 0.7146 | 128 of 128 | 0.02175 | 0.6147 | 128 of 128 |
| reg | gate | trace-weighted | 0.03538 | 0.02530 | 0.7150 | 128 of 128 | 0.02177 | 0.6153 | 128 of 128 |
| reg | up | unweighted | 0.04322 | 0.03095 | 0.7161 | 128 of 128 | 0.02580 | 0.5969 | 128 of 128 |
| reg | up | count-weighted | 0.04330 | 0.03100 | 0.7159 | 128 of 128 | 0.02615 | 0.6040 | 128 of 128 |
| reg | up | trace-weighted | 0.04333 | 0.03104 | 0.7162 | 128 of 128 | 0.02619 | 0.6044 | 128 of 128 |
| reg | both | unweighted | 0.03857 | 0.02760 | 0.7156 | 128 of 128 | 0.02325 | 0.6029 | 128 of 128 |
| reg | both | count-weighted | 0.03889 | 0.02781 | 0.7153 | 128 of 128 | 0.02370 | 0.6094 | 128 of 128 |
| reg | both | trace-weighted | 0.03890 | 0.02784 | 0.7156 | 128 of 128 | 0.02373 | 0.6100 | 128 of 128 |
| bias | gate | unweighted | 0.00438 | 9.208e-04 | 0.2103 | 128 of 128 | 3.923e-04 | 0.0896 | 128 of 128 |
| bias | gate | count-weighted | 0.00442 | 9.239e-04 | 0.2092 | 128 of 128 | 3.802e-04 | 0.0861 | 128 of 128 |
| bias | gate | trace-weighted | 0.00440 | 9.164e-04 | 0.2081 | 128 of 128 | 3.806e-04 | 0.0864 | 128 of 128 |
| bias | up | unweighted | 0.00545 | 0.00116 | 0.2135 | 128 of 128 | 2.071e-04 | 0.0380 | 128 of 128 |
| bias | up | count-weighted | 0.00543 | 0.00115 | 0.2119 | 128 of 128 | 1.973e-04 | 0.0363 | 128 of 128 |
| bias | up | trace-weighted | 0.00542 | 0.00114 | 0.2109 | 128 of 128 | 1.954e-04 | 0.0361 | 128 of 128 |
| bias | both | unweighted | 0.00484 | 0.00103 | 0.2119 | 128 of 128 | 3.114e-04 | 0.0643 | 128 of 128 |
| bias | both | count-weighted | 0.00486 | 0.00102 | 0.2106 | 128 of 128 | 2.996e-04 | 0.0617 | 128 of 128 |
| bias | both | trace-weighted | 0.00485 | 0.00102 | 0.2095 | 128 of 128 | 2.991e-04 | 0.0617 | 128 of 128 |
| rest | gate | unweighted | 0.02913 | 0.02185 | 0.7501 | 128 of 128 | 0.01756 | 0.6025 | 128 of 128 |
| rest | gate | count-weighted | 0.02953 | 0.02213 | 0.7494 | 128 of 128 | 0.01807 | 0.6120 | 128 of 128 |
| rest | gate | trace-weighted | 0.02955 | 0.02215 | 0.7497 | 128 of 128 | 0.01810 | 0.6125 | 128 of 128 |
| rest | up | unweighted | 0.03653 | 0.02741 | 0.7504 | 128 of 128 | 0.02171 | 0.5944 | 128 of 128 |
| rest | up | count-weighted | 0.03662 | 0.02747 | 0.7499 | 128 of 128 | 0.02215 | 0.6047 | 128 of 128 |
| rest | up | trace-weighted | 0.03667 | 0.02751 | 0.7503 | 128 of 128 | 0.02219 | 0.6052 | 128 of 128 |
| rest | both | unweighted | 0.03234 | 0.02427 | 0.7503 | 128 of 128 | 0.01936 | 0.5986 | 128 of 128 |
| rest | both | count-weighted | 0.03265 | 0.02447 | 0.7497 | 128 of 128 | 0.01986 | 0.6084 | 128 of 128 |
| rest | both | trace-weighted | 0.03267 | 0.02451 | 0.7500 | 128 of 128 | 0.01989 | 0.6089 | 128 of 128 |
| diag | gate | unweighted | 0.06164 | 0.06593 | 1.0696 | 0 of 128 | 0.08255 | 1.3392 | 0 of 128 |
| diag | gate | count-weighted | 0.06167 | 0.06596 | 1.0696 | 0 of 128 | 0.08176 | 1.3257 | 0 of 128 |
| diag | gate | trace-weighted | 0.06167 | 0.06596 | 1.0696 | 0 of 128 | 0.08171 | 1.3250 | 0 of 128 |
| diag | up | unweighted | 0.06170 | 0.06600 | 1.0696 | 0 of 128 | 0.08284 | 1.3426 | 0 of 128 |
| diag | up | count-weighted | 0.06173 | 0.06603 | 1.0696 | 0 of 128 | 0.08203 | 1.3288 | 0 of 128 |
| diag | up | trace-weighted | 0.06173 | 0.06602 | 1.0695 | 0 of 128 | 0.08198 | 1.3281 | 0 of 128 |
| diag | both | unweighted | 0.06167 | 0.06596 | 1.0696 | 0 of 128 | 0.08269 | 1.3409 | 0 of 128 |
| diag | both | count-weighted | 0.06170 | 0.06599 | 1.0696 | 0 of 128 | 0.08189 | 1.3273 | 0 of 128 |
| diag | both | trace-weighted | 0.06170 | 0.06599 | 1.0696 | 0 of 128 | 0.08184 | 1.3265 | 0 of 128 |
| plain | gate | unweighted | 0.26275 | 0.26964 | 1.0262 | 0 of 128 | 0.32578 | 1.2399 | 0 of 128 |
| plain | gate | count-weighted | 0.26273 | 0.26964 | 1.0263 | 0 of 128 | 0.32404 | 1.2334 | 0 of 128 |
| plain | gate | trace-weighted | 0.26275 | 0.26966 | 1.0263 | 0 of 128 | 0.32401 | 1.2331 | 0 of 128 |
| plain | up | unweighted | 0.26279 | 0.26967 | 1.0262 | 0 of 128 | 0.32674 | 1.2434 | 0 of 128 |
| plain | up | count-weighted | 0.26279 | 0.26969 | 1.0263 | 0 of 128 | 0.32497 | 1.2366 | 0 of 128 |
| plain | up | trace-weighted | 0.26281 | 0.26971 | 1.0263 | 0 of 128 | 0.32495 | 1.2365 | 0 of 128 |
| plain | both | unweighted | 0.26277 | 0.26966 | 1.0262 | 0 of 128 | 0.32626 | 1.2416 | 0 of 128 |
| plain | both | count-weighted | 0.26276 | 0.26966 | 1.0263 | 0 of 128 | 0.32451 | 1.2350 | 0 of 128 |
| plain | both | trace-weighted | 0.26278 | 0.26968 | 1.0263 | 0 of 128 | 0.32448 | 1.2348 | 0 of 128 |
| gain | gate | unweighted | 0.99719 | 0.99831 | 1.0011 | 2 of 128 | 0.98724 | 0.9900 | 128 of 128 |
| gain | gate | count-weighted | 0.99721 | 0.99836 | 1.0011 | 2 of 128 | 0.98758 | 0.9903 | 128 of 128 |
| gain | gate | trace-weighted | 0.99722 | 0.99836 | 1.0011 | 2 of 128 | 0.98760 | 0.9904 | 128 of 128 |
| gain | up | unweighted | 0.99721 | 0.99834 | 1.0011 | 3 of 128 | 0.98745 | 0.9902 | 128 of 128 |
| gain | up | count-weighted | 0.99723 | 0.99839 | 1.0012 | 3 of 128 | 0.98779 | 0.9905 | 128 of 128 |
| gain | up | trace-weighted | 0.99724 | 0.99840 | 1.0012 | 3 of 128 | 0.98781 | 0.9905 | 128 of 128 |
| gain | both | unweighted | 0.99720 | 0.99832 | 1.0011 | 2 of 128 | 0.98734 | 0.9901 | 128 of 128 |
| gain | both | count-weighted | 0.99722 | 0.99837 | 1.0012 | 2 of 128 | 0.98768 | 0.9904 | 128 of 128 |
| gain | both | trace-weighted | 0.99723 | 0.99838 | 1.0011 | 2 of 128 | 0.98770 | 0.9904 | 128 of 128 |
| energy | gate | unweighted | 1.06341 | 1.06932 | 1.0056 | 0 of 128 | 1.08063 | 1.0162 | 0 of 128 |
| energy | gate | count-weighted | 1.06346 | 1.06942 | 1.0056 | 0 of 128 | 1.08018 | 1.0157 | 0 of 128 |
| energy | gate | trace-weighted | 1.06348 | 1.06944 | 1.0056 | 0 of 128 | 1.08020 | 1.0157 | 0 of 128 |
| energy | up | unweighted | 1.06348 | 1.06941 | 1.0056 | 0 of 128 | 1.08168 | 1.0171 | 0 of 128 |
| energy | up | count-weighted | 1.06352 | 1.06951 | 1.0056 | 0 of 128 | 1.08120 | 1.0166 | 0 of 128 |
| energy | up | trace-weighted | 1.06355 | 1.06954 | 1.0056 | 0 of 128 | 1.08122 | 1.0166 | 0 of 128 |
| energy | both | unweighted | 1.06345 | 1.06936 | 1.0056 | 0 of 128 | 1.08115 | 1.0167 | 0 of 128 |
| energy | both | count-weighted | 1.06349 | 1.06946 | 1.0056 | 0 of 128 | 1.08069 | 1.0162 | 0 of 128 |
| energy | both | trace-weighted | 1.06352 | 1.06949 | 1.0056 | 0 of 128 | 1.08071 | 1.0162 | 0 of 128 |
| ogain | gate | unweighted | 0.99440 | 0.99915 | 1.0048 | 1 of 128 | 1.00790 | 1.0136 | 0 of 128 |
| ogain | gate | count-weighted | 0.99439 | 0.99917 | 1.0048 | 1 of 128 | 1.00764 | 1.0133 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99443 | 0.99916 | 1.0048 | 1 of 128 | 1.00765 | 1.0133 | 0 of 128 |
| ogain | up | unweighted | 0.99615 | 0.99893 | 1.0028 | 0 of 128 | 1.00576 | 1.0096 | 0 of 128 |
| ogain | up | count-weighted | 0.99622 | 0.99894 | 1.0027 | 0 of 128 | 1.00555 | 1.0094 | 0 of 128 |
| ogain | up | trace-weighted | 0.99623 | 0.99894 | 1.0027 | 0 of 128 | 1.00552 | 1.0093 | 0 of 128 |
| ogain | both | unweighted | 0.99517 | 0.99905 | 1.0039 | 0 of 128 | 1.00696 | 1.0118 | 0 of 128 |
| ogain | both | count-weighted | 0.99521 | 0.99906 | 1.0039 | 0 of 128 | 1.00672 | 1.0116 | 0 of 128 |
| ogain | both | trace-weighted | 0.99523 | 0.99906 | 1.0038 | 0 of 128 | 1.00671 | 1.0115 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.0893e-02 | [-1.1075e-02, -1.0701e-02] |
| up | -1.3436e-02 | [-1.3589e-02, -1.3284e-02] |
| both | -1.2009e-02 | [-1.2161e-02, -1.1853e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.5496e-02 | [-1.5795e-02, -1.5196e-02] | -4.6031e-03 | [-4.8099e-03, -4.3983e-03] |
| full | up | -1.9707e-02 | [-1.9963e-02, -1.9465e-02] | -6.2707e-03 | [-6.5258e-03, -6.0289e-03] |
| full | both | -1.7345e-02 | [-1.7600e-02, -1.7092e-02] | -5.3356e-03 | [-5.5619e-03, -5.1196e-03] |
| reg | gate | -1.3633e-02 | [-1.3886e-02, -1.3377e-02] | -3.5358e-03 | [-3.7155e-03, -3.3607e-03] |
| reg | up | -1.7146e-02 | [-1.7359e-02, -1.6945e-02] | -4.8450e-03 | [-5.0660e-03, -4.6351e-03] |
| reg | both | -1.5187e-02 | [-1.5400e-02, -1.4975e-02] | -4.1154e-03 | [-4.3105e-03, -3.9269e-03] |
| bias | gate | -4.0356e-03 | [-4.1498e-03, -3.9225e-03] | -5.4377e-04 | [-5.9356e-04, -4.9531e-04] |
| bias | up | -5.2305e-03 | [-5.3288e-03, -5.1322e-03] | -9.5301e-04 | [-1.0017e-03, -9.0651e-04] |
| bias | both | -4.5595e-03 | [-4.6582e-03, -4.4635e-03] | -7.2368e-04 | [-7.7194e-04, -6.7794e-04] |
| rest | gate | -1.1461e-02 | [-1.1662e-02, -1.1259e-02] | -4.0593e-03 | [-4.2365e-03, -3.8878e-03] |
| rest | up | -1.4477e-02 | [-1.4668e-02, -1.4299e-02] | -5.3177e-03 | [-5.5384e-03, -5.1067e-03] |
| rest | both | -1.2785e-02 | [-1.2964e-02, -1.2612e-02] | -4.6119e-03 | [-4.8067e-03, -4.4259e-03] |
| diag | gate | +2.0089e-02 | [+1.9773e-02, +2.0417e-02] | +1.5797e-02 | [+1.5481e-02, +1.6126e-02] |
| diag | up | +2.0297e-02 | [+1.9991e-02, +2.0613e-02] | +1.6002e-02 | [+1.5696e-02, +1.6321e-02] |
| diag | both | +2.0193e-02 | [+1.9883e-02, +2.0514e-02] | +1.5900e-02 | [+1.5588e-02, +1.6224e-02] |

## Layer 8 (IQ2_S; UD `.scale` present, min 3.905e-05, max 0.0001062)

- Gram layout: expert 87: max |strict lower| = 0, max |strict upper| = 5.4e+05
- Gram layout: expert 92: max |strict lower| = 0, max |strict upper| = 4.926e+05
- Gram layout: expert 66: max |strict lower| = 0, max |strict upper| = 4.41e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 5.46e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.8.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.828751 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 5405, median 34173, max 127559

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03371 | 0.02430 | 0.7208 | 128 of 128 | 0.01873 | 0.5557 | 128 of 128 |
| full | gate | count-weighted | 0.03419 | 0.02462 | 0.7201 | 128 of 128 | 0.01934 | 0.5655 | 128 of 128 |
| full | gate | trace-weighted | 0.03421 | 0.02465 | 0.7206 | 128 of 128 | 0.01937 | 0.5662 | 128 of 128 |
| full | up | unweighted | 0.04459 | 0.03214 | 0.7208 | 128 of 128 | 0.02388 | 0.5355 | 128 of 128 |
| full | up | count-weighted | 0.04482 | 0.03232 | 0.7211 | 128 of 128 | 0.02452 | 0.5470 | 128 of 128 |
| full | up | trace-weighted | 0.04493 | 0.03242 | 0.7215 | 128 of 128 | 0.02459 | 0.5474 | 128 of 128 |
| full | both | unweighted | 0.03826 | 0.02758 | 0.7208 | 128 of 128 | 0.02089 | 0.5458 | 128 of 128 |
| full | both | count-weighted | 0.03868 | 0.02787 | 0.7206 | 128 of 128 | 0.02152 | 0.5564 | 128 of 128 |
| full | both | trace-weighted | 0.03874 | 0.02793 | 0.7211 | 128 of 128 | 0.02157 | 0.5569 | 128 of 128 |
| reg | gate | unweighted | 0.03514 | 0.02641 | 0.7516 | 128 of 128 | 0.02195 | 0.6248 | 128 of 128 |
| reg | gate | count-weighted | 0.03562 | 0.02675 | 0.7510 | 128 of 128 | 0.02252 | 0.6323 | 128 of 128 |
| reg | gate | trace-weighted | 0.03564 | 0.02678 | 0.7514 | 128 of 128 | 0.02255 | 0.6329 | 128 of 128 |
| reg | up | unweighted | 0.04574 | 0.03437 | 0.7516 | 128 of 128 | 0.02776 | 0.6069 | 128 of 128 |
| reg | up | count-weighted | 0.04596 | 0.03455 | 0.7519 | 128 of 128 | 0.02830 | 0.6159 | 128 of 128 |
| reg | up | trace-weighted | 0.04606 | 0.03465 | 0.7523 | 128 of 128 | 0.02838 | 0.6162 | 128 of 128 |
| reg | both | unweighted | 0.03962 | 0.02978 | 0.7516 | 128 of 128 | 0.02441 | 0.6161 | 128 of 128 |
| reg | both | count-weighted | 0.04003 | 0.03008 | 0.7514 | 128 of 128 | 0.02499 | 0.6242 | 128 of 128 |
| reg | both | trace-weighted | 0.04008 | 0.03014 | 0.7519 | 128 of 128 | 0.02504 | 0.6246 | 128 of 128 |
| bias | gate | unweighted | 0.00438 | 0.00108 | 0.2461 | 128 of 128 | 4.534e-04 | 0.1036 | 128 of 128 |
| bias | gate | count-weighted | 0.00446 | 0.00109 | 0.2442 | 128 of 128 | 4.567e-04 | 0.1025 | 128 of 128 |
| bias | gate | trace-weighted | 0.00445 | 0.00108 | 0.2434 | 128 of 128 | 4.587e-04 | 0.1030 | 128 of 128 |
| bias | up | unweighted | 0.00578 | 0.00143 | 0.2477 | 128 of 128 | 1.915e-04 | 0.0332 | 128 of 128 |
| bias | up | count-weighted | 0.00578 | 0.00143 | 0.2475 | 128 of 128 | 1.913e-04 | 0.0331 | 128 of 128 |
| bias | up | trace-weighted | 0.00579 | 0.00143 | 0.2466 | 128 of 128 | 1.894e-04 | 0.0327 | 128 of 128 |
| bias | both | unweighted | 0.00496 | 0.00122 | 0.2469 | 128 of 128 | 3.427e-04 | 0.0691 | 128 of 128 |
| bias | both | count-weighted | 0.00502 | 0.00123 | 0.2458 | 128 of 128 | 3.439e-04 | 0.0685 | 128 of 128 |
| bias | both | trace-weighted | 0.00502 | 0.00123 | 0.2450 | 128 of 128 | 3.443e-04 | 0.0686 | 128 of 128 |
| rest | gate | unweighted | 0.02933 | 0.02322 | 0.7917 | 128 of 128 | 0.01828 | 0.6231 | 128 of 128 |
| rest | gate | count-weighted | 0.02974 | 0.02353 | 0.7914 | 128 of 128 | 0.01888 | 0.6349 | 128 of 128 |
| rest | gate | trace-weighted | 0.02976 | 0.02357 | 0.7920 | 128 of 128 | 0.01891 | 0.6355 | 128 of 128 |
| rest | up | unweighted | 0.03881 | 0.03071 | 0.7912 | 128 of 128 | 0.02368 | 0.6102 | 128 of 128 |
| rest | up | count-weighted | 0.03904 | 0.03089 | 0.7912 | 128 of 128 | 0.02432 | 0.6231 | 128 of 128 |
| rest | up | trace-weighted | 0.03914 | 0.03099 | 0.7918 | 128 of 128 | 0.02440 | 0.6235 | 128 of 128 |
| rest | both | unweighted | 0.03330 | 0.02636 | 0.7914 | 128 of 128 | 0.02054 | 0.6169 | 128 of 128 |
| rest | both | count-weighted | 0.03367 | 0.02664 | 0.7913 | 128 of 128 | 0.02118 | 0.6291 | 128 of 128 |
| rest | both | trace-weighted | 0.03372 | 0.02670 | 0.7919 | 128 of 128 | 0.02123 | 0.6296 | 128 of 128 |
| diag | gate | unweighted | 0.06168 | 0.06535 | 1.0595 | 0 of 128 | 0.08117 | 1.3159 | 0 of 128 |
| diag | gate | count-weighted | 0.06169 | 0.06536 | 1.0596 | 0 of 128 | 0.08017 | 1.2997 | 0 of 128 |
| diag | gate | trace-weighted | 0.06168 | 0.06535 | 1.0595 | 0 of 128 | 0.08012 | 1.2989 | 0 of 128 |
| diag | up | unweighted | 0.06178 | 0.06545 | 1.0594 | 0 of 128 | 0.08156 | 1.3202 | 0 of 128 |
| diag | up | count-weighted | 0.06177 | 0.06544 | 1.0594 | 0 of 128 | 0.08051 | 1.3035 | 0 of 128 |
| diag | up | trace-weighted | 0.06176 | 0.06543 | 1.0594 | 0 of 128 | 0.08045 | 1.3027 | 0 of 128 |
| diag | both | unweighted | 0.06173 | 0.06540 | 1.0595 | 0 of 128 | 0.08136 | 1.3181 | 0 of 128 |
| diag | both | count-weighted | 0.06173 | 0.06540 | 1.0595 | 0 of 128 | 0.08034 | 1.3016 | 0 of 128 |
| diag | both | trace-weighted | 0.06172 | 0.06539 | 1.0595 | 0 of 128 | 0.08028 | 1.3008 | 0 of 128 |
| plain | gate | unweighted | 0.26225 | 0.26812 | 1.0224 | 0 of 128 | 0.31723 | 1.2096 | 0 of 128 |
| plain | gate | count-weighted | 0.26225 | 0.26814 | 1.0224 | 0 of 128 | 0.31523 | 1.2020 | 0 of 128 |
| plain | gate | trace-weighted | 0.26228 | 0.26817 | 1.0224 | 0 of 128 | 0.31519 | 1.2018 | 0 of 128 |
| plain | up | unweighted | 0.26230 | 0.26817 | 1.0224 | 0 of 128 | 0.31841 | 1.2139 | 0 of 128 |
| plain | up | count-weighted | 0.26231 | 0.26819 | 1.0224 | 0 of 128 | 0.31634 | 1.2060 | 0 of 128 |
| plain | up | trace-weighted | 0.26233 | 0.26821 | 1.0224 | 0 of 128 | 0.31632 | 1.2058 | 0 of 128 |
| plain | both | unweighted | 0.26227 | 0.26814 | 1.0224 | 0 of 128 | 0.31782 | 1.2118 | 0 of 128 |
| plain | both | count-weighted | 0.26228 | 0.26816 | 1.0224 | 0 of 128 | 0.31578 | 1.2040 | 0 of 128 |
| plain | both | trace-weighted | 0.26230 | 0.26819 | 1.0224 | 0 of 128 | 0.31576 | 1.2038 | 0 of 128 |
| gain | gate | unweighted | 0.99745 | 0.99838 | 1.0009 | 5 of 128 | 0.98877 | 0.9913 | 128 of 128 |
| gain | gate | count-weighted | 0.99749 | 0.99846 | 1.0010 | 5 of 128 | 0.98917 | 0.9917 | 128 of 128 |
| gain | gate | trace-weighted | 0.99751 | 0.99847 | 1.0010 | 5 of 128 | 0.98921 | 0.9917 | 128 of 128 |
| gain | up | unweighted | 0.99747 | 0.99842 | 1.0010 | 3 of 128 | 0.98884 | 0.9914 | 128 of 128 |
| gain | up | count-weighted | 0.99752 | 0.99851 | 1.0010 | 3 of 128 | 0.98928 | 0.9917 | 128 of 128 |
| gain | up | trace-weighted | 0.99753 | 0.99852 | 1.0010 | 3 of 128 | 0.98931 | 0.9918 | 128 of 128 |
| gain | both | unweighted | 0.99746 | 0.99840 | 1.0009 | 5 of 128 | 0.98880 | 0.9913 | 128 of 128 |
| gain | both | count-weighted | 0.99750 | 0.99848 | 1.0010 | 5 of 128 | 0.98923 | 0.9917 | 128 of 128 |
| gain | both | trace-weighted | 0.99752 | 0.99849 | 1.0010 | 5 of 128 | 0.98926 | 0.9917 | 128 of 128 |
| energy | gate | unweighted | 1.06367 | 1.06864 | 1.0047 | 0 of 128 | 1.07819 | 1.0137 | 0 of 128 |
| energy | gate | count-weighted | 1.06376 | 1.06882 | 1.0048 | 0 of 128 | 1.07773 | 1.0131 | 0 of 128 |
| energy | gate | trace-weighted | 1.06380 | 1.06885 | 1.0047 | 0 of 128 | 1.07778 | 1.0131 | 0 of 128 |
| energy | up | unweighted | 1.06373 | 1.06875 | 1.0047 | 0 of 128 | 1.07909 | 1.0144 | 0 of 128 |
| energy | up | count-weighted | 1.06384 | 1.06893 | 1.0048 | 0 of 128 | 1.07865 | 1.0139 | 0 of 128 |
| energy | up | trace-weighted | 1.06388 | 1.06897 | 1.0048 | 0 of 128 | 1.07869 | 1.0139 | 0 of 128 |
| energy | both | unweighted | 1.06370 | 1.06870 | 1.0047 | 0 of 128 | 1.07864 | 1.0140 | 0 of 128 |
| energy | both | count-weighted | 1.06380 | 1.06888 | 1.0048 | 0 of 128 | 1.07819 | 1.0135 | 0 of 128 |
| energy | both | trace-weighted | 1.06384 | 1.06891 | 1.0048 | 0 of 128 | 1.07823 | 1.0135 | 0 of 128 |
| ogain | gate | unweighted | 0.99489 | 0.99906 | 1.0042 | 3 of 128 | 1.00862 | 1.0138 | 0 of 128 |
| ogain | gate | count-weighted | 0.99490 | 0.99909 | 1.0042 | 3 of 128 | 1.00856 | 1.0137 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99493 | 0.99909 | 1.0042 | 3 of 128 | 1.00859 | 1.0137 | 0 of 128 |
| ogain | up | unweighted | 0.99620 | 0.99877 | 1.0026 | 1 of 128 | 1.00534 | 1.0092 | 0 of 128 |
| ogain | up | count-weighted | 0.99637 | 0.99887 | 1.0025 | 1 of 128 | 1.00531 | 1.0090 | 0 of 128 |
| ogain | up | trace-weighted | 0.99639 | 0.99887 | 1.0025 | 1 of 128 | 1.00528 | 1.0089 | 0 of 128 |
| ogain | both | unweighted | 0.99545 | 0.99894 | 1.0035 | 0 of 128 | 1.00722 | 1.0118 | 0 of 128 |
| ogain | both | count-weighted | 0.99553 | 0.99899 | 1.0035 | 0 of 128 | 1.00716 | 1.0117 | 0 of 128 |
| ogain | both | trace-weighted | 0.99555 | 0.99900 | 1.0035 | 0 of 128 | 1.00717 | 1.0117 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -9.5702e-03 | [-9.7704e-03, -9.3671e-03] |
| up | -1.2502e-02 | [-1.2687e-02, -1.2330e-02] |
| both | -1.0810e-02 | [-1.0981e-02, -1.0646e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.4857e-02 | [-1.5187e-02, -1.4524e-02] | -5.2868e-03 | [-5.5221e-03, -5.0607e-03] |
| full | up | -2.0305e-02 | [-2.0576e-02, -2.0035e-02] | -7.8029e-03 | [-8.0693e-03, -7.5564e-03] |
| full | both | -1.7161e-02 | [-1.7429e-02, -1.6900e-02] | -6.3513e-03 | [-6.5900e-03, -6.1258e-03] |
| reg | gate | -1.3098e-02 | [-1.3379e-02, -1.2816e-02] | -4.2276e-03 | [-4.4294e-03, -4.0334e-03] |
| reg | up | -1.7654e-02 | [-1.7876e-02, -1.7429e-02] | -6.2503e-03 | [-6.4803e-03, -6.0357e-03] |
| reg | both | -1.5045e-02 | [-1.5267e-02, -1.4827e-02] | -5.0924e-03 | [-5.2978e-03, -4.8985e-03] |
| bias | gate | -3.9995e-03 | [-4.1061e-03, -3.8903e-03] | -6.3155e-04 | [-6.8437e-04, -5.7823e-04] |
| bias | up | -5.5920e-03 | [-5.6951e-03, -5.4859e-03] | -1.2399e-03 | [-1.2925e-03, -1.1866e-03] |
| bias | both | -4.6734e-03 | [-4.7663e-03, -4.5812e-03] | -8.8948e-04 | [-9.3937e-04, -8.3990e-04] |
| rest | gate | -1.0857e-02 | [-1.1103e-02, -1.0613e-02] | -4.6552e-03 | [-4.8555e-03, -4.4651e-03] |
| rest | up | -1.4713e-02 | [-1.4922e-02, -1.4516e-02] | -6.5630e-03 | [-6.8032e-03, -6.3421e-03] |
| rest | both | -1.2488e-02 | [-1.2690e-02, -1.2293e-02] | -5.4618e-03 | [-5.6738e-03, -5.2668e-03] |
| diag | gate | +1.8488e-02 | [+1.8172e-02, +1.8832e-02] | +1.4814e-02 | [+1.4496e-02, +1.5161e-02] |
| diag | up | +1.8745e-02 | [+1.8427e-02, +1.9096e-02] | +1.5073e-02 | [+1.4753e-02, +1.5425e-02] |
| diag | both | +1.8616e-02 | [+1.8300e-02, +1.8963e-02] | +1.4943e-02 | [+1.4624e-02, +1.5292e-02] |

## Layer 9 (IQ3_XXS; UD `.scale` present, min 3.921e-05, max 0.0001086)

- Gram layout: expert 21: max |strict lower| = 0, max |strict upper| = 4.563e+05
- Gram layout: expert 39: max |strict lower| = 0, max |strict upper| = 4.07e+05
- Gram layout: expert 86: max |strict lower| = 0, max |strict upper| = 3.123e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 8.15e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.9.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.839702 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 8164, median 33633, max 136451

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.01740 | 0.01335 | 0.7671 | 128 of 128 | 0.00997 | 0.5729 | 128 of 128 |
| full | gate | count-weighted | 0.01757 | 0.01345 | 0.7658 | 128 of 128 | 0.01021 | 0.5814 | 128 of 128 |
| full | gate | trace-weighted | 0.01757 | 0.01346 | 0.7661 | 128 of 128 | 0.01022 | 0.5819 | 128 of 128 |
| full | up | unweighted | 0.02405 | 0.01847 | 0.7678 | 128 of 128 | 0.01312 | 0.5457 | 128 of 128 |
| full | up | count-weighted | 0.02408 | 0.01845 | 0.7659 | 128 of 128 | 0.01335 | 0.5543 | 128 of 128 |
| full | up | trace-weighted | 0.02414 | 0.01849 | 0.7662 | 128 of 128 | 0.01338 | 0.5545 | 128 of 128 |
| full | both | unweighted | 0.02013 | 0.01545 | 0.7674 | 128 of 128 | 0.01126 | 0.5595 | 128 of 128 |
| full | both | count-weighted | 0.02027 | 0.01552 | 0.7658 | 128 of 128 | 0.01151 | 0.5680 | 128 of 128 |
| full | both | trace-weighted | 0.02029 | 0.01554 | 0.7661 | 128 of 128 | 0.01153 | 0.5683 | 128 of 128 |
| reg | gate | unweighted | 0.01814 | 0.01442 | 0.7950 | 128 of 128 | 0.01159 | 0.6391 | 128 of 128 |
| reg | gate | count-weighted | 0.01831 | 0.01453 | 0.7939 | 128 of 128 | 0.01182 | 0.6458 | 128 of 128 |
| reg | gate | trace-weighted | 0.01831 | 0.01454 | 0.7941 | 128 of 128 | 0.01183 | 0.6462 | 128 of 128 |
| reg | up | unweighted | 0.02459 | 0.01957 | 0.7957 | 128 of 128 | 0.01513 | 0.6151 | 128 of 128 |
| reg | up | count-weighted | 0.02463 | 0.01955 | 0.7939 | 128 of 128 | 0.01531 | 0.6219 | 128 of 128 |
| reg | up | trace-weighted | 0.02467 | 0.01960 | 0.7942 | 128 of 128 | 0.01535 | 0.6220 | 128 of 128 |
| reg | both | unweighted | 0.02082 | 0.01656 | 0.7953 | 128 of 128 | 0.01306 | 0.6273 | 128 of 128 |
| reg | both | count-weighted | 0.02096 | 0.01664 | 0.7939 | 128 of 128 | 0.01329 | 0.6340 | 128 of 128 |
| reg | both | trace-weighted | 0.02098 | 0.01666 | 0.7941 | 128 of 128 | 0.01330 | 0.6342 | 128 of 128 |
| bias | gate | unweighted | 0.00204 | 4.829e-04 | 0.2367 | 128 of 128 | 2.858e-04 | 0.1401 | 128 of 128 |
| bias | gate | count-weighted | 0.00208 | 4.966e-04 | 0.2387 | 128 of 128 | 2.899e-04 | 0.1394 | 128 of 128 |
| bias | gate | trace-weighted | 0.00208 | 4.928e-04 | 0.2371 | 128 of 128 | 2.910e-04 | 0.1400 | 128 of 128 |
| bias | up | unweighted | 0.00281 | 6.701e-04 | 0.2387 | 128 of 128 | 1.016e-04 | 0.0362 | 128 of 128 |
| bias | up | count-weighted | 0.00285 | 6.798e-04 | 0.2384 | 128 of 128 | 1.028e-04 | 0.0361 | 128 of 128 |
| bias | up | trace-weighted | 0.00285 | 6.762e-04 | 0.2371 | 128 of 128 | 1.017e-04 | 0.0357 | 128 of 128 |
| bias | both | unweighted | 0.00235 | 5.598e-04 | 0.2377 | 128 of 128 | 2.099e-04 | 0.0891 | 128 of 128 |
| bias | both | count-weighted | 0.00240 | 5.727e-04 | 0.2386 | 128 of 128 | 2.121e-04 | 0.0884 | 128 of 128 |
| bias | both | trace-weighted | 0.00240 | 5.689e-04 | 0.2371 | 128 of 128 | 2.123e-04 | 0.0885 | 128 of 128 |
| rest | gate | unweighted | 0.01536 | 0.01286 | 0.8375 | 128 of 128 | 0.00968 | 0.6303 | 128 of 128 |
| rest | gate | count-weighted | 0.01549 | 0.01296 | 0.8366 | 128 of 128 | 0.00992 | 0.6408 | 128 of 128 |
| rest | gate | trace-weighted | 0.01549 | 0.01297 | 0.8370 | 128 of 128 | 0.00993 | 0.6412 | 128 of 128 |
| rest | up | unweighted | 0.02124 | 0.01780 | 0.8377 | 128 of 128 | 0.01302 | 0.6130 | 128 of 128 |
| rest | up | count-weighted | 0.02123 | 0.01777 | 0.8367 | 128 of 128 | 0.01325 | 0.6239 | 128 of 128 |
| rest | up | trace-weighted | 0.02128 | 0.01782 | 0.8371 | 128 of 128 | 0.01328 | 0.6240 | 128 of 128 |
| rest | both | unweighted | 0.01777 | 0.01489 | 0.8376 | 128 of 128 | 0.01105 | 0.6218 | 128 of 128 |
| rest | both | count-weighted | 0.01787 | 0.01495 | 0.8366 | 128 of 128 | 0.01130 | 0.6325 | 128 of 128 |
| rest | both | trace-weighted | 0.01789 | 0.01498 | 0.8370 | 128 of 128 | 0.01132 | 0.6327 | 128 of 128 |
| diag | gate | unweighted | 0.03182 | 0.03419 | 1.0746 | 0 of 128 | 0.04140 | 1.3011 | 0 of 128 |
| diag | gate | count-weighted | 0.03182 | 0.03419 | 1.0744 | 0 of 128 | 0.04103 | 1.2894 | 0 of 128 |
| diag | gate | trace-weighted | 0.03182 | 0.03418 | 1.0743 | 0 of 128 | 0.04101 | 1.2889 | 0 of 128 |
| diag | up | unweighted | 0.03185 | 0.03423 | 1.0746 | 0 of 128 | 0.04171 | 1.3095 | 0 of 128 |
| diag | up | count-weighted | 0.03185 | 0.03422 | 1.0745 | 0 of 128 | 0.04133 | 1.2976 | 0 of 128 |
| diag | up | trace-weighted | 0.03184 | 0.03421 | 1.0744 | 0 of 128 | 0.04131 | 1.2972 | 0 of 128 |
| diag | both | unweighted | 0.03184 | 0.03421 | 1.0746 | 0 of 128 | 0.04156 | 1.3053 | 0 of 128 |
| diag | both | count-weighted | 0.03184 | 0.03421 | 1.0745 | 0 of 128 | 0.04118 | 1.2935 | 0 of 128 |
| diag | both | trace-weighted | 0.03183 | 0.03420 | 1.0744 | 0 of 128 | 0.04116 | 1.2931 | 0 of 128 |
| plain | gate | unweighted | 0.18715 | 0.19324 | 1.0326 | 0 of 128 | 0.22018 | 1.1765 | 0 of 128 |
| plain | gate | count-weighted | 0.18713 | 0.19322 | 1.0325 | 0 of 128 | 0.21916 | 1.1712 | 0 of 128 |
| plain | gate | trace-weighted | 0.18714 | 0.19323 | 1.0325 | 0 of 128 | 0.21916 | 1.1711 | 0 of 128 |
| plain | up | unweighted | 0.18712 | 0.19321 | 1.0325 | 0 of 128 | 0.22102 | 1.1811 | 0 of 128 |
| plain | up | count-weighted | 0.18711 | 0.19319 | 1.0325 | 0 of 128 | 0.21998 | 1.1757 | 0 of 128 |
| plain | up | trace-weighted | 0.18712 | 0.19321 | 1.0325 | 0 of 128 | 0.21999 | 1.1757 | 0 of 128 |
| plain | both | unweighted | 0.18713 | 0.19322 | 1.0325 | 0 of 128 | 0.22060 | 1.1788 | 0 of 128 |
| plain | both | count-weighted | 0.18712 | 0.19321 | 1.0325 | 0 of 128 | 0.21957 | 1.1734 | 0 of 128 |
| plain | both | trace-weighted | 0.18713 | 0.19322 | 1.0325 | 0 of 128 | 0.21958 | 1.1734 | 0 of 128 |
| gain | gate | unweighted | 0.99618 | 0.99647 | 1.0003 | 45 of 128 | 0.99079 | 0.9946 | 128 of 128 |
| gain | gate | count-weighted | 0.99620 | 0.99644 | 1.0002 | 45 of 128 | 0.99097 | 0.9947 | 128 of 128 |
| gain | gate | trace-weighted | 0.99621 | 0.99645 | 1.0002 | 45 of 128 | 0.99098 | 0.9948 | 128 of 128 |
| gain | up | unweighted | 0.99621 | 0.99652 | 1.0003 | 36 of 128 | 0.99088 | 0.9946 | 128 of 128 |
| gain | up | count-weighted | 0.99622 | 0.99650 | 1.0003 | 36 of 128 | 0.99107 | 0.9948 | 128 of 128 |
| gain | up | trace-weighted | 0.99623 | 0.99650 | 1.0003 | 36 of 128 | 0.99108 | 0.9948 | 128 of 128 |
| gain | both | unweighted | 0.99620 | 0.99649 | 1.0003 | 41 of 128 | 0.99083 | 0.9946 | 128 of 128 |
| gain | both | count-weighted | 0.99621 | 0.99647 | 1.0003 | 41 of 128 | 0.99102 | 0.9948 | 128 of 128 |
| gain | both | trace-weighted | 0.99622 | 0.99648 | 1.0003 | 41 of 128 | 0.99103 | 0.9948 | 128 of 128 |
| energy | gate | unweighted | 1.02739 | 1.03028 | 1.0028 | 1 of 128 | 1.03006 | 1.0026 | 0 of 128 |
| energy | gate | count-weighted | 1.02742 | 1.03021 | 1.0027 | 1 of 128 | 1.02998 | 1.0025 | 0 of 128 |
| energy | gate | trace-weighted | 1.02744 | 1.03023 | 1.0027 | 1 of 128 | 1.03001 | 1.0025 | 0 of 128 |
| energy | up | unweighted | 1.02744 | 1.03036 | 1.0028 | 0 of 128 | 1.03061 | 1.0031 | 0 of 128 |
| energy | up | count-weighted | 1.02746 | 1.03032 | 1.0028 | 0 of 128 | 1.03053 | 1.0030 | 0 of 128 |
| energy | up | trace-weighted | 1.02748 | 1.03034 | 1.0028 | 0 of 128 | 1.03055 | 1.0030 | 0 of 128 |
| energy | both | unweighted | 1.02742 | 1.03032 | 1.0028 | 0 of 128 | 1.03033 | 1.0028 | 0 of 128 |
| energy | both | count-weighted | 1.02744 | 1.03027 | 1.0028 | 0 of 128 | 1.03025 | 1.0027 | 0 of 128 |
| energy | both | trace-weighted | 1.02746 | 1.03029 | 1.0027 | 0 of 128 | 1.03028 | 1.0027 | 0 of 128 |
| ogain | gate | unweighted | 0.99874 | 0.99894 | 1.0002 | 50 of 128 | 1.00719 | 1.0085 | 0 of 128 |
| ogain | gate | count-weighted | 0.99872 | 0.99889 | 1.0002 | 50 of 128 | 1.00718 | 1.0085 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99875 | 0.99890 | 1.0002 | 50 of 128 | 1.00721 | 1.0085 | 0 of 128 |
| ogain | up | unweighted | 0.99829 | 0.99856 | 1.0003 | 48 of 128 | 1.00379 | 1.0055 | 0 of 128 |
| ogain | up | count-weighted | 0.99828 | 0.99850 | 1.0002 | 48 of 128 | 1.00376 | 1.0055 | 0 of 128 |
| ogain | up | trace-weighted | 0.99828 | 0.99851 | 1.0002 | 48 of 128 | 1.00375 | 1.0055 | 0 of 128 |
| ogain | both | unweighted | 0.99856 | 0.99878 | 1.0002 | 49 of 128 | 1.00577 | 1.0072 | 0 of 128 |
| ogain | both | count-weighted | 0.99854 | 0.99873 | 1.0002 | 49 of 128 | 1.00574 | 1.0072 | 0 of 128 |
| ogain | both | trace-weighted | 0.99855 | 0.99873 | 1.0002 | 49 of 128 | 1.00575 | 1.0072 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -4.1143e-03 | [-4.2004e-03, -4.0277e-03] |
| up | -5.6377e-03 | [-5.7347e-03, -5.5493e-03] |
| both | -4.7463e-03 | [-4.8272e-03, -4.6666e-03] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -7.3541e-03 | [-7.5133e-03, -7.2060e-03] | -3.2399e-03 | [-3.3754e-03, -3.1136e-03] |
| full | up | -1.0734e-02 | [-1.0878e-02, -1.0599e-02] | -5.0960e-03 | [-5.2714e-03, -4.9293e-03] |
| full | both | -8.7555e-03 | [-8.8915e-03, -8.6352e-03] | -4.0092e-03 | [-4.1548e-03, -3.8726e-03] |
| reg | gate | -6.4846e-03 | [-6.6198e-03, -6.3586e-03] | -2.7108e-03 | [-2.8284e-03, -2.6011e-03] |
| reg | up | -9.3120e-03 | [-9.4313e-03, -9.2007e-03] | -4.2375e-03 | [-4.3902e-03, -4.0920e-03] |
| reg | both | -7.6711e-03 | [-7.7849e-03, -7.5697e-03] | -3.3512e-03 | [-3.4773e-03, -3.2316e-03] |
| bias | gate | -1.7904e-03 | [-1.8402e-03, -1.7430e-03] | -2.0670e-04 | [-2.3327e-04, -1.8266e-04] |
| bias | up | -2.7483e-03 | [-2.7997e-03, -2.6967e-03] | -5.7704e-04 | [-6.0596e-04, -5.4993e-04] |
| bias | both | -2.1879e-03 | [-2.2307e-03, -2.1463e-03] | -3.6061e-04 | [-3.8729e-04, -3.3634e-04] |
| rest | gate | -5.5638e-03 | [-5.6847e-03, -5.4520e-03] | -3.0332e-03 | [-3.1521e-03, -2.9217e-03] |
| rest | up | -7.9854e-03 | [-8.1102e-03, -7.8713e-03] | -4.5190e-03 | [-4.6794e-03, -4.3685e-03] |
| rest | both | -6.5676e-03 | [-6.6777e-03, -6.4706e-03] | -3.6486e-03 | [-3.7788e-03, -3.5254e-03] |
| diag | gate | +9.2099e-03 | [+9.0689e-03, +9.3611e-03] | +6.8419e-03 | [+6.6992e-03, +6.9948e-03] |
| diag | up | +9.4793e-03 | [+9.3411e-03, +9.6302e-03] | +7.1058e-03 | [+6.9662e-03, +7.2587e-03] |
| diag | both | +9.3445e-03 | [+9.2059e-03, +9.4955e-03] | +6.9738e-03 | [+6.8334e-03, +7.1263e-03] |

## Reading (amendment 7 rule, applied literally)

Rule: supported if UD is lower than ours on the full error and higher or equal on the diagonal one in both layers; if UD is also worse on the full error, layer-local output error does not explain UD's edge.

| mean | half | layer 6 full UD < ours | layer 6 diag UD >= ours | layer 7 full UD < ours | layer 7 diag UD >= ours | layer 8 full UD < ours | layer 8 diag UD >= ours | layer 9 full UD < ours | layer 9 diag UD >= ours | branch |
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

In layer 6 (both halves, count-weighted), UD's full error is 0.02236 against our 0.03492 (ratio 0.640; UD - ours -1.256e-02, 95% CI [-1.291e-02, -1.220e-02]) and its diagonal error 0.06595 against our 0.06176 (ratio 1.068); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 6 (added after registration, not part of the rule): the input-mean part is 0.00545 ours against 9.260e-04 UD (ratio 0.170), the centred rest 0.02946 against 0.02143 (ratio 0.727). In layer 7 (both halves, count-weighted), UD's full error is 0.02550 against our 0.03751 (ratio 0.680; UD - ours -1.201e-02, 95% CI [-1.216e-02, -1.185e-02]) and its diagonal error 0.06599 against our 0.06170 (ratio 1.070); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 7 (added after registration, not part of the rule): the input-mean part is 0.00486 ours against 0.00102 UD (ratio 0.211), the centred rest 0.03265 against 0.02447 (ratio 0.750). In layer 8 (both halves, count-weighted), UD's full error is 0.02787 against our 0.03868 (ratio 0.721; UD - ours -1.081e-02, 95% CI [-1.098e-02, -1.065e-02]) and its diagonal error 0.06540 against our 0.06173 (ratio 1.060); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 8 (added after registration, not part of the rule): the input-mean part is 0.00502 ours against 0.00123 UD (ratio 0.246), the centred rest 0.03367 against 0.02664 (ratio 0.791). In layer 9 (both halves, count-weighted), UD's full error is 0.01552 against our 0.02027 (ratio 0.766; UD - ours -4.746e-03, 95% CI [-4.827e-03, -4.667e-03]) and its diagonal error 0.03421 against our 0.03184 (ratio 1.074); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 9 (added after registration, not part of the rule): the input-mean part is 0.00240 ours against 5.727e-04 UD (ratio 0.239), the centred rest 0.01787 against 0.01495 (ratio 0.837). Under the registered rule this is "supported" for both halves together, count-weighted; across the three means and three halves the branches are: supported.

## refined against ours and UD

In layer 6 (both halves, count-weighted), refined's full error is 0.01881 against our 0.03492 and UD's 0.02236 (refined - ours -1.611e-02, 95% CI [-1.651e-02, -1.569e-02]; (ours - refined) / (ours - UD) = 1.283; lower than ours in 126 of 128 experts). Bias part 5.679e-04 (ours 0.00545, UD 9.260e-04), rest 0.01824 (ours 0.02946, UD 0.02143). Regularised (the search objective) 0.02218 (ours 0.03632, UD 0.02467). Diagonal 0.08203 (ours 0.06176, UD 0.06595), plain 0.31668 (ours 0.26135, UD 0.26810). Gains (ours / UD / refined): plain 0.9963 / 0.9973 / 0.9876, energy 1.0610 / 1.0664 / 1.0755, output 0.9971 / 0.9989 / 1.0105. In layer 7 (both halves, count-weighted), refined's full error is 0.02016 against our 0.03751 and UD's 0.02550 (refined - ours -1.734e-02, 95% CI [-1.760e-02, -1.709e-02]; (ours - refined) / (ours - UD) = 1.444; lower than ours in 128 of 128 experts). Bias part 2.996e-04 (ours 0.00486, UD 0.00102), rest 0.01986 (ours 0.03265, UD 0.02447). Regularised (the search objective) 0.02370 (ours 0.03889, UD 0.02781). Diagonal 0.08189 (ours 0.06170, UD 0.06599), plain 0.32451 (ours 0.26276, UD 0.26966). Gains (ours / UD / refined): plain 0.9972 / 0.9984 / 0.9877, energy 1.0635 / 1.0695 / 1.0807, output 0.9952 / 0.9991 / 1.0067. In layer 8 (both halves, count-weighted), refined's full error is 0.02152 against our 0.03868 and UD's 0.02787 (refined - ours -1.716e-02, 95% CI [-1.743e-02, -1.690e-02]; (ours - refined) / (ours - UD) = 1.588; lower than ours in 128 of 128 experts). Bias part 3.439e-04 (ours 0.00502, UD 0.00123), rest 0.02118 (ours 0.03367, UD 0.02664). Regularised (the search objective) 0.02499 (ours 0.04003, UD 0.03008). Diagonal 0.08034 (ours 0.06173, UD 0.06540), plain 0.31578 (ours 0.26228, UD 0.26816). Gains (ours / UD / refined): plain 0.9975 / 0.9985 / 0.9892, energy 1.0638 / 1.0689 / 1.0782, output 0.9955 / 0.9990 / 1.0072. In layer 9 (both halves, count-weighted), refined's full error is 0.01151 against our 0.02027 and UD's 0.01552 (refined - ours -8.756e-03, 95% CI [-8.891e-03, -8.635e-03]; (ours - refined) / (ours - UD) = 1.845; lower than ours in 128 of 128 experts). Bias part 2.121e-04 (ours 0.00240, UD 5.727e-04), rest 0.01130 (ours 0.01787, UD 0.01495). Regularised (the search objective) 0.01329 (ours 0.02096, UD 0.01664). Diagonal 0.04118 (ours 0.03184, UD 0.03421), plain 0.21957 (ours 0.18712, UD 0.19321). Gains (ours / UD / refined): plain 0.9962 / 0.9965 / 0.9910, energy 1.0274 / 1.0303 / 1.0303, output 0.9985 / 0.9987 / 1.0057.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined (count-w.) | refined / ours | refined / UD |
|---|---|---|---|---|---|---|---|---|
| 6 | < 2304 | 2 | 0.000 | 0.02806 | 0.01671 | 0.02806 | 1.000 | 1.679 |
| 6 | >= 2304 | 126 | 1.000 | 0.03492 | 0.02236 | 0.01880 | 0.538 | 0.841 |
| 7 | >= 2304 | 128 | 1.000 | 0.03751 | 0.02550 | 0.02016 | 0.538 | 0.791 |
| 8 | >= 2304 | 128 | 1.000 | 0.03868 | 0.02787 | 0.02152 | 0.556 | 0.772 |
| 9 | >= 2304 | 128 | 1.000 | 0.02027 | 0.01552 | 0.01151 | 0.568 | 0.742 |

Refinement log `models/refined/gu-refined-mixc-6-7-8-9.gguf.json` (scripts/refine-codes.py; wall time 6850 s for 512 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 126 | 2 | 4: 126 | 56974464 | 2610899 of 2612736 | 7800 | 9.71e-17 | 5.90e-17 | yes |
| 7 | 128 | 0 | 4: 128 | 62921659 | 2653340 of 2654208 | 6963 | 1.04e-16 | 5.38e-17 | yes |
| 8 | 128 | 0 | 4: 128 | 61596841 | 2653219 of 2654208 | 5505 | 1.11e-16 | 5.55e-17 | yes |
| 9 | 128 | 0 | 4: 128 | 79461002 | 2653794 of 2654208 | 6995 | 5.20e-17 | 3.12e-17 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 6: max abs(log - here) on reg 8.33e-17 before, 6.59e-17 after; layer 7: max abs(log - here) on reg 1.11e-16 before, 6.59e-17 after; layer 8: max abs(log - here) on reg 1.39e-16 before, 6.25e-17 after; layer 9: max abs(log - here) on reg 6.25e-17 before, 3.64e-17 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304').
