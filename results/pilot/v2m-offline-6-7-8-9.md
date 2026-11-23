# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined = `models/refined/gu-refined-mix-6-7-8-9.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/v2m-offline-6-7-8-9.tsv`.
Grams: `models/grams/gu-6-7-8-9-mix` (capture settings in its log, not restated here), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

## Layer 6 (IQ2_S; UD `.scale` present, min 3.867e-05, max 0.0001495)

- Gram layout: expert 91: max |strict lower| = 0, max |strict upper| = 2.666e+05
- Gram layout: expert 23: max |strict lower| = 0, max |strict upper| = 3.798e+05
- Gram layout: expert 73: max |strict lower| = 0, max |strict upper| = 1.519e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 1.11e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.6.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.685388 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 657, median 23087, max 142632

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03026 | 0.01911 | 0.6315 | 128 of 128 | 0.01594 | 0.5268 | 126 of 128 |
| full | gate | count-weighted | 0.03102 | 0.01966 | 0.6338 | 128 of 128 | 0.01690 | 0.5449 | 126 of 128 |
| full | gate | trace-weighted | 0.03092 | 0.01961 | 0.6341 | 128 of 128 | 0.01688 | 0.5460 | 126 of 128 |
| full | up | unweighted | 0.04118 | 0.02590 | 0.6290 | 128 of 128 | 0.02060 | 0.5004 | 126 of 128 |
| full | up | count-weighted | 0.04129 | 0.02610 | 0.6321 | 128 of 128 | 0.02139 | 0.5180 | 126 of 128 |
| full | up | trace-weighted | 0.04131 | 0.02613 | 0.6324 | 128 of 128 | 0.02144 | 0.5189 | 126 of 128 |
| full | both | unweighted | 0.03467 | 0.02186 | 0.6305 | 128 of 128 | 0.01781 | 0.5138 | 126 of 128 |
| full | both | count-weighted | 0.03525 | 0.02232 | 0.6330 | 128 of 128 | 0.01875 | 0.5318 | 126 of 128 |
| full | both | trace-weighted | 0.03519 | 0.02229 | 0.6334 | 128 of 128 | 0.01875 | 0.5328 | 126 of 128 |
| reg | gate | unweighted | 0.03167 | 0.02126 | 0.6713 | 128 of 128 | 0.01912 | 0.6037 | 126 of 128 |
| reg | gate | count-weighted | 0.03244 | 0.02185 | 0.6734 | 128 of 128 | 0.02002 | 0.6170 | 126 of 128 |
| reg | gate | trace-weighted | 0.03234 | 0.02179 | 0.6736 | 128 of 128 | 0.01999 | 0.6179 | 126 of 128 |
| reg | up | unweighted | 0.04243 | 0.02839 | 0.6690 | 128 of 128 | 0.02461 | 0.5800 | 126 of 128 |
| reg | up | count-weighted | 0.04257 | 0.02859 | 0.6717 | 128 of 128 | 0.02524 | 0.5930 | 126 of 128 |
| reg | up | trace-weighted | 0.04258 | 0.02862 | 0.6720 | 128 of 128 | 0.02528 | 0.5938 | 126 of 128 |
| reg | both | unweighted | 0.03605 | 0.02417 | 0.6703 | 128 of 128 | 0.02135 | 0.5921 | 126 of 128 |
| reg | both | count-weighted | 0.03666 | 0.02466 | 0.6727 | 128 of 128 | 0.02219 | 0.6053 | 126 of 128 |
| reg | both | trace-weighted | 0.03659 | 0.02462 | 0.6729 | 128 of 128 | 0.02218 | 0.6061 | 126 of 128 |
| bias | gate | unweighted | 0.00495 | 8.615e-04 | 0.1742 | 128 of 128 | 8.267e-04 | 0.1672 | 126 of 128 |
| bias | gate | count-weighted | 0.00482 | 7.880e-04 | 0.1636 | 128 of 128 | 7.020e-04 | 0.1458 | 126 of 128 |
| bias | gate | trace-weighted | 0.00479 | 7.781e-04 | 0.1623 | 128 of 128 | 7.022e-04 | 0.1465 | 126 of 128 |
| bias | up | unweighted | 0.00689 | 0.00119 | 0.1720 | 128 of 128 | 5.673e-04 | 0.0823 | 126 of 128 |
| bias | up | count-weighted | 0.00654 | 0.00105 | 0.1607 | 128 of 128 | 3.889e-04 | 0.0595 | 126 of 128 |
| bias | up | trace-weighted | 0.00653 | 0.00104 | 0.1596 | 128 of 128 | 3.887e-04 | 0.0595 | 126 of 128 |
| bias | both | unweighted | 0.00572 | 9.914e-04 | 0.1732 | 128 of 128 | 7.159e-04 | 0.1251 | 126 of 128 |
| bias | both | count-weighted | 0.00552 | 8.958e-04 | 0.1622 | 128 of 128 | 5.698e-04 | 0.1032 | 126 of 128 |
| bias | both | trace-weighted | 0.00550 | 8.862e-04 | 0.1610 | 128 of 128 | 5.699e-04 | 0.1035 | 126 of 128 |
| rest | gate | unweighted | 0.02532 | 0.01825 | 0.7208 | 128 of 128 | 0.01512 | 0.5971 | 126 of 128 |
| rest | gate | count-weighted | 0.02620 | 0.01887 | 0.7202 | 128 of 128 | 0.01620 | 0.6183 | 126 of 128 |
| rest | gate | trace-weighted | 0.02613 | 0.01883 | 0.7206 | 128 of 128 | 0.01618 | 0.6193 | 126 of 128 |
| rest | up | unweighted | 0.03428 | 0.02472 | 0.7209 | 128 of 128 | 0.02004 | 0.5845 | 126 of 128 |
| rest | up | count-weighted | 0.03475 | 0.02505 | 0.7208 | 128 of 128 | 0.02100 | 0.6043 | 126 of 128 |
| rest | up | trace-weighted | 0.03478 | 0.02508 | 0.7212 | 128 of 128 | 0.02105 | 0.6052 | 126 of 128 |
| rest | both | unweighted | 0.02894 | 0.02087 | 0.7210 | 128 of 128 | 0.01710 | 0.5907 | 126 of 128 |
| rest | both | count-weighted | 0.02973 | 0.02142 | 0.7205 | 128 of 128 | 0.01818 | 0.6114 | 126 of 128 |
| rest | both | trace-weighted | 0.02968 | 0.02140 | 0.7209 | 128 of 128 | 0.01818 | 0.6124 | 126 of 128 |
| diag | gate | unweighted | 0.06139 | 0.06564 | 1.0692 | 0 of 128 | 0.08390 | 1.3667 | 0 of 128 |
| diag | gate | count-weighted | 0.06174 | 0.06597 | 1.0686 | 0 of 128 | 0.08255 | 1.3370 | 0 of 128 |
| diag | gate | trace-weighted | 0.06175 | 0.06598 | 1.0685 | 0 of 128 | 0.08250 | 1.3359 | 0 of 128 |
| diag | up | unweighted | 0.06154 | 0.06580 | 1.0692 | 0 of 128 | 0.08458 | 1.3744 | 0 of 128 |
| diag | up | count-weighted | 0.06184 | 0.06608 | 1.0686 | 0 of 128 | 0.08307 | 1.3434 | 0 of 128 |
| diag | up | trace-weighted | 0.06185 | 0.06609 | 1.0685 | 0 of 128 | 0.08303 | 1.3424 | 0 of 128 |
| diag | both | unweighted | 0.06147 | 0.06572 | 1.0692 | 0 of 128 | 0.08424 | 1.3705 | 0 of 128 |
| diag | both | count-weighted | 0.06179 | 0.06603 | 1.0686 | 0 of 128 | 0.08281 | 1.3402 | 0 of 128 |
| diag | both | trace-weighted | 0.06180 | 0.06603 | 1.0685 | 0 of 128 | 0.08276 | 1.3392 | 0 of 128 |
| plain | gate | unweighted | 0.26149 | 0.26828 | 1.0259 | 0 of 128 | 0.32134 | 1.2289 | 0 of 128 |
| plain | gate | count-weighted | 0.26136 | 0.26811 | 1.0258 | 0 of 128 | 0.31771 | 1.2156 | 0 of 128 |
| plain | gate | trace-weighted | 0.26141 | 0.26815 | 1.0258 | 0 of 128 | 0.31784 | 1.2159 | 0 of 128 |
| plain | up | unweighted | 0.26148 | 0.26826 | 1.0259 | 0 of 128 | 0.32274 | 1.2343 | 0 of 128 |
| plain | up | count-weighted | 0.26132 | 0.26806 | 1.0258 | 0 of 128 | 0.31895 | 1.2206 | 0 of 128 |
| plain | up | trace-weighted | 0.26137 | 0.26811 | 1.0258 | 0 of 128 | 0.31909 | 1.2208 | 0 of 128 |
| plain | both | unweighted | 0.26149 | 0.26827 | 1.0259 | 0 of 128 | 0.32204 | 1.2316 | 0 of 128 |
| plain | both | count-weighted | 0.26134 | 0.26808 | 1.0258 | 0 of 128 | 0.31833 | 1.2181 | 0 of 128 |
| plain | both | trace-weighted | 0.26139 | 0.26813 | 1.0258 | 0 of 128 | 0.31846 | 1.2183 | 0 of 128 |
| gain | gate | unweighted | 0.99629 | 0.99714 | 1.0009 | 9 of 128 | 0.98680 | 0.9905 | 126 of 128 |
| gain | gate | count-weighted | 0.99632 | 0.99726 | 1.0009 | 9 of 128 | 0.98744 | 0.9911 | 126 of 128 |
| gain | gate | trace-weighted | 0.99631 | 0.99724 | 1.0009 | 9 of 128 | 0.98741 | 0.9911 | 126 of 128 |
| gain | up | unweighted | 0.99630 | 0.99717 | 1.0009 | 11 of 128 | 0.98656 | 0.9902 | 126 of 128 |
| gain | up | count-weighted | 0.99635 | 0.99729 | 1.0009 | 11 of 128 | 0.98724 | 0.9909 | 126 of 128 |
| gain | up | trace-weighted | 0.99635 | 0.99729 | 1.0009 | 11 of 128 | 0.98723 | 0.9908 | 126 of 128 |
| gain | both | unweighted | 0.99629 | 0.99716 | 1.0009 | 11 of 128 | 0.98668 | 0.9903 | 126 of 128 |
| gain | both | count-weighted | 0.99633 | 0.99727 | 1.0009 | 11 of 128 | 0.98734 | 0.9910 | 126 of 128 |
| gain | both | trace-weighted | 0.99633 | 0.99727 | 1.0009 | 11 of 128 | 0.98732 | 0.9910 | 126 of 128 |
| energy | gate | unweighted | 1.06096 | 1.06626 | 1.0050 | 0 of 128 | 1.07701 | 1.0151 | 0 of 128 |
| energy | gate | count-weighted | 1.06094 | 1.06639 | 1.0051 | 0 of 128 | 1.07586 | 1.0141 | 0 of 128 |
| energy | gate | trace-weighted | 1.06095 | 1.06639 | 1.0051 | 0 of 128 | 1.07588 | 1.0141 | 0 of 128 |
| energy | up | unweighted | 1.06098 | 1.06631 | 1.0050 | 0 of 128 | 1.07743 | 1.0155 | 0 of 128 |
| energy | up | count-weighted | 1.06098 | 1.06644 | 1.0051 | 0 of 128 | 1.07625 | 1.0144 | 0 of 128 |
| energy | up | trace-weighted | 1.06101 | 1.06646 | 1.0051 | 0 of 128 | 1.07631 | 1.0144 | 0 of 128 |
| energy | both | unweighted | 1.06097 | 1.06629 | 1.0050 | 0 of 128 | 1.07722 | 1.0153 | 0 of 128 |
| energy | both | count-weighted | 1.06096 | 1.06642 | 1.0051 | 0 of 128 | 1.07606 | 1.0142 | 0 of 128 |
| energy | both | trace-weighted | 1.06098 | 1.06642 | 1.0051 | 0 of 128 | 1.07610 | 1.0142 | 0 of 128 |
| ogain | gate | unweighted | 0.99894 | 0.99916 | 1.0002 | 58 of 128 | 1.01315 | 1.0142 | 0 of 128 |
| ogain | gate | count-weighted | 0.99860 | 0.99909 | 1.0005 | 58 of 128 | 1.01239 | 1.0138 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99854 | 0.99909 | 1.0005 | 58 of 128 | 1.01236 | 1.0138 | 0 of 128 |
| ogain | up | unweighted | 0.99574 | 0.99865 | 1.0029 | 0 of 128 | 1.00891 | 1.0132 | 0 of 128 |
| ogain | up | count-weighted | 0.99559 | 0.99861 | 1.0030 | 0 of 128 | 1.00819 | 1.0127 | 0 of 128 |
| ogain | up | trace-weighted | 0.99556 | 0.99861 | 1.0031 | 0 of 128 | 1.00815 | 1.0127 | 0 of 128 |
| ogain | both | unweighted | 0.99759 | 0.99896 | 1.0014 | 23 of 128 | 1.01137 | 1.0138 | 0 of 128 |
| ogain | both | count-weighted | 0.99731 | 0.99889 | 1.0016 | 23 of 128 | 1.01060 | 1.0133 | 0 of 128 |
| ogain | both | trace-weighted | 0.99726 | 0.99889 | 1.0016 | 23 of 128 | 1.01057 | 1.0133 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.1358e-02 | [-1.1819e-02, -1.0881e-02] |
| up | -1.5193e-02 | [-1.5511e-02, -1.4879e-02] |
| both | -1.2937e-02 | [-1.3329e-02, -1.2529e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.4116e-02 | [-1.4651e-02, -1.3576e-02] | -2.7580e-03 | [-3.0005e-03, -2.5393e-03] |
| full | up | -1.9904e-02 | [-2.0270e-02, -1.9566e-02] | -4.7107e-03 | [-5.0169e-03, -4.4301e-03] |
| full | both | -1.6507e-02 | [-1.6941e-02, -1.6072e-02] | -3.5699e-03 | [-3.8298e-03, -3.3370e-03] |
| reg | gate | -1.2425e-02 | [-1.2880e-02, -1.1964e-02] | -1.8293e-03 | [-2.0377e-03, -1.6395e-03] |
| reg | up | -1.7324e-02 | [-1.7628e-02, -1.7044e-02] | -3.3512e-03 | [-3.6164e-03, -3.1060e-03] |
| reg | both | -1.4469e-02 | [-1.4832e-02, -1.4104e-02] | -2.4697e-03 | [-2.6964e-03, -2.2648e-03] |
| bias | gate | -4.1139e-03 | [-4.2829e-03, -3.9443e-03] | -8.6029e-05 | [-1.4613e-04, -2.4718e-05] |
| bias | up | -6.1504e-03 | [-6.3015e-03, -6.0121e-03] | -6.6179e-04 | [-7.0705e-04, -6.1342e-04] |
| bias | both | -4.9532e-03 | [-5.0929e-03, -4.8179e-03] | -3.2605e-04 | [-3.7648e-04, -2.7259e-04] |
| rest | gate | -1.0002e-02 | [-1.0383e-02, -9.6130e-03] | -2.6720e-03 | [-2.8809e-03, -2.4867e-03] |
| rest | up | -1.3754e-02 | [-1.4024e-02, -1.3496e-02] | -4.0489e-03 | [-4.3225e-03, -3.7966e-03] |
| rest | both | -1.1553e-02 | [-1.1875e-02, -1.1231e-02] | -3.2439e-03 | [-3.4749e-03, -3.0384e-03] |
| diag | gate | +2.0806e-02 | [+2.0359e-02, +2.1299e-02] | +1.6573e-02 | [+1.6134e-02, +1.7061e-02] |
| diag | up | +2.1235e-02 | [+2.0756e-02, +2.1763e-02] | +1.6995e-02 | [+1.6520e-02, +1.7517e-02] |
| diag | both | +2.1020e-02 | [+2.0558e-02, +2.1530e-02] | +1.6783e-02 | [+1.6327e-02, +1.7288e-02] |

## Layer 7 (IQ2_S; UD `.scale` present, min 3.885e-05, max 9.079e-05)

- Gram layout: expert 71: max |strict lower| = 0, max |strict upper| = 3.509e+05
- Gram layout: expert 127: max |strict lower| = 0, max |strict upper| = 2.918e+05
- Gram layout: expert 6: max |strict lower| = 0, max |strict upper| = 2.544e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 3.33e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.7.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.822263 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 3210, median 25880, max 123819

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03367 | 0.02255 | 0.6697 | 128 of 128 | 0.01758 | 0.5221 | 128 of 128 |
| full | gate | count-weighted | 0.03390 | 0.02268 | 0.6690 | 128 of 128 | 0.01815 | 0.5354 | 128 of 128 |
| full | gate | trace-weighted | 0.03391 | 0.02271 | 0.6698 | 128 of 128 | 0.01818 | 0.5363 | 128 of 128 |
| full | up | unweighted | 0.04208 | 0.02822 | 0.6705 | 128 of 128 | 0.02139 | 0.5083 | 128 of 128 |
| full | up | count-weighted | 0.04203 | 0.02816 | 0.6701 | 128 of 128 | 0.02195 | 0.5221 | 128 of 128 |
| full | up | trace-weighted | 0.04208 | 0.02823 | 0.6709 | 128 of 128 | 0.02200 | 0.5229 | 128 of 128 |
| full | both | unweighted | 0.03733 | 0.02502 | 0.6702 | 128 of 128 | 0.01924 | 0.5153 | 128 of 128 |
| full | both | count-weighted | 0.03747 | 0.02508 | 0.6695 | 128 of 128 | 0.01981 | 0.5288 | 128 of 128 |
| full | both | trace-weighted | 0.03749 | 0.02513 | 0.6704 | 128 of 128 | 0.01986 | 0.5297 | 128 of 128 |
| reg | gate | unweighted | 0.03510 | 0.02479 | 0.7062 | 128 of 128 | 0.02100 | 0.5983 | 128 of 128 |
| reg | gate | count-weighted | 0.03533 | 0.02493 | 0.7056 | 128 of 128 | 0.02150 | 0.6085 | 128 of 128 |
| reg | gate | trace-weighted | 0.03534 | 0.02496 | 0.7063 | 128 of 128 | 0.02153 | 0.6093 | 128 of 128 |
| reg | up | unweighted | 0.04331 | 0.03062 | 0.7069 | 128 of 128 | 0.02539 | 0.5862 | 128 of 128 |
| reg | up | count-weighted | 0.04327 | 0.03057 | 0.7065 | 128 of 128 | 0.02583 | 0.5968 | 128 of 128 |
| reg | up | trace-weighted | 0.04332 | 0.03064 | 0.7072 | 128 of 128 | 0.02588 | 0.5974 | 128 of 128 |
| reg | both | unweighted | 0.03870 | 0.02735 | 0.7066 | 128 of 128 | 0.02293 | 0.5924 | 128 of 128 |
| reg | both | count-weighted | 0.03884 | 0.02742 | 0.7060 | 128 of 128 | 0.02341 | 0.6028 | 128 of 128 |
| reg | both | trace-weighted | 0.03886 | 0.02747 | 0.7068 | 128 of 128 | 0.02345 | 0.6034 | 128 of 128 |
| bias | gate | unweighted | 0.00452 | 9.014e-04 | 0.1992 | 128 of 128 | 4.118e-04 | 0.0910 | 128 of 128 |
| bias | gate | count-weighted | 0.00450 | 8.757e-04 | 0.1946 | 128 of 128 | 4.041e-04 | 0.0898 | 128 of 128 |
| bias | gate | trace-weighted | 0.00449 | 8.703e-04 | 0.1940 | 128 of 128 | 4.041e-04 | 0.0901 | 128 of 128 |
| bias | up | unweighted | 0.00563 | 0.00113 | 0.2015 | 128 of 128 | 2.225e-04 | 0.0395 | 128 of 128 |
| bias | up | count-weighted | 0.00556 | 0.00109 | 0.1963 | 128 of 128 | 2.103e-04 | 0.0378 | 128 of 128 |
| bias | up | trace-weighted | 0.00555 | 0.00109 | 0.1958 | 128 of 128 | 2.080e-04 | 0.0375 | 128 of 128 |
| bias | both | unweighted | 0.00500 | 0.00100 | 0.2004 | 128 of 128 | 3.289e-04 | 0.0657 | 128 of 128 |
| bias | both | count-weighted | 0.00496 | 9.704e-04 | 0.1955 | 128 of 128 | 3.188e-04 | 0.0642 | 128 of 128 |
| bias | both | trace-weighted | 0.00495 | 9.652e-04 | 0.1949 | 128 of 128 | 3.179e-04 | 0.0642 | 128 of 128 |
| rest | gate | unweighted | 0.02915 | 0.02165 | 0.7428 | 128 of 128 | 0.01717 | 0.5890 | 128 of 128 |
| rest | gate | count-weighted | 0.02940 | 0.02180 | 0.7417 | 128 of 128 | 0.01774 | 0.6036 | 128 of 128 |
| rest | gate | trace-weighted | 0.02942 | 0.02184 | 0.7424 | 128 of 128 | 0.01778 | 0.6044 | 128 of 128 |
| rest | up | unweighted | 0.03645 | 0.02708 | 0.7430 | 128 of 128 | 0.02117 | 0.5807 | 128 of 128 |
| rest | up | count-weighted | 0.03647 | 0.02707 | 0.7423 | 128 of 128 | 0.02174 | 0.5960 | 128 of 128 |
| rest | up | trace-weighted | 0.03653 | 0.02714 | 0.7431 | 128 of 128 | 0.02180 | 0.5967 | 128 of 128 |
| rest | both | unweighted | 0.03233 | 0.02402 | 0.7429 | 128 of 128 | 0.01891 | 0.5849 | 128 of 128 |
| rest | both | count-weighted | 0.03250 | 0.02411 | 0.7420 | 128 of 128 | 0.01949 | 0.5998 | 128 of 128 |
| rest | both | trace-weighted | 0.03254 | 0.02417 | 0.7427 | 128 of 128 | 0.01954 | 0.6006 | 128 of 128 |
| diag | gate | unweighted | 0.06151 | 0.06587 | 1.0708 | 0 of 128 | 0.08364 | 1.3597 | 0 of 128 |
| diag | gate | count-weighted | 0.06158 | 0.06592 | 1.0705 | 0 of 128 | 0.08245 | 1.3389 | 0 of 128 |
| diag | gate | trace-weighted | 0.06158 | 0.06592 | 1.0704 | 0 of 128 | 0.08238 | 1.3378 | 0 of 128 |
| diag | up | unweighted | 0.06157 | 0.06593 | 1.0708 | 0 of 128 | 0.08397 | 1.3637 | 0 of 128 |
| diag | up | count-weighted | 0.06164 | 0.06598 | 1.0705 | 0 of 128 | 0.08274 | 1.3423 | 0 of 128 |
| diag | up | trace-weighted | 0.06163 | 0.06597 | 1.0704 | 0 of 128 | 0.08267 | 1.3413 | 0 of 128 |
| diag | both | unweighted | 0.06154 | 0.06590 | 1.0708 | 0 of 128 | 0.08381 | 1.3617 | 0 of 128 |
| diag | both | count-weighted | 0.06161 | 0.06595 | 1.0705 | 0 of 128 | 0.08259 | 1.3406 | 0 of 128 |
| diag | both | trace-weighted | 0.06161 | 0.06595 | 1.0704 | 0 of 128 | 0.08252 | 1.3395 | 0 of 128 |
| plain | gate | unweighted | 0.26275 | 0.26964 | 1.0262 | 0 of 128 | 0.32830 | 1.2495 | 0 of 128 |
| plain | gate | count-weighted | 0.26271 | 0.26963 | 1.0263 | 0 of 128 | 0.32564 | 1.2395 | 0 of 128 |
| plain | gate | trace-weighted | 0.26274 | 0.26966 | 1.0263 | 0 of 128 | 0.32558 | 1.2392 | 0 of 128 |
| plain | up | unweighted | 0.26279 | 0.26967 | 1.0262 | 0 of 128 | 0.32931 | 1.2532 | 0 of 128 |
| plain | up | count-weighted | 0.26277 | 0.26969 | 1.0263 | 0 of 128 | 0.32665 | 1.2431 | 0 of 128 |
| plain | up | trace-weighted | 0.26279 | 0.26971 | 1.0263 | 0 of 128 | 0.32660 | 1.2428 | 0 of 128 |
| plain | both | unweighted | 0.26277 | 0.26966 | 1.0262 | 0 of 128 | 0.32881 | 1.2513 | 0 of 128 |
| plain | both | count-weighted | 0.26274 | 0.26966 | 1.0263 | 0 of 128 | 0.32615 | 1.2413 | 0 of 128 |
| plain | both | trace-weighted | 0.26276 | 0.26968 | 1.0263 | 0 of 128 | 0.32609 | 1.2410 | 0 of 128 |
| gain | gate | unweighted | 0.99719 | 0.99831 | 1.0011 | 2 of 128 | 0.98684 | 0.9896 | 128 of 128 |
| gain | gate | count-weighted | 0.99719 | 0.99832 | 1.0011 | 2 of 128 | 0.98733 | 0.9901 | 128 of 128 |
| gain | gate | trace-weighted | 0.99720 | 0.99833 | 1.0011 | 2 of 128 | 0.98736 | 0.9901 | 128 of 128 |
| gain | up | unweighted | 0.99721 | 0.99834 | 1.0011 | 3 of 128 | 0.98698 | 0.9897 | 128 of 128 |
| gain | up | count-weighted | 0.99722 | 0.99836 | 1.0011 | 3 of 128 | 0.98747 | 0.9902 | 128 of 128 |
| gain | up | trace-weighted | 0.99724 | 0.99837 | 1.0011 | 3 of 128 | 0.98750 | 0.9902 | 128 of 128 |
| gain | both | unweighted | 0.99720 | 0.99832 | 1.0011 | 2 of 128 | 0.98691 | 0.9897 | 128 of 128 |
| gain | both | count-weighted | 0.99721 | 0.99834 | 1.0011 | 2 of 128 | 0.98740 | 0.9902 | 128 of 128 |
| gain | both | trace-weighted | 0.99722 | 0.99835 | 1.0011 | 2 of 128 | 0.98743 | 0.9902 | 128 of 128 |
| energy | gate | unweighted | 1.06341 | 1.06932 | 1.0056 | 0 of 128 | 1.08150 | 1.0170 | 0 of 128 |
| energy | gate | count-weighted | 1.06340 | 1.06934 | 1.0056 | 0 of 128 | 1.08073 | 1.0163 | 0 of 128 |
| energy | gate | trace-weighted | 1.06343 | 1.06937 | 1.0056 | 0 of 128 | 1.08075 | 1.0163 | 0 of 128 |
| energy | up | unweighted | 1.06348 | 1.06941 | 1.0056 | 0 of 128 | 1.08245 | 1.0178 | 0 of 128 |
| energy | up | count-weighted | 1.06350 | 1.06946 | 1.0056 | 0 of 128 | 1.08167 | 1.0171 | 0 of 128 |
| energy | up | trace-weighted | 1.06353 | 1.06949 | 1.0056 | 0 of 128 | 1.08169 | 1.0171 | 0 of 128 |
| energy | both | unweighted | 1.06345 | 1.06936 | 1.0056 | 0 of 128 | 1.08197 | 1.0174 | 0 of 128 |
| energy | both | count-weighted | 1.06345 | 1.06940 | 1.0056 | 0 of 128 | 1.08120 | 1.0167 | 0 of 128 |
| energy | both | trace-weighted | 1.06348 | 1.06943 | 1.0056 | 0 of 128 | 1.08122 | 1.0167 | 0 of 128 |
| ogain | gate | unweighted | 0.99448 | 0.99915 | 1.0047 | 1 of 128 | 1.00832 | 1.0139 | 0 of 128 |
| ogain | gate | count-weighted | 0.99468 | 0.99911 | 1.0045 | 1 of 128 | 1.00812 | 1.0135 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99471 | 0.99911 | 1.0044 | 1 of 128 | 1.00811 | 1.0135 | 0 of 128 |
| ogain | up | unweighted | 0.99612 | 0.99897 | 1.0029 | 0 of 128 | 1.00613 | 1.0101 | 0 of 128 |
| ogain | up | count-weighted | 0.99619 | 0.99894 | 1.0028 | 0 of 128 | 1.00586 | 1.0097 | 0 of 128 |
| ogain | up | trace-weighted | 0.99620 | 0.99895 | 1.0028 | 0 of 128 | 1.00582 | 1.0097 | 0 of 128 |
| ogain | both | unweighted | 0.99520 | 0.99907 | 1.0039 | 0 of 128 | 1.00736 | 1.0122 | 0 of 128 |
| ogain | both | count-weighted | 0.99535 | 0.99904 | 1.0037 | 0 of 128 | 1.00712 | 1.0118 | 0 of 128 |
| ogain | both | trace-weighted | 0.99537 | 0.99904 | 1.0037 | 0 of 128 | 1.00710 | 1.0118 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.1220e-02 | [-1.1435e-02, -1.0995e-02] |
| up | -1.3868e-02 | [-1.4028e-02, -1.3700e-02] |
| both | -1.2381e-02 | [-1.2555e-02, -1.2201e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.5751e-02 | [-1.6109e-02, -1.5387e-02] | -4.5312e-03 | [-4.7559e-03, -4.3199e-03] |
| full | up | -2.0085e-02 | [-2.0378e-02, -1.9812e-02] | -6.2173e-03 | [-6.4834e-03, -5.9770e-03] |
| full | both | -1.7652e-02 | [-1.7955e-02, -1.7355e-02] | -5.2714e-03 | [-5.5089e-03, -5.0554e-03] |
| reg | gate | -1.3831e-02 | [-1.4132e-02, -1.3524e-02] | -3.4282e-03 | [-3.6167e-03, -3.2527e-03] |
| reg | up | -1.7447e-02 | [-1.7686e-02, -1.7222e-02] | -4.7458e-03 | [-4.9705e-03, -4.5423e-03] |
| reg | both | -1.5429e-02 | [-1.5680e-02, -1.5183e-02] | -4.0112e-03 | [-4.2115e-03, -3.8274e-03] |
| bias | gate | -4.0960e-03 | [-4.2202e-03, -3.9732e-03] | -4.7162e-04 | [-5.1241e-04, -4.3107e-04] |
| bias | up | -5.3507e-03 | [-5.4579e-03, -5.2437e-03] | -8.8140e-04 | [-9.1638e-04, -8.4854e-04] |
| bias | both | -4.6459e-03 | [-4.7514e-03, -4.5416e-03] | -6.5159e-04 | [-6.8649e-04, -6.1815e-04] |
| rest | gate | -1.1655e-02 | [-1.1908e-02, -1.1401e-02] | -4.0596e-03 | [-4.2598e-03, -3.8741e-03] |
| rest | up | -1.4735e-02 | [-1.4953e-02, -1.4536e-02] | -5.3359e-03 | [-5.5783e-03, -5.1160e-03] |
| rest | both | -1.3007e-02 | [-1.3226e-02, -1.2792e-02] | -4.6198e-03 | [-4.8367e-03, -4.4220e-03] |
| diag | gate | +2.0870e-02 | [+2.0441e-02, +2.1310e-02] | +1.6528e-02 | [+1.6103e-02, +1.6959e-02] |
| diag | up | +2.1100e-02 | [+2.0687e-02, +2.1533e-02] | +1.6755e-02 | [+1.6344e-02, +1.7181e-02] |
| diag | both | +2.0985e-02 | [+2.0566e-02, +2.1420e-02] | +1.6641e-02 | [+1.6224e-02, +1.7069e-02] |

## Layer 8 (IQ2_S; UD `.scale` present, min 3.905e-05, max 0.0001062)

- Gram layout: expert 92: max |strict lower| = 0, max |strict upper| = 4.706e+05
- Gram layout: expert 84: max |strict lower| = 0, max |strict upper| = 3.09e+05
- Gram layout: expert 34: max |strict lower| = 0, max |strict upper| = 3.383e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 5.9e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.8.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.807175 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 3002, median 24479, max 107564

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03381 | 0.02398 | 0.7092 | 128 of 128 | 0.01823 | 0.5392 | 128 of 128 |
| full | gate | count-weighted | 0.03427 | 0.02428 | 0.7085 | 128 of 128 | 0.01907 | 0.5564 | 128 of 128 |
| full | gate | trace-weighted | 0.03429 | 0.02433 | 0.7095 | 128 of 128 | 0.01911 | 0.5573 | 128 of 128 |
| full | up | unweighted | 0.04458 | 0.03162 | 0.7092 | 128 of 128 | 0.02313 | 0.5189 | 128 of 128 |
| full | up | count-weighted | 0.04479 | 0.03178 | 0.7094 | 128 of 128 | 0.02404 | 0.5368 | 128 of 128 |
| full | up | trace-weighted | 0.04491 | 0.03190 | 0.7104 | 128 of 128 | 0.02413 | 0.5373 | 128 of 128 |
| full | both | unweighted | 0.03833 | 0.02718 | 0.7092 | 128 of 128 | 0.02029 | 0.5293 | 128 of 128 |
| full | both | count-weighted | 0.03872 | 0.02745 | 0.7089 | 128 of 128 | 0.02117 | 0.5468 | 128 of 128 |
| full | both | trace-weighted | 0.03878 | 0.02753 | 0.7099 | 128 of 128 | 0.02123 | 0.5475 | 128 of 128 |
| reg | gate | unweighted | 0.03524 | 0.02611 | 0.7411 | 128 of 128 | 0.02156 | 0.6120 | 128 of 128 |
| reg | gate | count-weighted | 0.03569 | 0.02643 | 0.7405 | 128 of 128 | 0.02232 | 0.6253 | 128 of 128 |
| reg | gate | trace-weighted | 0.03571 | 0.02648 | 0.7414 | 128 of 128 | 0.02236 | 0.6261 | 128 of 128 |
| reg | up | unweighted | 0.04572 | 0.03388 | 0.7412 | 128 of 128 | 0.02716 | 0.5941 | 128 of 128 |
| reg | up | count-weighted | 0.04592 | 0.03405 | 0.7414 | 128 of 128 | 0.02791 | 0.6079 | 128 of 128 |
| reg | up | trace-weighted | 0.04603 | 0.03417 | 0.7423 | 128 of 128 | 0.02800 | 0.6083 | 128 of 128 |
| reg | both | unweighted | 0.03968 | 0.02941 | 0.7411 | 128 of 128 | 0.02394 | 0.6032 | 128 of 128 |
| reg | both | count-weighted | 0.04006 | 0.02968 | 0.7409 | 128 of 128 | 0.02471 | 0.6167 | 128 of 128 |
| reg | both | trace-weighted | 0.04012 | 0.02976 | 0.7418 | 128 of 128 | 0.02477 | 0.6173 | 128 of 128 |
| bias | gate | unweighted | 0.00456 | 0.00105 | 0.2305 | 128 of 128 | 4.772e-04 | 0.1047 | 128 of 128 |
| bias | gate | count-weighted | 0.00462 | 0.00105 | 0.2267 | 128 of 128 | 4.933e-04 | 0.1068 | 128 of 128 |
| bias | gate | trace-weighted | 0.00461 | 0.00104 | 0.2263 | 128 of 128 | 4.946e-04 | 0.1073 | 128 of 128 |
| bias | up | unweighted | 0.00600 | 0.00139 | 0.2324 | 128 of 128 | 2.110e-04 | 0.0351 | 128 of 128 |
| bias | up | count-weighted | 0.00599 | 0.00137 | 0.2292 | 128 of 128 | 2.131e-04 | 0.0356 | 128 of 128 |
| bias | up | trace-weighted | 0.00599 | 0.00137 | 0.2287 | 128 of 128 | 2.104e-04 | 0.0351 | 128 of 128 |
| bias | both | unweighted | 0.00516 | 0.00119 | 0.2313 | 128 of 128 | 3.644e-04 | 0.0706 | 128 of 128 |
| bias | both | count-weighted | 0.00520 | 0.00118 | 0.2279 | 128 of 128 | 3.739e-04 | 0.0720 | 128 of 128 |
| bias | both | trace-weighted | 0.00519 | 0.00118 | 0.2275 | 128 of 128 | 3.736e-04 | 0.0719 | 128 of 128 |
| rest | gate | unweighted | 0.02925 | 0.02293 | 0.7837 | 128 of 128 | 0.01775 | 0.6069 | 128 of 128 |
| rest | gate | count-weighted | 0.02965 | 0.02323 | 0.7835 | 128 of 128 | 0.01858 | 0.6265 | 128 of 128 |
| rest | gate | trace-weighted | 0.02968 | 0.02329 | 0.7846 | 128 of 128 | 0.01862 | 0.6272 | 128 of 128 |
| rest | up | unweighted | 0.03858 | 0.03022 | 0.7834 | 128 of 128 | 0.02292 | 0.5941 | 128 of 128 |
| rest | up | count-weighted | 0.03881 | 0.03040 | 0.7835 | 128 of 128 | 0.02383 | 0.6141 | 128 of 128 |
| rest | up | trace-weighted | 0.03892 | 0.03053 | 0.7845 | 128 of 128 | 0.02392 | 0.6146 | 128 of 128 |
| rest | both | unweighted | 0.03317 | 0.02599 | 0.7836 | 128 of 128 | 0.01992 | 0.6007 | 128 of 128 |
| rest | both | count-weighted | 0.03352 | 0.02627 | 0.7835 | 128 of 128 | 0.02080 | 0.6204 | 128 of 128 |
| rest | both | trace-weighted | 0.03358 | 0.02635 | 0.7845 | 128 of 128 | 0.02086 | 0.6210 | 128 of 128 |
| diag | gate | unweighted | 0.06154 | 0.06528 | 1.0608 | 0 of 128 | 0.08250 | 1.3407 | 0 of 128 |
| diag | gate | count-weighted | 0.06156 | 0.06528 | 1.0606 | 0 of 128 | 0.08087 | 1.3138 | 0 of 128 |
| diag | gate | trace-weighted | 0.06155 | 0.06527 | 1.0605 | 0 of 128 | 0.08080 | 1.3128 | 0 of 128 |
| diag | up | unweighted | 0.06163 | 0.06537 | 1.0607 | 0 of 128 | 0.08291 | 1.3452 | 0 of 128 |
| diag | up | count-weighted | 0.06163 | 0.06536 | 1.0605 | 0 of 128 | 0.08121 | 1.3176 | 0 of 128 |
| diag | up | trace-weighted | 0.06162 | 0.06535 | 1.0604 | 0 of 128 | 0.08113 | 1.3166 | 0 of 128 |
| diag | both | unweighted | 0.06158 | 0.06533 | 1.0608 | 0 of 128 | 0.08270 | 1.3429 | 0 of 128 |
| diag | both | count-weighted | 0.06159 | 0.06532 | 1.0605 | 0 of 128 | 0.08104 | 1.3157 | 0 of 128 |
| diag | both | trace-weighted | 0.06159 | 0.06531 | 1.0605 | 0 of 128 | 0.08097 | 1.3147 | 0 of 128 |
| plain | gate | unweighted | 0.26225 | 0.26812 | 1.0224 | 0 of 128 | 0.32019 | 1.2209 | 0 of 128 |
| plain | gate | count-weighted | 0.26225 | 0.26812 | 1.0224 | 0 of 128 | 0.31689 | 1.2083 | 0 of 128 |
| plain | gate | trace-weighted | 0.26228 | 0.26815 | 1.0224 | 0 of 128 | 0.31684 | 1.2080 | 0 of 128 |
| plain | up | unweighted | 0.26230 | 0.26817 | 1.0224 | 0 of 128 | 0.32140 | 1.2253 | 0 of 128 |
| plain | up | count-weighted | 0.26231 | 0.26818 | 1.0224 | 0 of 128 | 0.31803 | 1.2124 | 0 of 128 |
| plain | up | trace-weighted | 0.26234 | 0.26821 | 1.0224 | 0 of 128 | 0.31801 | 1.2122 | 0 of 128 |
| plain | both | unweighted | 0.26227 | 0.26814 | 1.0224 | 0 of 128 | 0.32080 | 1.2231 | 0 of 128 |
| plain | both | count-weighted | 0.26228 | 0.26815 | 1.0224 | 0 of 128 | 0.31746 | 1.2104 | 0 of 128 |
| plain | both | trace-weighted | 0.26231 | 0.26818 | 1.0224 | 0 of 128 | 0.31742 | 1.2101 | 0 of 128 |
| gain | gate | unweighted | 0.99745 | 0.99838 | 1.0009 | 5 of 128 | 0.98832 | 0.9909 | 128 of 128 |
| gain | gate | count-weighted | 0.99747 | 0.99841 | 1.0009 | 5 of 128 | 0.98896 | 0.9915 | 128 of 128 |
| gain | gate | trace-weighted | 0.99749 | 0.99842 | 1.0009 | 5 of 128 | 0.98900 | 0.9915 | 128 of 128 |
| gain | up | unweighted | 0.99747 | 0.99842 | 1.0010 | 3 of 128 | 0.98835 | 0.9909 | 128 of 128 |
| gain | up | count-weighted | 0.99751 | 0.99846 | 1.0010 | 3 of 128 | 0.98899 | 0.9915 | 128 of 128 |
| gain | up | trace-weighted | 0.99752 | 0.99848 | 1.0010 | 3 of 128 | 0.98902 | 0.9915 | 128 of 128 |
| gain | both | unweighted | 0.99746 | 0.99840 | 1.0009 | 5 of 128 | 0.98833 | 0.9909 | 128 of 128 |
| gain | both | count-weighted | 0.99749 | 0.99844 | 1.0010 | 5 of 128 | 0.98898 | 0.9915 | 128 of 128 |
| gain | both | trace-weighted | 0.99751 | 0.99845 | 1.0009 | 5 of 128 | 0.98901 | 0.9915 | 128 of 128 |
| energy | gate | unweighted | 1.06367 | 1.06864 | 1.0047 | 0 of 128 | 1.07921 | 1.0146 | 0 of 128 |
| energy | gate | count-weighted | 1.06372 | 1.06871 | 1.0047 | 0 of 128 | 1.07836 | 1.0138 | 0 of 128 |
| energy | gate | trace-weighted | 1.06377 | 1.06875 | 1.0047 | 0 of 128 | 1.07841 | 1.0138 | 0 of 128 |
| energy | up | unweighted | 1.06373 | 1.06875 | 1.0047 | 0 of 128 | 1.08003 | 1.0153 | 0 of 128 |
| energy | up | count-weighted | 1.06382 | 1.06885 | 1.0047 | 0 of 128 | 1.07915 | 1.0144 | 0 of 128 |
| energy | up | trace-weighted | 1.06387 | 1.06889 | 1.0047 | 0 of 128 | 1.07919 | 1.0144 | 0 of 128 |
| energy | both | unweighted | 1.06370 | 1.06870 | 1.0047 | 0 of 128 | 1.07962 | 1.0150 | 0 of 128 |
| energy | both | count-weighted | 1.06377 | 1.06878 | 1.0047 | 0 of 128 | 1.07876 | 1.0141 | 0 of 128 |
| energy | both | trace-weighted | 1.06382 | 1.06882 | 1.0047 | 0 of 128 | 1.07880 | 1.0141 | 0 of 128 |
| ogain | gate | unweighted | 0.99500 | 0.99909 | 1.0041 | 5 of 128 | 1.00911 | 1.0142 | 0 of 128 |
| ogain | gate | count-weighted | 0.99539 | 0.99903 | 1.0037 | 5 of 128 | 1.00923 | 1.0139 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99542 | 0.99903 | 1.0036 | 5 of 128 | 1.00925 | 1.0139 | 0 of 128 |
| ogain | up | unweighted | 0.99617 | 0.99880 | 1.0026 | 0 of 128 | 1.00587 | 1.0097 | 0 of 128 |
| ogain | up | count-weighted | 0.99633 | 0.99887 | 1.0025 | 0 of 128 | 1.00581 | 1.0095 | 0 of 128 |
| ogain | up | trace-weighted | 0.99635 | 0.99887 | 1.0025 | 0 of 128 | 1.00577 | 1.0095 | 0 of 128 |
| ogain | both | unweighted | 0.99550 | 0.99897 | 1.0035 | 1 of 128 | 1.00772 | 1.0123 | 0 of 128 |
| ogain | both | count-weighted | 0.99579 | 0.99896 | 1.0032 | 1 of 128 | 1.00776 | 1.0120 | 0 of 128 |
| ogain | both | trace-weighted | 0.99581 | 0.99896 | 1.0032 | 1 of 128 | 1.00775 | 1.0120 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -9.9897e-03 | [-1.0215e-02, -9.7548e-03] |
| up | -1.3016e-02 | [-1.3220e-02, -1.2819e-02] |
| both | -1.1270e-02 | [-1.1463e-02, -1.1078e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.5200e-02 | [-1.5554e-02, -1.4833e-02] | -5.2107e-03 | [-5.4435e-03, -4.9897e-03] |
| full | up | -2.0749e-02 | [-2.1071e-02, -2.0435e-02] | -7.7329e-03 | [-8.0133e-03, -7.4759e-03] |
| full | both | -1.7550e-02 | [-1.7845e-02, -1.7259e-02] | -6.2795e-03 | [-6.5198e-03, -6.0558e-03] |
| reg | gate | -1.3373e-02 | [-1.3669e-02, -1.3063e-02] | -4.1109e-03 | [-4.3075e-03, -3.9233e-03] |
| reg | up | -1.8008e-02 | [-1.8270e-02, -1.7750e-02] | -6.1306e-03 | [-6.3694e-03, -5.9122e-03] |
| reg | both | -1.5356e-02 | [-1.5601e-02, -1.5113e-02] | -4.9758e-03 | [-5.1792e-03, -4.7858e-03] |
| bias | gate | -4.1247e-03 | [-4.2522e-03, -3.9940e-03] | -5.5363e-04 | [-6.0201e-04, -5.0607e-04] |
| bias | up | -5.7722e-03 | [-5.8980e-03, -5.6456e-03] | -1.1586e-03 | [-1.2006e-03, -1.1174e-03] |
| bias | both | -4.8227e-03 | [-4.9390e-03, -4.7104e-03] | -8.1035e-04 | [-8.5224e-04, -7.7036e-04] |
| rest | gate | -1.1076e-02 | [-1.1335e-02, -1.0815e-02] | -4.6571e-03 | [-4.8637e-03, -4.4620e-03] |
| rest | up | -1.4977e-02 | [-1.5226e-02, -1.4742e-02] | -6.5743e-03 | [-6.8433e-03, -6.3321e-03] |
| rest | both | -1.2727e-02 | [-1.2950e-02, -1.2513e-02] | -5.4692e-03 | [-5.6930e-03, -5.2632e-03] |
| diag | gate | +1.9316e-02 | [+1.8962e-02, +1.9720e-02] | +1.5586e-02 | [+1.5237e-02, +1.5991e-02] |
| diag | up | +1.9575e-02 | [+1.9214e-02, +1.9985e-02] | +1.5848e-02 | [+1.5495e-02, +1.6254e-02] |
| diag | both | +1.9445e-02 | [+1.9089e-02, +1.9852e-02] | +1.5717e-02 | [+1.5365e-02, +1.6125e-02] |

## Layer 9 (IQ3_XXS; UD `.scale` present, min 3.921e-05, max 0.0001086)

- Gram layout: expert 21: max |strict lower| = 0, max |strict upper| = 4.746e+05
- Gram layout: expert 86: max |strict lower| = 0, max |strict upper| = 3.057e+05
- Gram layout: expert 114: max |strict lower| = 0, max |strict upper| = 2.807e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 4.94e-09; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.9.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.825022 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 3370, median 22722, max 140969

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.01749 | 0.01321 | 0.7553 | 128 of 128 | 0.00968 | 0.5533 | 128 of 128 |
| full | gate | count-weighted | 0.01762 | 0.01328 | 0.7537 | 128 of 128 | 0.01001 | 0.5679 | 128 of 128 |
| full | gate | trace-weighted | 0.01762 | 0.01330 | 0.7547 | 128 of 128 | 0.01003 | 0.5689 | 128 of 128 |
| full | up | unweighted | 0.02399 | 0.01813 | 0.7557 | 128 of 128 | 0.01263 | 0.5264 | 128 of 128 |
| full | up | count-weighted | 0.02403 | 0.01811 | 0.7538 | 128 of 128 | 0.01300 | 0.5412 | 128 of 128 |
| full | up | trace-weighted | 0.02409 | 0.01818 | 0.7550 | 128 of 128 | 0.01305 | 0.5419 | 128 of 128 |
| full | both | unweighted | 0.02017 | 0.01524 | 0.7555 | 128 of 128 | 0.01089 | 0.5401 | 128 of 128 |
| full | both | count-weighted | 0.02028 | 0.01529 | 0.7537 | 128 of 128 | 0.01125 | 0.5547 | 128 of 128 |
| full | both | trace-weighted | 0.02031 | 0.01533 | 0.7548 | 128 of 128 | 0.01128 | 0.5556 | 128 of 128 |
| reg | gate | unweighted | 0.01823 | 0.01430 | 0.7845 | 128 of 128 | 0.01136 | 0.6234 | 128 of 128 |
| reg | gate | count-weighted | 0.01836 | 0.01437 | 0.7830 | 128 of 128 | 0.01166 | 0.6349 | 128 of 128 |
| reg | gate | trace-weighted | 0.01836 | 0.01439 | 0.7839 | 128 of 128 | 0.01167 | 0.6357 | 128 of 128 |
| reg | up | unweighted | 0.02453 | 0.01925 | 0.7848 | 128 of 128 | 0.01471 | 0.5998 | 128 of 128 |
| reg | up | count-weighted | 0.02457 | 0.01924 | 0.7830 | 128 of 128 | 0.01502 | 0.6113 | 128 of 128 |
| reg | up | trace-weighted | 0.02462 | 0.01930 | 0.7841 | 128 of 128 | 0.01506 | 0.6118 | 128 of 128 |
| reg | both | unweighted | 0.02086 | 0.01636 | 0.7846 | 128 of 128 | 0.01276 | 0.6118 | 128 of 128 |
| reg | both | count-weighted | 0.02097 | 0.01642 | 0.7829 | 128 of 128 | 0.01307 | 0.6232 | 128 of 128 |
| reg | both | trace-weighted | 0.02099 | 0.01645 | 0.7839 | 128 of 128 | 0.01310 | 0.6239 | 128 of 128 |
| bias | gate | unweighted | 0.00213 | 4.801e-04 | 0.2254 | 128 of 128 | 2.960e-04 | 0.1390 | 128 of 128 |
| bias | gate | count-weighted | 0.00215 | 4.793e-04 | 0.2228 | 128 of 128 | 3.030e-04 | 0.1408 | 128 of 128 |
| bias | gate | trace-weighted | 0.00215 | 4.768e-04 | 0.2220 | 128 of 128 | 3.038e-04 | 0.1414 | 128 of 128 |
| bias | up | unweighted | 0.00292 | 6.598e-04 | 0.2261 | 128 of 128 | 1.131e-04 | 0.0388 | 128 of 128 |
| bias | up | count-weighted | 0.00294 | 6.537e-04 | 0.2224 | 128 of 128 | 1.145e-04 | 0.0389 | 128 of 128 |
| bias | up | trace-weighted | 0.00294 | 6.522e-04 | 0.2220 | 128 of 128 | 1.127e-04 | 0.0384 | 128 of 128 |
| bias | both | unweighted | 0.00246 | 5.542e-04 | 0.2257 | 128 of 128 | 2.202e-04 | 0.0897 | 128 of 128 |
| bias | both | count-weighted | 0.00248 | 5.518e-04 | 0.2226 | 128 of 128 | 2.244e-04 | 0.0905 | 128 of 128 |
| bias | both | trace-weighted | 0.00248 | 5.497e-04 | 0.2220 | 128 of 128 | 2.242e-04 | 0.0906 | 128 of 128 |
| rest | gate | unweighted | 0.01536 | 0.01273 | 0.8288 | 128 of 128 | 0.00938 | 0.6107 | 128 of 128 |
| rest | gate | count-weighted | 0.01547 | 0.01280 | 0.8275 | 128 of 128 | 0.00971 | 0.6273 | 128 of 128 |
| rest | gate | trace-weighted | 0.01548 | 0.01282 | 0.8287 | 128 of 128 | 0.00972 | 0.6283 | 128 of 128 |
| rest | up | unweighted | 0.02107 | 0.01747 | 0.8291 | 128 of 128 | 0.01251 | 0.5939 | 128 of 128 |
| rest | up | count-weighted | 0.02109 | 0.01746 | 0.8278 | 128 of 128 | 0.01289 | 0.6112 | 128 of 128 |
| rest | up | trace-weighted | 0.02115 | 0.01753 | 0.8290 | 128 of 128 | 0.01294 | 0.6118 | 128 of 128 |
| rest | both | unweighted | 0.01771 | 0.01468 | 0.8289 | 128 of 128 | 0.01067 | 0.6025 | 128 of 128 |
| rest | both | count-weighted | 0.01781 | 0.01474 | 0.8276 | 128 of 128 | 0.01103 | 0.6194 | 128 of 128 |
| rest | both | trace-weighted | 0.01783 | 0.01478 | 0.8288 | 128 of 128 | 0.01106 | 0.6202 | 128 of 128 |
| diag | gate | unweighted | 0.03173 | 0.03414 | 1.0763 | 0 of 128 | 0.04203 | 1.3250 | 0 of 128 |
| diag | gate | count-weighted | 0.03174 | 0.03414 | 1.0757 | 0 of 128 | 0.04141 | 1.3047 | 0 of 128 |
| diag | gate | trace-weighted | 0.03174 | 0.03413 | 1.0755 | 0 of 128 | 0.04137 | 1.3035 | 0 of 128 |
| diag | up | unweighted | 0.03176 | 0.03418 | 1.0763 | 0 of 128 | 0.04236 | 1.3338 | 0 of 128 |
| diag | up | count-weighted | 0.03177 | 0.03417 | 1.0758 | 0 of 128 | 0.04171 | 1.3130 | 0 of 128 |
| diag | up | trace-weighted | 0.03176 | 0.03416 | 1.0756 | 0 of 128 | 0.04167 | 1.3119 | 0 of 128 |
| diag | both | unweighted | 0.03174 | 0.03416 | 1.0763 | 0 of 128 | 0.04220 | 1.3294 | 0 of 128 |
| diag | both | count-weighted | 0.03175 | 0.03416 | 1.0757 | 0 of 128 | 0.04156 | 1.3088 | 0 of 128 |
| diag | both | trace-weighted | 0.03175 | 0.03415 | 1.0756 | 0 of 128 | 0.04152 | 1.3077 | 0 of 128 |
| plain | gate | unweighted | 0.18715 | 0.19324 | 1.0326 | 0 of 128 | 0.22217 | 1.1871 | 0 of 128 |
| plain | gate | count-weighted | 0.18713 | 0.19322 | 1.0326 | 0 of 128 | 0.22041 | 1.1779 | 0 of 128 |
| plain | gate | trace-weighted | 0.18713 | 0.19323 | 1.0326 | 0 of 128 | 0.22037 | 1.1776 | 0 of 128 |
| plain | up | unweighted | 0.18712 | 0.19321 | 1.0325 | 0 of 128 | 0.22303 | 1.1919 | 0 of 128 |
| plain | up | count-weighted | 0.18710 | 0.19319 | 1.0326 | 0 of 128 | 0.22125 | 1.1825 | 0 of 128 |
| plain | up | trace-weighted | 0.18711 | 0.19321 | 1.0326 | 0 of 128 | 0.22122 | 1.1823 | 0 of 128 |
| plain | both | unweighted | 0.18713 | 0.19322 | 1.0325 | 0 of 128 | 0.22260 | 1.1895 | 0 of 128 |
| plain | both | count-weighted | 0.18711 | 0.19321 | 1.0326 | 0 of 128 | 0.22083 | 1.1802 | 0 of 128 |
| plain | both | trace-weighted | 0.18712 | 0.19322 | 1.0326 | 0 of 128 | 0.22079 | 1.1799 | 0 of 128 |
| gain | gate | unweighted | 0.99618 | 0.99647 | 1.0003 | 45 of 128 | 0.99058 | 0.9944 | 128 of 128 |
| gain | gate | count-weighted | 0.99620 | 0.99644 | 1.0002 | 45 of 128 | 0.99082 | 0.9946 | 128 of 128 |
| gain | gate | trace-weighted | 0.99621 | 0.99645 | 1.0002 | 45 of 128 | 0.99084 | 0.9946 | 128 of 128 |
| gain | up | unweighted | 0.99621 | 0.99652 | 1.0003 | 36 of 128 | 0.99066 | 0.9944 | 128 of 128 |
| gain | up | count-weighted | 0.99623 | 0.99650 | 1.0003 | 36 of 128 | 0.99093 | 0.9947 | 128 of 128 |
| gain | up | trace-weighted | 0.99624 | 0.99651 | 1.0003 | 36 of 128 | 0.99094 | 0.9947 | 128 of 128 |
| gain | both | unweighted | 0.99620 | 0.99649 | 1.0003 | 41 of 128 | 0.99062 | 0.9944 | 128 of 128 |
| gain | both | count-weighted | 0.99621 | 0.99647 | 1.0003 | 41 of 128 | 0.99087 | 0.9946 | 128 of 128 |
| gain | both | trace-weighted | 0.99622 | 0.99648 | 1.0003 | 41 of 128 | 0.99089 | 0.9946 | 128 of 128 |
| energy | gate | unweighted | 1.02739 | 1.03028 | 1.0028 | 1 of 128 | 1.03053 | 1.0031 | 0 of 128 |
| energy | gate | count-weighted | 1.02741 | 1.03021 | 1.0027 | 1 of 128 | 1.03024 | 1.0028 | 0 of 128 |
| energy | gate | trace-weighted | 1.02743 | 1.03023 | 1.0027 | 1 of 128 | 1.03026 | 1.0028 | 0 of 128 |
| energy | up | unweighted | 1.02744 | 1.03036 | 1.0028 | 0 of 128 | 1.03107 | 1.0035 | 0 of 128 |
| energy | up | count-weighted | 1.02746 | 1.03032 | 1.0028 | 0 of 128 | 1.03081 | 1.0033 | 0 of 128 |
| energy | up | trace-weighted | 1.02749 | 1.03034 | 1.0028 | 0 of 128 | 1.03083 | 1.0033 | 0 of 128 |
| energy | both | unweighted | 1.02742 | 1.03032 | 1.0028 | 0 of 128 | 1.03080 | 1.0033 | 0 of 128 |
| energy | both | count-weighted | 1.02744 | 1.03027 | 1.0028 | 0 of 128 | 1.03052 | 1.0030 | 0 of 128 |
| energy | both | trace-weighted | 1.02746 | 1.03029 | 1.0028 | 0 of 128 | 1.03054 | 1.0030 | 0 of 128 |
| ogain | gate | unweighted | 0.99890 | 0.99897 | 1.0001 | 56 of 128 | 1.00758 | 1.0087 | 0 of 128 |
| ogain | gate | count-weighted | 0.99888 | 0.99890 | 1.0000 | 56 of 128 | 1.00756 | 1.0087 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99889 | 0.99890 | 1.0000 | 56 of 128 | 1.00758 | 1.0087 | 0 of 128 |
| ogain | up | unweighted | 0.99837 | 0.99859 | 1.0002 | 54 of 128 | 1.00427 | 1.0059 | 0 of 128 |
| ogain | up | count-weighted | 0.99839 | 0.99855 | 1.0002 | 54 of 128 | 1.00423 | 1.0058 | 0 of 128 |
| ogain | up | trace-weighted | 0.99839 | 0.99855 | 1.0002 | 54 of 128 | 1.00419 | 1.0058 | 0 of 128 |
| ogain | both | unweighted | 0.99868 | 0.99881 | 1.0001 | 53 of 128 | 1.00619 | 1.0075 | 0 of 128 |
| ogain | both | count-weighted | 0.99868 | 0.99875 | 1.0001 | 53 of 128 | 1.00616 | 1.0075 | 0 of 128 |
| ogain | both | trace-weighted | 0.99868 | 0.99875 | 1.0001 | 53 of 128 | 1.00615 | 1.0075 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -4.3408e-03 | [-4.4359e-03, -4.2415e-03] |
| up | -5.9171e-03 | [-6.0272e-03, -5.8102e-03] |
| both | -4.9966e-03 | [-5.0889e-03, -4.9047e-03] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -7.6149e-03 | [-7.8001e-03, -7.4447e-03] | -3.2741e-03 | [-3.4263e-03, -3.1331e-03] |
| full | up | -1.1024e-02 | [-1.1215e-02, -1.0853e-02] | -5.1072e-03 | [-5.3084e-03, -4.9156e-03] |
| full | both | -9.0323e-03 | [-9.1962e-03, -8.8857e-03] | -4.0357e-03 | [-4.1993e-03, -3.8820e-03] |
| reg | gate | -6.7025e-03 | [-6.8586e-03, -6.5588e-03] | -2.7178e-03 | [-2.8471e-03, -2.5971e-03] |
| reg | up | -9.5495e-03 | [-9.7054e-03, -9.4094e-03] | -4.2177e-03 | [-4.3896e-03, -4.0527e-03] |
| reg | both | -7.9001e-03 | [-8.0362e-03, -7.7787e-03] | -3.3483e-03 | [-3.4890e-03, -3.2159e-03] |
| bias | gate | -1.8485e-03 | [-1.9012e-03, -1.7990e-03] | -1.7626e-04 | [-1.9693e-04, -1.5701e-04] |
| bias | up | -2.8248e-03 | [-2.8871e-03, -2.7613e-03] | -5.3922e-04 | [-5.6170e-04, -5.1717e-04] |
| bias | both | -2.2548e-03 | [-2.3032e-03, -2.2076e-03] | -3.2739e-04 | [-3.4655e-04, -3.0962e-04] |
| rest | gate | -5.7664e-03 | [-5.9126e-03, -5.6325e-03] | -3.0978e-03 | [-3.2367e-03, -2.9670e-03] |
| rest | up | -8.1995e-03 | [-8.3627e-03, -8.0536e-03] | -4.5680e-03 | [-4.7604e-03, -4.3870e-03] |
| rest | both | -6.7774e-03 | [-6.9170e-03, -6.6540e-03] | -3.7083e-03 | [-3.8620e-03, -3.5622e-03] |
| diag | gate | +9.6702e-03 | [+9.4394e-03, +9.9191e-03] | +7.2677e-03 | [+7.0445e-03, +7.5092e-03] |
| diag | up | +9.9426e-03 | [+9.7121e-03, +1.0194e-02] | +7.5354e-03 | [+7.3090e-03, +7.7791e-03] |
| diag | both | +9.8063e-03 | [+9.5764e-03, +1.0056e-02] | +7.4015e-03 | [+7.1770e-03, +7.6437e-03] |

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

In layer 6 (both halves, count-weighted), UD's full error is 0.02232 against our 0.03525 (ratio 0.633; UD - ours -1.294e-02, 95% CI [-1.333e-02, -1.253e-02]) and its diagonal error 0.06603 against our 0.06179 (ratio 1.069); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 6 (added after registration, not part of the rule): the input-mean part is 0.00552 ours against 8.958e-04 UD (ratio 0.162), the centred rest 0.02973 against 0.02142 (ratio 0.721). In layer 7 (both halves, count-weighted), UD's full error is 0.02508 against our 0.03747 (ratio 0.670; UD - ours -1.238e-02, 95% CI [-1.255e-02, -1.220e-02]) and its diagonal error 0.06595 against our 0.06161 (ratio 1.071); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 7 (added after registration, not part of the rule): the input-mean part is 0.00496 ours against 9.704e-04 UD (ratio 0.195), the centred rest 0.03250 against 0.02411 (ratio 0.742). In layer 8 (both halves, count-weighted), UD's full error is 0.02745 against our 0.03872 (ratio 0.709; UD - ours -1.127e-02, 95% CI [-1.146e-02, -1.108e-02]) and its diagonal error 0.06532 against our 0.06159 (ratio 1.061); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 8 (added after registration, not part of the rule): the input-mean part is 0.00520 ours against 0.00118 UD (ratio 0.228), the centred rest 0.03352 against 0.02627 (ratio 0.783). In layer 9 (both halves, count-weighted), UD's full error is 0.01529 against our 0.02028 (ratio 0.754; UD - ours -4.997e-03, 95% CI [-5.089e-03, -4.905e-03]) and its diagonal error 0.03416 against our 0.03175 (ratio 1.076); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 9 (added after registration, not part of the rule): the input-mean part is 0.00248 ours against 5.518e-04 UD (ratio 0.223), the centred rest 0.01781 against 0.01474 (ratio 0.828). Under the registered rule this is "supported" for both halves together, count-weighted; across the three means and three halves the branches are: supported.

## refined against ours and UD

In layer 6 (both halves, count-weighted), refined's full error is 0.01875 against our 0.03525 and UD's 0.02232 (refined - ours -1.651e-02, 95% CI [-1.694e-02, -1.607e-02]; (ours - refined) / (ours - UD) = 1.276; lower than ours in 126 of 128 experts). Bias part 5.698e-04 (ours 0.00552, UD 8.958e-04), rest 0.01818 (ours 0.02973, UD 0.02142). Regularised (the search objective) 0.02219 (ours 0.03666, UD 0.02466). Diagonal 0.08281 (ours 0.06179, UD 0.06603), plain 0.31833 (ours 0.26134, UD 0.26808). Gains (ours / UD / refined): plain 0.9963 / 0.9973 / 0.9873, energy 1.0610 / 1.0664 / 1.0761, output 0.9973 / 0.9989 / 1.0106. In layer 7 (both halves, count-weighted), refined's full error is 0.01981 against our 0.03747 and UD's 0.02508 (refined - ours -1.765e-02, 95% CI [-1.795e-02, -1.736e-02]; (ours - refined) / (ours - UD) = 1.426; lower than ours in 128 of 128 experts). Bias part 3.188e-04 (ours 0.00496, UD 9.704e-04), rest 0.01949 (ours 0.03250, UD 0.02411). Regularised (the search objective) 0.02341 (ours 0.03884, UD 0.02742). Diagonal 0.08259 (ours 0.06161, UD 0.06595), plain 0.32615 (ours 0.26274, UD 0.26966). Gains (ours / UD / refined): plain 0.9972 / 0.9983 / 0.9874, energy 1.0635 / 1.0694 / 1.0812, output 0.9954 / 0.9990 / 1.0071. In layer 8 (both halves, count-weighted), refined's full error is 0.02117 against our 0.03872 and UD's 0.02745 (refined - ours -1.755e-02, 95% CI [-1.785e-02, -1.726e-02]; (ours - refined) / (ours - UD) = 1.557; lower than ours in 128 of 128 experts). Bias part 3.739e-04 (ours 0.00520, UD 0.00118), rest 0.02080 (ours 0.03352, UD 0.02627). Regularised (the search objective) 0.02471 (ours 0.04006, UD 0.02968). Diagonal 0.08104 (ours 0.06159, UD 0.06532), plain 0.31746 (ours 0.26228, UD 0.26815). Gains (ours / UD / refined): plain 0.9975 / 0.9984 / 0.9890, energy 1.0638 / 1.0688 / 1.0788, output 0.9958 / 0.9990 / 1.0078. In layer 9 (both halves, count-weighted), refined's full error is 0.01125 against our 0.02028 and UD's 0.01529 (refined - ours -9.032e-03, 95% CI [-9.196e-03, -8.886e-03]; (ours - refined) / (ours - UD) = 1.808; lower than ours in 128 of 128 experts). Bias part 2.244e-04 (ours 0.00248, UD 5.518e-04), rest 0.01103 (ours 0.01781, UD 0.01474). Regularised (the search objective) 0.01307 (ours 0.02097, UD 0.01642). Diagonal 0.04156 (ours 0.03175, UD 0.03416), plain 0.22083 (ours 0.18711, UD 0.19321). Gains (ours / UD / refined): plain 0.9962 / 0.9965 / 0.9909, energy 1.0274 / 1.0303 / 1.0305, output 0.9987 / 0.9987 / 1.0062.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined (count-w.) | refined / ours | refined / UD |
|---|---|---|---|---|---|---|---|---|
| 6 | < 2304 | 2 | 0.000 | 0.02841 | 0.01660 | 0.02841 | 1.000 | 1.711 |
| 6 | >= 2304 | 126 | 1.000 | 0.03526 | 0.02232 | 0.01874 | 0.532 | 0.840 |
| 7 | >= 2304 | 128 | 1.000 | 0.03747 | 0.02508 | 0.01981 | 0.529 | 0.790 |
| 8 | >= 2304 | 128 | 1.000 | 0.03872 | 0.02745 | 0.02117 | 0.547 | 0.771 |
| 9 | >= 2304 | 128 | 1.000 | 0.02028 | 0.01529 | 0.01125 | 0.555 | 0.736 |

Refinement log `models/refined/gu-refined-mix-6-7-8-9.gguf.json` (scripts/refine-codes.py; wall time 7056 s for 512 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 126 | 2 | 4: 126 | 57296446 | 2611017 of 2612736 | 5562 | 5.21e-10 | 3.62e-10 | yes |
| 7 | 128 | 0 | 4: 128 | 62973078 | 2653417 of 2654208 | 6455 | 4.10e-10 | 3.43e-10 | yes |
| 8 | 128 | 0 | 4: 128 | 61598882 | 2653213 of 2654208 | 7341 | 3.49e-10 | 3.52e-10 | yes |
| 9 | 128 | 0 | 4: 128 | 80015493 | 2653786 of 2654208 | 8753 | 1.90e-10 | 1.56e-10 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 6: max abs(log - here) on reg 5.03e-10 before, 3.60e-10 after; layer 7: max abs(log - here) on reg 3.91e-10 before, 3.29e-10 after; layer 8: max abs(log - here) on reg 3.36e-10 before, 3.48e-10 after; layer 9: max abs(log - here) on reg 1.80e-10 before, 1.51e-10 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304').
