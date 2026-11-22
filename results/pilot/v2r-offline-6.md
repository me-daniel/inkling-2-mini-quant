# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined = `models/refined/gu-refined-6.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/v2r-offline-6.tsv`.
Grams: `models/grams/gu-6-20` (patched llama-imatrix on cand.gguf at P = 3, first 100 chunks of `data/calib-wt.txt`, pooled over all visits), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

## Layer 6 (IQ2_S; UD `.scale` present, min 3.867e-05, max 0.0001495)

- Gram layout: expert 23: max |strict lower| = 0, max |strict upper| = 1.83e+05
- Gram layout: expert 105: max |strict lower| = 0, max |strict upper| = 8.698e+04
- Gram layout: expert 53: max |strict lower| = 0, max |strict upper| = 7.821e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 5.84e-09; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.6.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.313552 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 2, median 5615, max 52992

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03091 | 0.01994 | 0.6452 | 128 of 128 | 0.01895 | 0.6131 | 96 of 128 |
| full | gate | count-weighted | 0.03025 | 0.01957 | 0.6468 | 128 of 128 | 0.01564 | 0.5169 | 96 of 128 |
| full | gate | trace-weighted | 0.03016 | 0.01951 | 0.6469 | 128 of 128 | 0.01563 | 0.5182 | 96 of 128 |
| full | up | unweighted | 0.04257 | 0.02732 | 0.6417 | 128 of 128 | 0.02508 | 0.5890 | 96 of 128 |
| full | up | count-weighted | 0.04167 | 0.02685 | 0.6443 | 128 of 128 | 0.02024 | 0.4859 | 96 of 128 |
| full | up | trace-weighted | 0.04166 | 0.02685 | 0.6446 | 128 of 128 | 0.02029 | 0.4870 | 96 of 128 |
| full | both | unweighted | 0.03557 | 0.02290 | 0.6437 | 128 of 128 | 0.02140 | 0.6015 | 96 of 128 |
| full | both | count-weighted | 0.03487 | 0.02252 | 0.6457 | 128 of 128 | 0.01750 | 0.5018 | 96 of 128 |
| full | both | trace-weighted | 0.03480 | 0.02248 | 0.6459 | 128 of 128 | 0.01750 | 0.5030 | 96 of 128 |
| reg | gate | unweighted | 0.03231 | 0.02207 | 0.6832 | 128 of 128 | 0.02189 | 0.6777 | 96 of 128 |
| reg | gate | count-weighted | 0.03169 | 0.02170 | 0.6848 | 128 of 128 | 0.01888 | 0.5959 | 96 of 128 |
| reg | gate | trace-weighted | 0.03159 | 0.02164 | 0.6849 | 128 of 128 | 0.01886 | 0.5969 | 96 of 128 |
| reg | up | unweighted | 0.04379 | 0.02978 | 0.6800 | 128 of 128 | 0.02874 | 0.6564 | 96 of 128 |
| reg | up | count-weighted | 0.04293 | 0.02930 | 0.6825 | 128 of 128 | 0.02440 | 0.5683 | 96 of 128 |
| reg | up | trace-weighted | 0.04293 | 0.02931 | 0.6828 | 128 of 128 | 0.02443 | 0.5692 | 96 of 128 |
| reg | both | unweighted | 0.03695 | 0.02519 | 0.6818 | 128 of 128 | 0.02466 | 0.6674 | 96 of 128 |
| reg | both | count-weighted | 0.03629 | 0.02481 | 0.6838 | 128 of 128 | 0.02113 | 0.5824 | 96 of 128 |
| reg | both | trace-weighted | 0.03622 | 0.02477 | 0.6840 | 128 of 128 | 0.02113 | 0.5834 | 96 of 128 |
| bias | gate | unweighted | 0.00621 | 0.00142 | 0.2294 | 128 of 128 | 0.00258 | 0.4150 | 96 of 128 |
| bias | gate | count-weighted | 0.00554 | 0.00112 | 0.2012 | 128 of 128 | 0.00100 | 0.1806 | 96 of 128 |
| bias | gate | trace-weighted | 0.00552 | 0.00110 | 0.1997 | 128 of 128 | 0.00101 | 0.1822 | 96 of 128 |
| bias | up | unweighted | 0.00877 | 0.00199 | 0.2267 | 128 of 128 | 0.00310 | 0.3534 | 96 of 128 |
| bias | up | count-weighted | 0.00775 | 0.00152 | 0.1964 | 128 of 128 | 6.849e-04 | 0.0883 | 96 of 128 |
| bias | up | trace-weighted | 0.00774 | 0.00151 | 0.1954 | 128 of 128 | 6.917e-04 | 0.0894 | 96 of 128 |
| bias | both | unweighted | 0.00722 | 0.00165 | 0.2281 | 128 of 128 | 0.00278 | 0.3847 | 96 of 128 |
| bias | both | count-weighted | 0.00643 | 0.00128 | 0.1988 | 128 of 128 | 8.689e-04 | 0.1351 | 96 of 128 |
| bias | both | trace-weighted | 0.00641 | 0.00127 | 0.1976 | 128 of 128 | 8.738e-04 | 0.1363 | 96 of 128 |
| rest | gate | unweighted | 0.02470 | 0.01852 | 0.7497 | 128 of 128 | 0.01637 | 0.6629 | 96 of 128 |
| rest | gate | count-weighted | 0.02471 | 0.01845 | 0.7468 | 128 of 128 | 0.01464 | 0.5924 | 96 of 128 |
| rest | gate | trace-weighted | 0.02464 | 0.01841 | 0.7471 | 128 of 128 | 0.01462 | 0.5934 | 96 of 128 |
| rest | up | unweighted | 0.03381 | 0.02533 | 0.7494 | 128 of 128 | 0.02198 | 0.6501 | 96 of 128 |
| rest | up | count-weighted | 0.03391 | 0.02532 | 0.7467 | 128 of 128 | 0.01956 | 0.5768 | 96 of 128 |
| rest | up | trace-weighted | 0.03392 | 0.02534 | 0.7471 | 128 of 128 | 0.01960 | 0.5777 | 96 of 128 |
| rest | both | unweighted | 0.02835 | 0.02125 | 0.7496 | 128 of 128 | 0.01862 | 0.6567 | 96 of 128 |
| rest | both | count-weighted | 0.02844 | 0.02124 | 0.7468 | 128 of 128 | 0.01663 | 0.5847 | 96 of 128 |
| rest | both | trace-weighted | 0.02839 | 0.02121 | 0.7471 | 128 of 128 | 0.01663 | 0.5857 | 96 of 128 |
| diag | gate | unweighted | 0.06175 | 0.06570 | 1.0640 | 0 of 128 | 0.08207 | 1.3291 | 0 of 128 |
| diag | gate | count-weighted | 0.06203 | 0.06604 | 1.0647 | 0 of 128 | 0.08590 | 1.3849 | 0 of 128 |
| diag | gate | trace-weighted | 0.06202 | 0.06603 | 1.0647 | 0 of 128 | 0.08583 | 1.3841 | 0 of 128 |
| diag | up | unweighted | 0.06189 | 0.06585 | 1.0640 | 0 of 128 | 0.08273 | 1.3367 | 0 of 128 |
| diag | up | count-weighted | 0.06211 | 0.06613 | 1.0647 | 0 of 128 | 0.08655 | 1.3934 | 0 of 128 |
| diag | up | trace-weighted | 0.06210 | 0.06612 | 1.0647 | 0 of 128 | 0.08649 | 1.3926 | 0 of 128 |
| diag | both | unweighted | 0.06182 | 0.06577 | 1.0640 | 0 of 128 | 0.08240 | 1.3329 | 0 of 128 |
| diag | both | count-weighted | 0.06207 | 0.06609 | 1.0647 | 0 of 128 | 0.08623 | 1.3892 | 0 of 128 |
| diag | both | trace-weighted | 0.06206 | 0.06608 | 1.0647 | 0 of 128 | 0.08616 | 1.3883 | 0 of 128 |
| plain | gate | unweighted | 0.26149 | 0.26828 | 1.0259 | 0 of 128 | 0.31370 | 1.1996 | 0 of 128 |
| plain | gate | count-weighted | 0.26139 | 0.26816 | 1.0259 | 0 of 128 | 0.32408 | 1.2398 | 0 of 128 |
| plain | gate | trace-weighted | 0.26144 | 0.26821 | 1.0259 | 0 of 128 | 0.32426 | 1.2403 | 0 of 128 |
| plain | up | unweighted | 0.26148 | 0.26826 | 1.0259 | 0 of 128 | 0.31494 | 1.2044 | 0 of 128 |
| plain | up | count-weighted | 0.26137 | 0.26813 | 1.0259 | 0 of 128 | 0.32558 | 1.2457 | 0 of 128 |
| plain | up | trace-weighted | 0.26142 | 0.26818 | 1.0258 | 0 of 128 | 0.32576 | 1.2461 | 0 of 128 |
| plain | both | unweighted | 0.26149 | 0.26827 | 1.0259 | 0 of 128 | 0.31432 | 1.2020 | 0 of 128 |
| plain | both | count-weighted | 0.26138 | 0.26814 | 1.0259 | 0 of 128 | 0.32483 | 1.2428 | 0 of 128 |
| plain | both | trace-weighted | 0.26143 | 0.26819 | 1.0259 | 0 of 128 | 0.32501 | 1.2432 | 0 of 128 |
| gain | gate | unweighted | 0.99629 | 0.99714 | 1.0009 | 9 of 128 | 0.98806 | 0.9917 | 96 of 128 |
| gain | gate | count-weighted | 0.99629 | 0.99721 | 1.0009 | 9 of 128 | 0.98652 | 0.9902 | 96 of 128 |
| gain | gate | trace-weighted | 0.99629 | 0.99720 | 1.0009 | 9 of 128 | 0.98648 | 0.9902 | 96 of 128 |
| gain | up | unweighted | 0.99630 | 0.99717 | 1.0009 | 11 of 128 | 0.98788 | 0.9915 | 96 of 128 |
| gain | up | count-weighted | 0.99633 | 0.99725 | 1.0009 | 11 of 128 | 0.98631 | 0.9899 | 96 of 128 |
| gain | up | trace-weighted | 0.99633 | 0.99725 | 1.0009 | 11 of 128 | 0.98629 | 0.9899 | 96 of 128 |
| gain | both | unweighted | 0.99629 | 0.99716 | 1.0009 | 11 of 128 | 0.98797 | 0.9916 | 96 of 128 |
| gain | both | count-weighted | 0.99631 | 0.99723 | 1.0009 | 11 of 128 | 0.98641 | 0.9901 | 96 of 128 |
| gain | both | trace-weighted | 0.99631 | 0.99722 | 1.0009 | 11 of 128 | 0.98639 | 0.9900 | 96 of 128 |
| energy | gate | unweighted | 1.06096 | 1.06626 | 1.0050 | 0 of 128 | 1.07549 | 1.0137 | 0 of 128 |
| energy | gate | count-weighted | 1.06091 | 1.06633 | 1.0051 | 0 of 128 | 1.07821 | 1.0163 | 0 of 128 |
| energy | gate | trace-weighted | 1.06093 | 1.06633 | 1.0051 | 0 of 128 | 1.07825 | 1.0163 | 0 of 128 |
| energy | up | unweighted | 1.06098 | 1.06631 | 1.0050 | 0 of 128 | 1.07595 | 1.0141 | 0 of 128 |
| energy | up | count-weighted | 1.06098 | 1.06640 | 1.0051 | 0 of 128 | 1.07877 | 1.0168 | 0 of 128 |
| energy | up | trace-weighted | 1.06101 | 1.06641 | 1.0051 | 0 of 128 | 1.07885 | 1.0168 | 0 of 128 |
| energy | both | unweighted | 1.06097 | 1.06629 | 1.0050 | 0 of 128 | 1.07572 | 1.0139 | 0 of 128 |
| energy | both | count-weighted | 1.06095 | 1.06637 | 1.0051 | 0 of 128 | 1.07849 | 1.0165 | 0 of 128 |
| energy | both | trace-weighted | 1.06097 | 1.06637 | 1.0051 | 0 of 128 | 1.07855 | 1.0166 | 0 of 128 |
| ogain | gate | unweighted | 0.99919 | 0.99910 | 0.9999 | 66 of 128 | 1.01125 | 1.0121 | 0 of 128 |
| ogain | gate | count-weighted | 0.99903 | 0.99900 | 1.0000 | 66 of 128 | 1.01416 | 1.0151 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99899 | 0.99900 | 1.0000 | 66 of 128 | 1.01413 | 1.0152 | 0 of 128 |
| ogain | up | unweighted | 0.99580 | 0.99860 | 1.0028 | 3 of 128 | 1.00704 | 1.0113 | 0 of 128 |
| ogain | up | count-weighted | 0.99542 | 0.99851 | 1.0031 | 3 of 128 | 1.00955 | 1.0142 | 0 of 128 |
| ogain | up | trace-weighted | 0.99542 | 0.99852 | 1.0031 | 3 of 128 | 1.00954 | 1.0142 | 0 of 128 |
| ogain | both | unweighted | 0.99776 | 0.99890 | 1.0011 | 30 of 128 | 1.00948 | 1.0117 | 0 of 128 |
| ogain | both | count-weighted | 0.99750 | 0.99879 | 1.0013 | 30 of 128 | 1.01222 | 1.0148 | 0 of 128 |
| ogain | both | trace-weighted | 0.99747 | 0.99880 | 1.0013 | 30 of 128 | 1.01220 | 1.0148 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.0685e-02 | [-1.1155e-02, -1.0230e-02] |
| up | -1.4819e-02 | [-1.5154e-02, -1.4490e-02] |
| both | -1.2356e-02 | [-1.2757e-02, -1.1974e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.4613e-02 | [-1.5232e-02, -1.3988e-02] | -3.9283e-03 | [-4.3089e-03, -3.5655e-03] |
| full | up | -2.1421e-02 | [-2.1944e-02, -2.0927e-02] | -6.6025e-03 | [-7.0829e-03, -6.1469e-03] |
| full | both | -1.7374e-02 | [-1.7914e-02, -1.6837e-02] | -5.0188e-03 | [-5.4329e-03, -4.6313e-03] |
| reg | gate | -1.2805e-02 | [-1.3331e-02, -1.2272e-02] | -2.8179e-03 | [-3.1447e-03, -2.5070e-03] |
| reg | up | -1.8536e-02 | [-1.8967e-02, -1.8124e-02] | -4.9042e-03 | [-5.3111e-03, -4.5085e-03] |
| reg | both | -1.5156e-02 | [-1.5608e-02, -1.4705e-02] | -3.6794e-03 | [-4.0362e-03, -3.3411e-03] |
| bias | gate | -4.5427e-03 | [-4.7804e-03, -4.3141e-03] | -1.1386e-04 | [-2.0915e-04, -1.6724e-05] |
| bias | up | -7.0679e-03 | [-7.2788e-03, -6.8593e-03] | -8.3754e-04 | [-9.5131e-04, -7.1464e-04] |
| bias | both | -5.5651e-03 | [-5.7749e-03, -5.3655e-03] | -4.1043e-04 | [-5.1075e-04, -3.0535e-04] |
| rest | gate | -1.0070e-02 | [-1.0487e-02, -9.6484e-03] | -3.8144e-03 | [-4.1208e-03, -3.5334e-03] |
| rest | up | -1.4353e-02 | [-1.4747e-02, -1.3991e-02] | -5.7649e-03 | [-6.1507e-03, -5.4094e-03] |
| rest | both | -1.1809e-02 | [-1.2190e-02, -1.1426e-02] | -4.6084e-03 | [-4.9429e-03, -4.3041e-03] |
| diag | gate | +2.3874e-02 | [+2.3263e-02, +2.4590e-02] | +1.9860e-02 | [+1.9254e-02, +2.0578e-02] |
| diag | up | +2.4438e-02 | [+2.3793e-02, +2.5197e-02] | +2.0420e-02 | [+1.9778e-02, +2.1178e-02] |
| diag | both | +2.4155e-02 | [+2.3528e-02, +2.4894e-02] | +2.0139e-02 | [+1.9515e-02, +2.0877e-02] |

## Reading (amendment 7 rule, applied literally)

Rule: supported if UD is lower than ours on the full error and higher or equal on the diagonal one in both layers; if UD is also worse on the full error, layer-local output error does not explain UD's edge.

| mean | half | layer 6 full UD < ours | layer 6 diag UD >= ours | branch |
|---|---|---|---|---|
| unweighted | gate | yes | yes | supported |
| unweighted | up | yes | yes | supported |
| unweighted | both | yes | yes | supported |
| count-weighted | gate | yes | yes | supported |
| count-weighted | up | yes | yes | supported |
| count-weighted | both | yes | yes | supported |
| trace-weighted | gate | yes | yes | supported |
| trace-weighted | up | yes | yes | supported |
| trace-weighted | both | yes | yes | supported |

In layer 6 (both halves, count-weighted), UD's full error is 0.02252 against our 0.03487 (ratio 0.646; UD - ours -1.236e-02, 95% CI [-1.276e-02, -1.197e-02]) and its diagonal error 0.06609 against our 0.06207 (ratio 1.065); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 6 (added after registration, not part of the rule): the input-mean part is 0.00643 ours against 0.00128 UD (ratio 0.199), the centred rest 0.02844 against 0.02124 (ratio 0.747). Under the registered rule this is "supported" for both halves together, count-weighted; across the three means and three halves the branches are: supported.

## refined against ours and UD

In layer 6 (both halves, count-weighted), refined's full error is 0.01750 against our 0.03487 and UD's 0.02252 (refined - ours -1.737e-02, 95% CI [-1.791e-02, -1.684e-02]; (ours - refined) / (ours - UD) = 1.406; lower than ours in 96 of 128 experts). Bias part 8.689e-04 (ours 0.00643, UD 0.00128), rest 0.01663 (ours 0.02844, UD 0.02124). Regularised (the search objective) 0.02113 (ours 0.03629, UD 0.02481). Diagonal 0.08623 (ours 0.06207, UD 0.06609), plain 0.32483 (ours 0.26138, UD 0.26814). Gains (ours / UD / refined): plain 0.9963 / 0.9972 / 0.9864, energy 1.0609 / 1.0664 / 1.0785, output 0.9975 / 0.9988 / 1.0122.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined (count-w.) | refined / ours | refined / UD |
|---|---|---|---|---|---|---|---|---|
| 6 | < 2304 | 32 | 0.021 | 0.03889 | 0.02507 | 0.03889 | 1.000 | 1.551 |
| 6 | >= 2304 | 96 | 0.979 | 0.03479 | 0.02246 | 0.01704 | 0.490 | 0.758 |

Refinement log `models/refined/gu-refined-6.gguf.json` (scripts/refine-codes.py; wall time 1334 s for 128 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 96 | 32 | 4: 96 | 44003994 | 1989531 of 1990656 | 5213 | 4.86e-10 | 4.63e-10 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 6: max abs(log - here) on reg 4.67e-10 before, 4.62e-10 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304').
