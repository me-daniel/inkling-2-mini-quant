# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a)

Written by `scripts/gram-error.py` (Addendum H amendment 7). Per-expert rows: `results/pilot/gram-error.tsv`.
Grams: `models/grams/gu-6-20` (patched llama-imatrix on cand.gguf at P = 3, first 100 chunks of `data/calib-wt.txt`, pooled over all visits), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

## Layer 6 (IQ2_S; UD `.scale` present, min 3.867e-05, max 0.0001495)

- Gram layout: expert 23: max |strict lower| = 0, max |strict upper| = 1.83e+05
- Gram layout: expert 105: max |strict lower| = 0, max |strict upper| = 8.698e+04
- Gram layout: expert 53: max |strict lower| = 0, max |strict upper| = 7.821e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 5.84e-09; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.6.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.313552 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 2, median 5615, max 52992

| metric | half | mean | ours | UD | UD / ours | UD < ours |
|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03091 | 0.01994 | 0.6452 | 128 of 128 |
| full | gate | count-weighted | 0.03025 | 0.01957 | 0.6468 | 128 of 128 |
| full | gate | trace-weighted | 0.03016 | 0.01951 | 0.6469 | 128 of 128 |
| full | up | unweighted | 0.04257 | 0.02732 | 0.6417 | 128 of 128 |
| full | up | count-weighted | 0.04167 | 0.02685 | 0.6443 | 128 of 128 |
| full | up | trace-weighted | 0.04166 | 0.02685 | 0.6446 | 128 of 128 |
| full | both | unweighted | 0.03557 | 0.02290 | 0.6437 | 128 of 128 |
| full | both | count-weighted | 0.03487 | 0.02252 | 0.6457 | 128 of 128 |
| full | both | trace-weighted | 0.03480 | 0.02248 | 0.6459 | 128 of 128 |
| bias | gate | unweighted | 0.00621 | 0.00142 | 0.2294 | 128 of 128 |
| bias | gate | count-weighted | 0.00554 | 0.00112 | 0.2012 | 128 of 128 |
| bias | gate | trace-weighted | 0.00552 | 0.00110 | 0.1997 | 128 of 128 |
| bias | up | unweighted | 0.00877 | 0.00199 | 0.2267 | 128 of 128 |
| bias | up | count-weighted | 0.00775 | 0.00152 | 0.1964 | 128 of 128 |
| bias | up | trace-weighted | 0.00774 | 0.00151 | 0.1954 | 128 of 128 |
| bias | both | unweighted | 0.00722 | 0.00165 | 0.2281 | 128 of 128 |
| bias | both | count-weighted | 0.00643 | 0.00128 | 0.1988 | 128 of 128 |
| bias | both | trace-weighted | 0.00641 | 0.00127 | 0.1976 | 128 of 128 |
| rest | gate | unweighted | 0.02470 | 0.01852 | 0.7497 | 128 of 128 |
| rest | gate | count-weighted | 0.02471 | 0.01845 | 0.7468 | 128 of 128 |
| rest | gate | trace-weighted | 0.02464 | 0.01841 | 0.7471 | 128 of 128 |
| rest | up | unweighted | 0.03381 | 0.02533 | 0.7494 | 128 of 128 |
| rest | up | count-weighted | 0.03391 | 0.02532 | 0.7467 | 128 of 128 |
| rest | up | trace-weighted | 0.03392 | 0.02534 | 0.7471 | 128 of 128 |
| rest | both | unweighted | 0.02835 | 0.02125 | 0.7496 | 128 of 128 |
| rest | both | count-weighted | 0.02844 | 0.02124 | 0.7468 | 128 of 128 |
| rest | both | trace-weighted | 0.02839 | 0.02121 | 0.7471 | 128 of 128 |
| diag | gate | unweighted | 0.06175 | 0.06570 | 1.0640 | 0 of 128 |
| diag | gate | count-weighted | 0.06203 | 0.06604 | 1.0647 | 0 of 128 |
| diag | gate | trace-weighted | 0.06202 | 0.06603 | 1.0647 | 0 of 128 |
| diag | up | unweighted | 0.06189 | 0.06585 | 1.0640 | 0 of 128 |
| diag | up | count-weighted | 0.06211 | 0.06613 | 1.0647 | 0 of 128 |
| diag | up | trace-weighted | 0.06210 | 0.06612 | 1.0647 | 0 of 128 |
| diag | both | unweighted | 0.06182 | 0.06577 | 1.0640 | 0 of 128 |
| diag | both | count-weighted | 0.06207 | 0.06609 | 1.0647 | 0 of 128 |
| diag | both | trace-weighted | 0.06206 | 0.06608 | 1.0647 | 0 of 128 |
| plain | gate | unweighted | 0.26149 | 0.26828 | 1.0259 | 0 of 128 |
| plain | gate | count-weighted | 0.26139 | 0.26816 | 1.0259 | 0 of 128 |
| plain | gate | trace-weighted | 0.26144 | 0.26821 | 1.0259 | 0 of 128 |
| plain | up | unweighted | 0.26148 | 0.26826 | 1.0259 | 0 of 128 |
| plain | up | count-weighted | 0.26137 | 0.26813 | 1.0259 | 0 of 128 |
| plain | up | trace-weighted | 0.26142 | 0.26818 | 1.0258 | 0 of 128 |
| plain | both | unweighted | 0.26149 | 0.26827 | 1.0259 | 0 of 128 |
| plain | both | count-weighted | 0.26138 | 0.26814 | 1.0259 | 0 of 128 |
| plain | both | trace-weighted | 0.26143 | 0.26819 | 1.0259 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.0685e-02 | [-1.1155e-02, -1.0230e-02] |
| up | -1.4819e-02 | [-1.5154e-02, -1.4490e-02] |
| both | -1.2356e-02 | [-1.2757e-02, -1.1974e-02] |

## Layer 20 (IQ3_XXS; UD `.scale` absent)

- Gram layout: expert 102: max |strict lower| = 0, max |strict upper| = 3.719e+04
- Gram layout: expert 40: max |strict lower| = 0, max |strict upper| = 3.206e+04
- Gram layout: expert 120: max |strict lower| = 0, max |strict upper| = 3.096e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 4.16e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.20.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = -0.000000 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 1, median 1797, max 20118

| metric | half | mean | ours | UD | UD / ours | UD < ours |
|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.02301 | 0.01952 | 0.8485 | 128 of 128 |
| full | gate | count-weighted | 0.02165 | 0.01794 | 0.8284 | 128 of 128 |
| full | gate | trace-weighted | 0.02178 | 0.01811 | 0.8314 | 128 of 128 |
| full | up | unweighted | 0.02917 | 0.02471 | 0.8470 | 128 of 128 |
| full | up | count-weighted | 0.02896 | 0.02397 | 0.8277 | 128 of 128 |
| full | up | trace-weighted | 0.02903 | 0.02411 | 0.8306 | 128 of 128 |
| full | both | unweighted | 0.02565 | 0.02175 | 0.8479 | 128 of 128 |
| full | both | count-weighted | 0.02472 | 0.02047 | 0.8281 | 128 of 128 |
| full | both | trace-weighted | 0.02483 | 0.02064 | 0.8311 | 128 of 128 |
| bias | gate | unweighted | 0.00426 | 0.00206 | 0.4831 | 128 of 128 |
| bias | gate | count-weighted | 0.00368 | 0.00149 | 0.4043 | 128 of 128 |
| bias | gate | trace-weighted | 0.00368 | 0.00150 | 0.4067 | 128 of 128 |
| bias | up | unweighted | 0.00544 | 0.00262 | 0.4817 | 128 of 128 |
| bias | up | count-weighted | 0.00490 | 0.00199 | 0.4061 | 128 of 128 |
| bias | up | trace-weighted | 0.00488 | 0.00199 | 0.4084 | 128 of 128 |
| bias | both | unweighted | 0.00477 | 0.00230 | 0.4824 | 128 of 128 |
| bias | both | count-weighted | 0.00419 | 0.00170 | 0.4053 | 128 of 128 |
| bias | both | trace-weighted | 0.00418 | 0.00170 | 0.4076 | 128 of 128 |
| rest | gate | unweighted | 0.01875 | 0.01746 | 0.9317 | 124 of 128 |
| rest | gate | count-weighted | 0.01797 | 0.01645 | 0.9153 | 124 of 128 |
| rest | gate | trace-weighted | 0.01811 | 0.01662 | 0.9177 | 124 of 128 |
| rest | up | unweighted | 0.02373 | 0.02208 | 0.9308 | 124 of 128 |
| rest | up | count-weighted | 0.02406 | 0.02198 | 0.9135 | 124 of 128 |
| rest | up | trace-weighted | 0.02415 | 0.02212 | 0.9158 | 124 of 128 |
| rest | both | unweighted | 0.02088 | 0.01945 | 0.9313 | 124 of 128 |
| rest | both | count-weighted | 0.02053 | 0.01877 | 0.9145 | 124 of 128 |
| rest | both | trace-weighted | 0.02065 | 0.01893 | 0.9168 | 124 of 128 |
| diag | gate | unweighted | 0.03304 | 0.03546 | 1.0733 | 0 of 128 |
| diag | gate | count-weighted | 0.03299 | 0.03541 | 1.0733 | 0 of 128 |
| diag | gate | trace-weighted | 0.03299 | 0.03541 | 1.0733 | 0 of 128 |
| diag | up | unweighted | 0.03305 | 0.03548 | 1.0734 | 0 of 128 |
| diag | up | count-weighted | 0.03301 | 0.03543 | 1.0734 | 0 of 128 |
| diag | up | trace-weighted | 0.03301 | 0.03543 | 1.0734 | 0 of 128 |
| diag | both | unweighted | 0.03305 | 0.03547 | 1.0733 | 0 of 128 |
| diag | both | count-weighted | 0.03300 | 0.03542 | 1.0734 | 0 of 128 |
| diag | both | trace-weighted | 0.03300 | 0.03542 | 1.0734 | 0 of 128 |
| plain | gate | unweighted | 0.18317 | 0.18961 | 1.0352 | 0 of 128 |
| plain | gate | count-weighted | 0.18302 | 0.18952 | 1.0355 | 0 of 128 |
| plain | gate | trace-weighted | 0.18302 | 0.18952 | 1.0355 | 0 of 128 |
| plain | up | unweighted | 0.18316 | 0.18962 | 1.0353 | 0 of 128 |
| plain | up | count-weighted | 0.18302 | 0.18953 | 1.0356 | 0 of 128 |
| plain | up | trace-weighted | 0.18302 | 0.18953 | 1.0356 | 0 of 128 |
| plain | both | unweighted | 0.18316 | 0.18961 | 1.0352 | 0 of 128 |
| plain | both | count-weighted | 0.18302 | 0.18953 | 1.0356 | 0 of 128 |
| plain | both | trace-weighted | 0.18302 | 0.18953 | 1.0356 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -3.7147e-03 | [-3.8593e-03, -3.5630e-03] |
| up | -4.9896e-03 | [-5.1732e-03, -4.7856e-03] |
| both | -4.2483e-03 | [-4.3965e-03, -4.0870e-03] |

## Reading (amendment 7 rule, applied literally)

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
