# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined = `models/refined/gu-refined-23.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/v2r-offline-23.tsv`.
Grams: `models/grams/gu-23` (capture settings in its log, not restated here), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

## Layer 23 (IQ3_XXS; UD `.scale` absent)

- Gram layout: expert 4: max |strict lower| = 0, max |strict upper| = 3.395e+05
- Gram layout: expert 122: max |strict lower| = 0, max |strict upper| = 2.403e+05
- Gram layout: expert 39: max |strict lower| = 0, max |strict upper| = 1.986e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 1.09e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.23.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = -0.000000 (must be >= 0 up to rounding)
- Experts with tokens: 127 of 128; tokens per expert min 0, median 913, max 31411

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.02031 | 0.01579 | 0.7774 | 127 of 127 | 0.01663 | 0.8188 | 46 of 127 |
| full | gate | count-weighted | 0.01760 | 0.01320 | 0.7502 | 127 of 127 | 0.00896 | 0.5089 | 46 of 127 |
| full | gate | trace-weighted | 0.01766 | 0.01326 | 0.7507 | 127 of 127 | 0.00902 | 0.5105 | 46 of 127 |
| full | up | unweighted | 0.02497 | 0.01924 | 0.7704 | 127 of 127 | 0.01996 | 0.7993 | 46 of 127 |
| full | up | count-weighted | 0.02320 | 0.01723 | 0.7426 | 127 of 127 | 0.01125 | 0.4849 | 46 of 127 |
| full | up | trace-weighted | 0.02322 | 0.01725 | 0.7429 | 127 of 127 | 0.01128 | 0.4858 | 46 of 127 |
| full | both | unweighted | 0.02181 | 0.01690 | 0.7750 | 127 of 127 | 0.01767 | 0.8104 | 46 of 127 |
| full | both | count-weighted | 0.01965 | 0.01467 | 0.7469 | 127 of 127 | 0.00977 | 0.4973 | 46 of 127 |
| full | both | trace-weighted | 0.01970 | 0.01472 | 0.7473 | 127 of 127 | 0.00982 | 0.4987 | 46 of 127 |
| reg | gate | unweighted | 0.02088 | 0.01679 | 0.8044 | 127 of 127 | 0.01769 | 0.8475 | 46 of 127 |
| reg | gate | count-weighted | 0.01827 | 0.01425 | 0.7800 | 127 of 127 | 0.01075 | 0.5884 | 46 of 127 |
| reg | gate | trace-weighted | 0.01834 | 0.01431 | 0.7805 | 127 of 127 | 0.01081 | 0.5897 | 46 of 127 |
| reg | up | unweighted | 0.02539 | 0.02027 | 0.7984 | 127 of 127 | 0.02111 | 0.8314 | 46 of 127 |
| reg | up | count-weighted | 0.02370 | 0.01832 | 0.7732 | 127 of 127 | 0.01344 | 0.5673 | 46 of 127 |
| reg | up | trace-weighted | 0.02372 | 0.01835 | 0.7735 | 127 of 127 | 0.01347 | 0.5681 | 46 of 127 |
| reg | both | unweighted | 0.02237 | 0.01794 | 0.8023 | 127 of 127 | 0.01880 | 0.8405 | 46 of 127 |
| reg | both | count-weighted | 0.02030 | 0.01577 | 0.7770 | 127 of 127 | 0.01173 | 0.5782 | 46 of 127 |
| reg | both | trace-weighted | 0.02035 | 0.01582 | 0.7774 | 127 of 127 | 0.01179 | 0.5793 | 46 of 127 |
| bias | gate | unweighted | 0.00533 | 0.00216 | 0.4049 | 127 of 127 | 0.00405 | 0.7605 | 46 of 127 |
| bias | gate | count-weighted | 0.00436 | 0.00145 | 0.3325 | 127 of 127 | 0.00115 | 0.2647 | 46 of 127 |
| bias | gate | trace-weighted | 0.00437 | 0.00145 | 0.3324 | 127 of 127 | 0.00116 | 0.2653 | 46 of 127 |
| bias | up | unweighted | 0.00664 | 0.00263 | 0.3959 | 127 of 127 | 0.00478 | 0.7188 | 46 of 127 |
| bias | up | count-weighted | 0.00596 | 0.00197 | 0.3295 | 127 of 127 | 0.00123 | 0.2057 | 46 of 127 |
| bias | up | trace-weighted | 0.00596 | 0.00196 | 0.3293 | 127 of 127 | 0.00123 | 0.2057 | 46 of 127 |
| bias | both | unweighted | 0.00574 | 0.00230 | 0.4011 | 127 of 127 | 0.00426 | 0.7418 | 46 of 127 |
| bias | both | count-weighted | 0.00495 | 0.00164 | 0.3311 | 127 of 127 | 0.00116 | 0.2350 | 46 of 127 |
| bias | both | trace-weighted | 0.00495 | 0.00164 | 0.3310 | 127 of 127 | 0.00117 | 0.2355 | 46 of 127 |
| rest | gate | unweighted | 0.01498 | 0.01363 | 0.9098 | 123 of 127 | 0.01258 | 0.8396 | 46 of 127 |
| rest | gate | count-weighted | 0.01324 | 0.01175 | 0.8878 | 123 of 127 | 0.00780 | 0.5893 | 46 of 127 |
| rest | gate | trace-weighted | 0.01330 | 0.01181 | 0.8881 | 123 of 127 | 0.00786 | 0.5910 | 46 of 127 |
| rest | up | unweighted | 0.01832 | 0.01661 | 0.9062 | 125 of 127 | 0.01518 | 0.8285 | 46 of 127 |
| rest | up | count-weighted | 0.01723 | 0.01526 | 0.8855 | 125 of 127 | 0.01002 | 0.5816 | 46 of 127 |
| rest | up | trace-weighted | 0.01726 | 0.01529 | 0.8857 | 125 of 127 | 0.01006 | 0.5826 | 46 of 127 |
| rest | both | unweighted | 0.01607 | 0.01460 | 0.9085 | 125 of 127 | 0.01342 | 0.8350 | 46 of 127 |
| rest | both | count-weighted | 0.01470 | 0.01304 | 0.8868 | 125 of 127 | 0.00861 | 0.5856 | 46 of 127 |
| rest | both | trace-weighted | 0.01475 | 0.01308 | 0.8870 | 125 of 127 | 0.00866 | 0.5870 | 46 of 127 |
| diag | gate | unweighted | 0.03151 | 0.03395 | 1.0773 | 0 of 127 | 0.03696 | 1.1729 | 0 of 127 |
| diag | gate | count-weighted | 0.03120 | 0.03370 | 1.0799 | 0 of 127 | 0.04349 | 1.3937 | 0 of 127 |
| diag | gate | trace-weighted | 0.03122 | 0.03371 | 1.0799 | 0 of 127 | 0.04348 | 1.3928 | 0 of 127 |
| diag | up | unweighted | 0.03148 | 0.03390 | 1.0771 | 0 of 127 | 0.03704 | 1.1767 | 0 of 127 |
| diag | up | count-weighted | 0.03117 | 0.03365 | 1.0796 | 0 of 127 | 0.04372 | 1.4027 | 0 of 127 |
| diag | up | trace-weighted | 0.03119 | 0.03367 | 1.0796 | 0 of 127 | 0.04372 | 1.4018 | 0 of 127 |
| diag | both | unweighted | 0.03150 | 0.03393 | 1.0772 | 0 of 127 | 0.03700 | 1.1748 | 0 of 127 |
| diag | both | count-weighted | 0.03119 | 0.03367 | 1.0797 | 0 of 127 | 0.04361 | 1.3982 | 0 of 127 |
| diag | both | trace-weighted | 0.03120 | 0.03369 | 1.0797 | 0 of 127 | 0.04360 | 1.3973 | 0 of 127 |
| plain | gate | unweighted | 0.18493 | 0.19082 | 1.0319 | 0 of 128 | 0.20086 | 1.0861 | 0 of 128 |
| plain | gate | count-weighted | 0.18496 | 0.19083 | 1.0317 | 0 of 128 | 0.22115 | 1.1957 | 0 of 128 |
| plain | gate | trace-weighted | 0.18496 | 0.19083 | 1.0318 | 0 of 128 | 0.22106 | 1.1952 | 0 of 128 |
| plain | up | unweighted | 0.18489 | 0.19077 | 1.0318 | 0 of 128 | 0.20105 | 1.0874 | 0 of 128 |
| plain | up | count-weighted | 0.18489 | 0.19075 | 1.0317 | 0 of 128 | 0.22166 | 1.1989 | 0 of 128 |
| plain | up | trace-weighted | 0.18489 | 0.19075 | 1.0317 | 0 of 128 | 0.22156 | 1.1984 | 0 of 128 |
| plain | both | unweighted | 0.18491 | 0.19080 | 1.0318 | 0 of 128 | 0.20095 | 1.0868 | 0 of 128 |
| plain | both | count-weighted | 0.18493 | 0.19079 | 1.0317 | 0 of 128 | 0.22141 | 1.1973 | 0 of 128 |
| plain | both | trace-weighted | 0.18492 | 0.19079 | 1.0317 | 0 of 128 | 0.22131 | 1.1968 | 0 of 128 |
| gain | gate | unweighted | 0.99443 | 0.99473 | 1.0003 | 14 of 128 | 0.99195 | 0.9975 | 46 of 128 |
| gain | gate | count-weighted | 0.99442 | 0.99473 | 1.0003 | 14 of 128 | 0.98875 | 0.9943 | 46 of 128 |
| gain | gate | trace-weighted | 0.99442 | 0.99473 | 1.0003 | 14 of 128 | 0.98876 | 0.9943 | 46 of 128 |
| gain | up | unweighted | 0.99442 | 0.99474 | 1.0003 | 12 of 128 | 0.99194 | 0.9975 | 46 of 128 |
| gain | up | count-weighted | 0.99439 | 0.99473 | 1.0003 | 12 of 128 | 0.98876 | 0.9943 | 46 of 128 |
| gain | up | trace-weighted | 0.99439 | 0.99473 | 1.0003 | 12 of 128 | 0.98877 | 0.9943 | 46 of 128 |
| gain | both | unweighted | 0.99443 | 0.99473 | 1.0003 | 11 of 128 | 0.99195 | 0.9975 | 46 of 128 |
| gain | both | count-weighted | 0.99441 | 0.99473 | 1.0003 | 11 of 128 | 0.98875 | 0.9943 | 46 of 128 |
| gain | both | trace-weighted | 0.99441 | 0.99473 | 1.0003 | 11 of 128 | 0.98877 | 0.9943 | 46 of 128 |
| energy | gate | unweighted | 1.02307 | 1.02588 | 1.0027 | 0 of 128 | 1.02471 | 1.0016 | 0 of 128 |
| energy | gate | count-weighted | 1.02306 | 1.02587 | 1.0028 | 0 of 128 | 1.02659 | 1.0035 | 0 of 128 |
| energy | gate | trace-weighted | 1.02306 | 1.02587 | 1.0027 | 0 of 128 | 1.02658 | 1.0034 | 0 of 128 |
| energy | up | unweighted | 1.02302 | 1.02586 | 1.0028 | 0 of 128 | 1.02479 | 1.0017 | 0 of 128 |
| energy | up | count-weighted | 1.02296 | 1.02584 | 1.0028 | 0 of 128 | 1.02685 | 1.0038 | 0 of 128 |
| energy | up | trace-weighted | 1.02297 | 1.02584 | 1.0028 | 0 of 128 | 1.02684 | 1.0038 | 0 of 128 |
| energy | both | unweighted | 1.02304 | 1.02587 | 1.0028 | 0 of 128 | 1.02475 | 1.0017 | 0 of 128 |
| energy | both | count-weighted | 1.02301 | 1.02585 | 1.0028 | 0 of 128 | 1.02672 | 1.0036 | 0 of 128 |
| energy | both | trace-weighted | 1.02301 | 1.02585 | 1.0028 | 0 of 128 | 1.02671 | 1.0036 | 0 of 128 |
| ogain | gate | unweighted | 0.99879 | 0.99783 | 0.9990 | 96 of 127 | 1.00360 | 1.0048 | 0 of 127 |
| ogain | gate | count-weighted | 0.99924 | 0.99803 | 0.9988 | 96 of 127 | 1.01153 | 1.0123 | 0 of 127 |
| ogain | gate | trace-weighted | 0.99921 | 0.99801 | 0.9988 | 96 of 127 | 1.01148 | 1.0123 | 0 of 127 |
| ogain | up | unweighted | 0.99784 | 0.99742 | 0.9996 | 79 of 127 | 1.00246 | 1.0046 | 0 of 127 |
| ogain | up | count-weighted | 0.99791 | 0.99738 | 0.9995 | 79 of 127 | 1.00985 | 1.0120 | 0 of 127 |
| ogain | up | trace-weighted | 0.99789 | 0.99737 | 0.9995 | 79 of 127 | 1.00982 | 1.0120 | 0 of 127 |
| ogain | both | unweighted | 0.99845 | 0.99770 | 0.9992 | 94 of 127 | 1.00319 | 1.0047 | 0 of 127 |
| ogain | both | count-weighted | 0.99870 | 0.99778 | 0.9991 | 94 of 127 | 1.01085 | 1.0122 | 0 of 127 |
| ogain | both | trace-weighted | 0.99867 | 0.99776 | 0.9991 | 94 of 127 | 1.01081 | 1.0122 | 0 of 127 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -4.3959e-03 | [-4.6916e-03, -4.0711e-03] |
| up | -5.9722e-03 | [-6.3483e-03, -5.5475e-03] |
| both | -4.9734e-03 | [-5.2559e-03, -4.6484e-03] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -8.6434e-03 | [-9.3518e-03, -7.8259e-03] | -4.2476e-03 | [-4.7761e-03, -3.6377e-03] |
| full | up | -1.1949e-02 | [-1.2831e-02, -1.0910e-02] | -5.9767e-03 | [-6.6418e-03, -5.1735e-03] |
| full | both | -9.8768e-03 | [-1.0551e-02, -9.0474e-03] | -4.9034e-03 | [-5.4295e-03, -4.2606e-03] |
| reg | gate | -7.5223e-03 | [-8.1213e-03, -6.8217e-03] | -3.5021e-03 | [-3.9471e-03, -2.9815e-03] |
| reg | up | -1.0254e-02 | [-1.0978e-02, -9.3751e-03] | -4.8793e-03 | [-5.4242e-03, -4.2067e-03] |
| reg | both | -8.5615e-03 | [-9.1280e-03, -7.8517e-03] | -4.0353e-03 | [-4.4769e-03, -3.4890e-03] |
| bias | gate | -3.2061e-03 | [-3.5048e-03, -2.8518e-03] | -2.9567e-04 | [-4.5768e-04, -6.8986e-05] |
| bias | up | -4.7375e-03 | [-5.1496e-03, -4.2198e-03] | -7.3789e-04 | [-9.7362e-04, -3.9394e-04] |
| bias | both | -3.7841e-03 | [-4.0833e-03, -3.3928e-03] | -4.7537e-04 | [-6.5304e-04, -2.2032e-04] |
| rest | gate | -5.4374e-03 | [-5.8978e-03, -4.9255e-03] | -3.9519e-03 | [-4.3802e-03, -3.5033e-03] |
| rest | up | -7.2114e-03 | [-7.7429e-03, -6.6168e-03] | -5.2388e-03 | [-5.7419e-03, -4.7025e-03] |
| rest | both | -6.0927e-03 | [-6.5244e-03, -5.5935e-03] | -4.4280e-03 | [-4.8428e-03, -3.9805e-03] |
| diag | gate | +1.2285e-02 | [+1.1559e-02, +1.2891e-02] | +9.7921e-03 | [+9.0705e-03, +1.0390e-02] |
| diag | up | +1.2553e-02 | [+1.1817e-02, +1.3165e-02] | +1.0072e-02 | [+9.3355e-03, +1.0681e-02] |
| diag | both | +1.2419e-02 | [+1.1689e-02, +1.3026e-02] | +9.9320e-03 | [+9.2054e-03, +1.0536e-02] |

## Reading (amendment 7 rule, applied literally)

Rule: supported if UD is lower than ours on the full error and higher or equal on the diagonal one in both layers; if UD is also worse on the full error, layer-local output error does not explain UD's edge.

| mean | half | layer 23 full UD < ours | layer 23 diag UD >= ours | branch |
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

In layer 23 (both halves, count-weighted), UD's full error is 0.01467 against our 0.01965 (ratio 0.747; UD - ours -4.973e-03, 95% CI [-5.256e-03, -4.648e-03]) and its diagonal error 0.03367 against our 0.03119 (ratio 1.080); UD is lower in 127 of 127 experts on full and 0 of 127 on diag. Split of the full error in layer 23 (added after registration, not part of the rule): the input-mean part is 0.00495 ours against 0.00164 UD (ratio 0.331), the centred rest 0.01470 against 0.01304 (ratio 0.887). Under the registered rule this is "supported" for both halves together, count-weighted; across the three means and three halves the branches are: supported.

## refined against ours and UD

In layer 23 (both halves, count-weighted), refined's full error is 0.00977 against our 0.01965 and UD's 0.01467 (refined - ours -9.877e-03, 95% CI [-1.055e-02, -9.047e-03]; (ours - refined) / (ours - UD) = 1.986; lower than ours in 46 of 127 experts). Bias part 0.00116 (ours 0.00495, UD 0.00164), rest 0.00861 (ours 0.01470, UD 0.01304). Regularised (the search objective) 0.01173 (ours 0.02030, UD 0.01577). Diagonal 0.04361 (ours 0.03119, UD 0.03367), plain 0.22141 (ours 0.18493, UD 0.19079). Gains (ours / UD / refined): plain 0.9944 / 0.9947 / 0.9888, energy 1.0230 / 1.0259 / 1.0267, output 0.9987 / 0.9978 / 1.0109.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined (count-w.) | refined / ours | refined / UD |
|---|---|---|---|---|---|---|---|---|
| 23 | < 2304 | 82 | 0.109 | 0.02217 | 0.01754 | 0.02217 | 1.000 | 1.264 |
| 23 | >= 2304 | 46 | 0.891 | 0.01934 | 0.01433 | 0.00826 | 0.427 | 0.577 |

Refinement log `models/refined/gu-refined-23.gguf.json` (scripts/refine-codes.py; wall time 626 s for 128 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 23 | 46 | 82 | 4: 46 | 27575639 | 953291 of 953856 | 2327 | 3.96e-10 | 3.95e-10 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 23: max abs(log - here) on reg 3.71e-10 before, 3.60e-10 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304'; 'empty Gram').
