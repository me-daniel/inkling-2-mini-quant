# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a) and refined

Written by `scripts/gram-error.py` (Addendum H amendment 7); extra encodings (amendment 9): refined = `models/refined/gu-refined-mix-1-2-3-5.gguf`; reg = full with G + 0.1 * Diag(G) in numerator and denominator (amendment 9b's objective); gain = sum <q,w> / sum <w,w>, energy = sum <q,q> / sum <w,w>, ogain = sum q H w^T / sum w H w^T with H = G + 0.1 * Diag(G) (amendment 9c; 1 means no attenuation; their `< ours` columns count lower values, not better ones). Per-expert rows: `results/pilot/v2m-offline-1-2-3-5.tsv`.
Grams: `models/grams/gu-1-2-3-5-mix` (capture settings in its log, not restated here), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per row), rest = full - bias (the centred part). Per expert and half, then averaged over 128 experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). `UD < ours` counts experts where UD's per-expert error is lower.

## Layer 1 (IQ2_S; UD `.scale` present, min 3.958e-05, max 0.0001538)

- Gram layout: expert 19: max |strict lower| = 0, max |strict upper| = 7.902e+04
- Gram layout: expert 81: max |strict lower| = 0, max |strict upper| = 7.655e+04
- Gram layout: expert 79: max |strict lower| = 0, max |strict upper| = 1.491e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 4.82e-09; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.1.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.732576 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 604, median 7601, max 38249

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03550 | 0.02276 | 0.6412 | 128 of 128 | 0.01546 | 0.4354 | 125 of 128 |
| full | gate | count-weighted | 0.03485 | 0.02206 | 0.6331 | 128 of 128 | 0.01548 | 0.4442 | 125 of 128 |
| full | gate | trace-weighted | 0.03483 | 0.02212 | 0.6349 | 128 of 128 | 0.01555 | 0.4465 | 125 of 128 |
| full | up | unweighted | 0.04460 | 0.02866 | 0.6426 | 128 of 128 | 0.01886 | 0.4228 | 125 of 128 |
| full | up | count-weighted | 0.04444 | 0.02821 | 0.6349 | 128 of 128 | 0.01914 | 0.4307 | 125 of 128 |
| full | up | trace-weighted | 0.04446 | 0.02831 | 0.6367 | 128 of 128 | 0.01925 | 0.4329 | 125 of 128 |
| full | both | unweighted | 0.03946 | 0.02533 | 0.6419 | 128 of 128 | 0.01693 | 0.4292 | 125 of 128 |
| full | both | count-weighted | 0.03898 | 0.02472 | 0.6340 | 128 of 128 | 0.01706 | 0.4376 | 125 of 128 |
| full | both | trace-weighted | 0.03898 | 0.02479 | 0.6358 | 128 of 128 | 0.01715 | 0.4399 | 125 of 128 |
| reg | gate | unweighted | 0.03691 | 0.02513 | 0.6808 | 128 of 128 | 0.01964 | 0.5322 | 125 of 128 |
| reg | gate | count-weighted | 0.03627 | 0.02443 | 0.6734 | 128 of 128 | 0.01950 | 0.5377 | 125 of 128 |
| reg | gate | trace-weighted | 0.03626 | 0.02448 | 0.6750 | 128 of 128 | 0.01956 | 0.5394 | 125 of 128 |
| reg | up | unweighted | 0.04577 | 0.03123 | 0.6822 | 128 of 128 | 0.02386 | 0.5213 | 125 of 128 |
| reg | up | count-weighted | 0.04562 | 0.03080 | 0.6752 | 128 of 128 | 0.02400 | 0.5260 | 125 of 128 |
| reg | up | trace-weighted | 0.04564 | 0.03089 | 0.6768 | 128 of 128 | 0.02409 | 0.5277 | 125 of 128 |
| reg | both | unweighted | 0.04080 | 0.02780 | 0.6815 | 128 of 128 | 0.02149 | 0.5269 | 125 of 128 |
| reg | both | count-weighted | 0.04034 | 0.02720 | 0.6743 | 128 of 128 | 0.02146 | 0.5320 | 125 of 128 |
| reg | both | trace-weighted | 0.04034 | 0.02727 | 0.6759 | 128 of 128 | 0.02153 | 0.5337 | 125 of 128 |
| bias | gate | unweighted | 0.00516 | 9.006e-04 | 0.1746 | 128 of 128 | 5.093e-04 | 0.0987 | 125 of 128 |
| bias | gate | count-weighted | 0.00516 | 8.774e-04 | 0.1701 | 128 of 128 | 4.337e-04 | 0.0841 | 125 of 128 |
| bias | gate | trace-weighted | 0.00513 | 8.694e-04 | 0.1695 | 128 of 128 | 4.318e-04 | 0.0842 | 125 of 128 |
| bias | up | unweighted | 0.00640 | 0.00115 | 0.1791 | 128 of 128 | 3.123e-04 | 0.0488 | 125 of 128 |
| bias | up | count-weighted | 0.00651 | 0.00114 | 0.1753 | 128 of 128 | 2.052e-04 | 0.0315 | 125 of 128 |
| bias | up | trace-weighted | 0.00648 | 0.00113 | 0.1749 | 128 of 128 | 2.042e-04 | 0.0315 | 125 of 128 |
| bias | both | unweighted | 0.00570 | 0.00101 | 0.1768 | 128 of 128 | 4.241e-04 | 0.0745 | 125 of 128 |
| bias | both | count-weighted | 0.00574 | 9.906e-04 | 0.1727 | 128 of 128 | 3.353e-04 | 0.0585 | 125 of 128 |
| bias | both | trace-weighted | 0.00571 | 9.825e-04 | 0.1722 | 128 of 128 | 3.339e-04 | 0.0585 | 125 of 128 |
| rest | gate | unweighted | 0.03034 | 0.02186 | 0.7205 | 128 of 128 | 0.01495 | 0.4926 | 125 of 128 |
| rest | gate | count-weighted | 0.02969 | 0.02118 | 0.7136 | 128 of 128 | 0.01505 | 0.5068 | 125 of 128 |
| rest | gate | trace-weighted | 0.02970 | 0.02125 | 0.7153 | 128 of 128 | 0.01512 | 0.5091 | 125 of 128 |
| rest | up | unweighted | 0.03820 | 0.02752 | 0.7203 | 128 of 128 | 0.01854 | 0.4854 | 125 of 128 |
| rest | up | count-weighted | 0.03793 | 0.02707 | 0.7137 | 128 of 128 | 0.01893 | 0.4992 | 125 of 128 |
| rest | up | trace-weighted | 0.03798 | 0.02717 | 0.7154 | 128 of 128 | 0.01904 | 0.5013 | 125 of 128 |
| rest | both | unweighted | 0.03376 | 0.02432 | 0.7204 | 128 of 128 | 0.01651 | 0.4890 | 125 of 128 |
| rest | both | count-weighted | 0.03325 | 0.02373 | 0.7136 | 128 of 128 | 0.01672 | 0.5030 | 125 of 128 |
| rest | both | trace-weighted | 0.03328 | 0.02380 | 0.7154 | 128 of 128 | 0.01681 | 0.5053 | 125 of 128 |
| diag | gate | unweighted | 0.06234 | 0.06723 | 1.0785 | 0 of 128 | 0.09380 | 1.5047 | 0 of 128 |
| diag | gate | count-weighted | 0.06238 | 0.06726 | 1.0783 | 0 of 128 | 0.09206 | 1.4759 | 0 of 128 |
| diag | gate | trace-weighted | 0.06241 | 0.06728 | 1.0781 | 0 of 128 | 0.09191 | 1.4727 | 0 of 128 |
| diag | up | unweighted | 0.06231 | 0.06721 | 1.0786 | 0 of 128 | 0.09404 | 1.5093 | 0 of 128 |
| diag | up | count-weighted | 0.06234 | 0.06723 | 1.0784 | 0 of 128 | 0.09230 | 1.4806 | 0 of 128 |
| diag | up | trace-weighted | 0.06237 | 0.06726 | 1.0783 | 0 of 128 | 0.09216 | 1.4775 | 0 of 128 |
| diag | both | unweighted | 0.06232 | 0.06722 | 1.0786 | 0 of 128 | 0.09392 | 1.5070 | 0 of 128 |
| diag | both | count-weighted | 0.06236 | 0.06724 | 1.0783 | 0 of 128 | 0.09218 | 1.4783 | 0 of 128 |
| diag | both | trace-weighted | 0.06239 | 0.06727 | 1.0782 | 0 of 128 | 0.09203 | 1.4751 | 0 of 128 |
| plain | gate | unweighted | 0.26005 | 0.26790 | 1.0302 | 0 of 128 | 0.32822 | 1.2622 | 0 of 128 |
| plain | gate | count-weighted | 0.26007 | 0.26796 | 1.0303 | 0 of 128 | 0.32539 | 1.2512 | 0 of 128 |
| plain | gate | trace-weighted | 0.26005 | 0.26795 | 1.0304 | 0 of 128 | 0.32502 | 1.2498 | 0 of 128 |
| plain | up | unweighted | 0.25990 | 0.26775 | 1.0302 | 0 of 128 | 0.32880 | 1.2651 | 0 of 128 |
| plain | up | count-weighted | 0.25991 | 0.26780 | 1.0303 | 0 of 128 | 0.32601 | 1.2543 | 0 of 128 |
| plain | up | trace-weighted | 0.25989 | 0.26779 | 1.0304 | 0 of 128 | 0.32564 | 1.2530 | 0 of 128 |
| plain | both | unweighted | 0.25997 | 0.26783 | 1.0302 | 0 of 128 | 0.32851 | 1.2636 | 0 of 128 |
| plain | both | count-weighted | 0.25999 | 0.26788 | 1.0303 | 0 of 128 | 0.32570 | 1.2527 | 0 of 128 |
| plain | both | trace-weighted | 0.25997 | 0.26787 | 1.0304 | 0 of 128 | 0.32533 | 1.2514 | 0 of 128 |
| gain | gate | unweighted | 0.99668 | 0.99782 | 1.0011 | 6 of 128 | 0.98503 | 0.9883 | 125 of 128 |
| gain | gate | count-weighted | 0.99662 | 0.99777 | 1.0012 | 6 of 128 | 0.98550 | 0.9888 | 125 of 128 |
| gain | gate | trace-weighted | 0.99661 | 0.99778 | 1.0012 | 6 of 128 | 0.98554 | 0.9889 | 125 of 128 |
| gain | up | unweighted | 0.99666 | 0.99781 | 1.0012 | 5 of 128 | 0.98527 | 0.9886 | 125 of 128 |
| gain | up | count-weighted | 0.99660 | 0.99776 | 1.0012 | 5 of 128 | 0.98579 | 0.9892 | 125 of 128 |
| gain | up | trace-weighted | 0.99659 | 0.99777 | 1.0012 | 5 of 128 | 0.98583 | 0.9892 | 125 of 128 |
| gain | both | unweighted | 0.99667 | 0.99781 | 1.0011 | 5 of 128 | 0.98515 | 0.9884 | 125 of 128 |
| gain | both | count-weighted | 0.99661 | 0.99777 | 1.0012 | 5 of 128 | 0.98564 | 0.9890 | 125 of 128 |
| gain | both | trace-weighted | 0.99660 | 0.99777 | 1.0012 | 5 of 128 | 0.98568 | 0.9890 | 125 of 128 |
| energy | gate | unweighted | 1.06098 | 1.06741 | 1.0061 | 0 of 128 | 1.07801 | 1.0160 | 0 of 128 |
| energy | gate | count-weighted | 1.06088 | 1.06735 | 1.0061 | 0 of 128 | 1.07699 | 1.0152 | 0 of 128 |
| energy | gate | trace-weighted | 1.06085 | 1.06735 | 1.0061 | 0 of 128 | 1.07682 | 1.0151 | 0 of 128 |
| energy | up | unweighted | 1.06086 | 1.06731 | 1.0061 | 0 of 128 | 1.07886 | 1.0170 | 0 of 128 |
| energy | up | count-weighted | 1.06075 | 1.06724 | 1.0061 | 0 of 128 | 1.07796 | 1.0162 | 0 of 128 |
| energy | up | trace-weighted | 1.06072 | 1.06725 | 1.0062 | 0 of 128 | 1.07781 | 1.0161 | 0 of 128 |
| energy | both | unweighted | 1.06092 | 1.06736 | 1.0061 | 0 of 128 | 1.07843 | 1.0165 | 0 of 128 |
| energy | both | count-weighted | 1.06081 | 1.06730 | 1.0061 | 0 of 128 | 1.07748 | 1.0157 | 0 of 128 |
| energy | both | trace-weighted | 1.06078 | 1.06730 | 1.0061 | 0 of 128 | 1.07732 | 1.0156 | 0 of 128 |
| ogain | gate | unweighted | 0.99133 | 0.99917 | 1.0079 | 0 of 128 | 1.00769 | 1.0165 | 0 of 128 |
| ogain | gate | count-weighted | 0.99115 | 0.99922 | 1.0081 | 0 of 128 | 1.00808 | 1.0171 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99107 | 0.99920 | 1.0082 | 0 of 128 | 1.00798 | 1.0171 | 0 of 128 |
| ogain | up | unweighted | 0.99597 | 0.99881 | 1.0029 | 3 of 128 | 1.00598 | 1.0100 | 0 of 128 |
| ogain | up | count-weighted | 0.99594 | 0.99885 | 1.0029 | 3 of 128 | 1.00611 | 1.0102 | 0 of 128 |
| ogain | up | trace-weighted | 0.99591 | 0.99884 | 1.0029 | 3 of 128 | 1.00602 | 1.0101 | 0 of 128 |
| ogain | both | unweighted | 0.99338 | 0.99901 | 1.0057 | 0 of 128 | 1.00695 | 1.0137 | 0 of 128 |
| ogain | both | count-weighted | 0.99325 | 0.99906 | 1.0058 | 0 of 128 | 1.00723 | 1.0141 | 0 of 128 |
| ogain | both | trace-weighted | 0.99320 | 0.99904 | 1.0059 | 0 of 128 | 1.00713 | 1.0140 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.2783e-02 | [-1.3142e-02, -1.2409e-02] |
| up | -1.6225e-02 | [-1.6562e-02, -1.5866e-02] |
| both | -1.4267e-02 | [-1.4604e-02, -1.3915e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.9366e-02 | [-1.9871e-02, -1.8870e-02] | -6.5829e-03 | [-6.9987e-03, -6.2044e-03] |
| full | up | -2.5299e-02 | [-2.5777e-02, -2.4840e-02] | -9.0747e-03 | [-9.5611e-03, -8.6280e-03] |
| full | both | -2.1924e-02 | [-2.2389e-02, -2.1478e-02] | -7.6577e-03 | [-8.1015e-03, -7.2555e-03] |
| reg | gate | -1.6769e-02 | [-1.7182e-02, -1.6364e-02] | -4.9246e-03 | [-5.2718e-03, -4.6079e-03] |
| reg | up | -2.1621e-02 | [-2.1998e-02, -2.1258e-02] | -6.8047e-03 | [-7.2066e-03, -6.4351e-03] |
| reg | both | -1.8879e-02 | [-1.9251e-02, -1.8519e-02] | -5.7426e-03 | [-6.1097e-03, -5.4090e-03] |
| bias | gate | -4.7231e-03 | [-4.8491e-03, -4.5916e-03] | -4.4375e-04 | [-4.8853e-04, -3.9872e-04] |
| bias | up | -6.3013e-03 | [-6.4659e-03, -6.1373e-03] | -9.3559e-04 | [-1.0040e-03, -8.7150e-04] |
| bias | both | -5.4013e-03 | [-5.5274e-03, -5.2730e-03] | -6.5526e-04 | [-7.0614e-04, -6.0687e-04] |
| rest | gate | -1.4643e-02 | [-1.5093e-02, -1.4225e-02] | -6.1392e-03 | [-6.5358e-03, -5.7832e-03] |
| rest | up | -1.8998e-02 | [-1.9429e-02, -1.8594e-02] | -8.1391e-03 | [-8.6025e-03, -7.7119e-03] |
| rest | both | -1.6523e-02 | [-1.6949e-02, -1.6120e-02] | -7.0025e-03 | [-7.4248e-03, -6.6196e-03] |
| diag | gate | +2.9686e-02 | [+2.8808e-02, +3.0645e-02] | +2.4803e-02 | [+2.3935e-02, +2.5751e-02] |
| diag | up | +2.9961e-02 | [+2.9074e-02, +3.0925e-02] | +2.5074e-02 | [+2.4195e-02, +2.6030e-02] |
| diag | both | +2.9823e-02 | [+2.8943e-02, +3.0786e-02] | +2.4938e-02 | [+2.4066e-02, +2.5891e-02] |

## Layer 2 (IQ2_S; UD `.scale` present, min 3.92e-05, max 0.0001678)

- Gram layout: expert 119: max |strict lower| = 0, max |strict upper| = 1.198e+05
- Gram layout: expert 40: max |strict lower| = 0, max |strict upper| = 4.896e+04
- Gram layout: expert 49: max |strict lower| = 0, max |strict upper| = 7.826e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 2.67e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.2.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.779855 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 1954, median 8847, max 31920

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03518 | 0.02380 | 0.6764 | 128 of 128 | 0.01624 | 0.4617 | 126 of 128 |
| full | gate | count-weighted | 0.03461 | 0.02320 | 0.6704 | 128 of 128 | 0.01618 | 0.4676 | 126 of 128 |
| full | gate | trace-weighted | 0.03457 | 0.02320 | 0.6711 | 128 of 128 | 0.01616 | 0.4675 | 126 of 128 |
| full | up | unweighted | 0.04685 | 0.03167 | 0.6761 | 128 of 128 | 0.02052 | 0.4380 | 126 of 128 |
| full | up | count-weighted | 0.04635 | 0.03109 | 0.6706 | 128 of 128 | 0.02054 | 0.4431 | 126 of 128 |
| full | up | trace-weighted | 0.04633 | 0.03110 | 0.6713 | 128 of 128 | 0.02052 | 0.4429 | 126 of 128 |
| full | both | unweighted | 0.04014 | 0.02714 | 0.6762 | 128 of 128 | 0.01806 | 0.4500 | 126 of 128 |
| full | both | count-weighted | 0.03958 | 0.02654 | 0.6705 | 128 of 128 | 0.01803 | 0.4555 | 126 of 128 |
| full | both | trace-weighted | 0.03954 | 0.02654 | 0.6712 | 128 of 128 | 0.01800 | 0.4553 | 126 of 128 |
| reg | gate | unweighted | 0.03659 | 0.02607 | 0.7125 | 128 of 128 | 0.02026 | 0.5536 | 126 of 128 |
| reg | gate | count-weighted | 0.03603 | 0.02547 | 0.7070 | 128 of 128 | 0.02008 | 0.5574 | 126 of 128 |
| reg | gate | trace-weighted | 0.03599 | 0.02547 | 0.7076 | 128 of 128 | 0.02005 | 0.5572 | 126 of 128 |
| reg | up | unweighted | 0.04788 | 0.03410 | 0.7122 | 128 of 128 | 0.02550 | 0.5326 | 126 of 128 |
| reg | up | count-weighted | 0.04742 | 0.03353 | 0.7072 | 128 of 128 | 0.02540 | 0.5357 | 126 of 128 |
| reg | up | trace-weighted | 0.04740 | 0.03355 | 0.7079 | 128 of 128 | 0.02538 | 0.5355 | 126 of 128 |
| reg | both | unweighted | 0.04144 | 0.02952 | 0.7123 | 128 of 128 | 0.02251 | 0.5432 | 126 of 128 |
| reg | both | count-weighted | 0.04090 | 0.02892 | 0.7071 | 128 of 128 | 0.02236 | 0.5466 | 126 of 128 |
| reg | both | trace-weighted | 0.04086 | 0.02892 | 0.7077 | 128 of 128 | 0.02233 | 0.5464 | 126 of 128 |
| bias | gate | unweighted | 0.00485 | 9.109e-04 | 0.1877 | 128 of 128 | 5.565e-04 | 0.1147 | 126 of 128 |
| bias | gate | count-weighted | 0.00486 | 8.985e-04 | 0.1848 | 128 of 128 | 5.173e-04 | 0.1064 | 126 of 128 |
| bias | gate | trace-weighted | 0.00486 | 9.040e-04 | 0.1860 | 128 of 128 | 5.188e-04 | 0.1067 | 126 of 128 |
| bias | up | unweighted | 0.00645 | 0.00121 | 0.1870 | 128 of 128 | 2.618e-04 | 0.0406 | 126 of 128 |
| bias | up | count-weighted | 0.00647 | 0.00120 | 0.1851 | 128 of 128 | 1.909e-04 | 0.0295 | 126 of 128 |
| bias | up | trace-weighted | 0.00648 | 0.00121 | 0.1864 | 128 of 128 | 1.902e-04 | 0.0294 | 126 of 128 |
| bias | both | unweighted | 0.00553 | 0.00104 | 0.1873 | 128 of 128 | 4.321e-04 | 0.0782 | 126 of 128 |
| bias | both | count-weighted | 0.00554 | 0.00103 | 0.1849 | 128 of 128 | 3.794e-04 | 0.0684 | 126 of 128 |
| bias | both | trace-weighted | 0.00554 | 0.00103 | 0.1862 | 128 of 128 | 3.800e-04 | 0.0685 | 126 of 128 |
| rest | gate | unweighted | 0.03033 | 0.02289 | 0.7546 | 128 of 128 | 0.01569 | 0.5172 | 126 of 128 |
| rest | gate | count-weighted | 0.02975 | 0.02230 | 0.7497 | 128 of 128 | 0.01567 | 0.5267 | 126 of 128 |
| rest | gate | trace-weighted | 0.02971 | 0.02229 | 0.7505 | 128 of 128 | 0.01564 | 0.5265 | 126 of 128 |
| rest | up | unweighted | 0.04040 | 0.03047 | 0.7541 | 128 of 128 | 0.02026 | 0.5014 | 126 of 128 |
| rest | up | count-weighted | 0.03988 | 0.02989 | 0.7494 | 128 of 128 | 0.02035 | 0.5102 | 126 of 128 |
| rest | up | trace-weighted | 0.03986 | 0.02990 | 0.7501 | 128 of 128 | 0.02033 | 0.5101 | 126 of 128 |
| rest | both | unweighted | 0.03461 | 0.02611 | 0.7544 | 128 of 128 | 0.01763 | 0.5094 | 126 of 128 |
| rest | both | count-weighted | 0.03404 | 0.02551 | 0.7496 | 128 of 128 | 0.01765 | 0.5185 | 126 of 128 |
| rest | both | trace-weighted | 0.03400 | 0.02551 | 0.7503 | 128 of 128 | 0.01762 | 0.5184 | 126 of 128 |
| diag | gate | unweighted | 0.06176 | 0.06633 | 1.0739 | 0 of 128 | 0.09081 | 1.4703 | 0 of 128 |
| diag | gate | count-weighted | 0.06180 | 0.06635 | 1.0736 | 0 of 128 | 0.08986 | 1.4539 | 0 of 128 |
| diag | gate | trace-weighted | 0.06180 | 0.06635 | 1.0736 | 0 of 128 | 0.08985 | 1.4539 | 0 of 128 |
| diag | up | unweighted | 0.06174 | 0.06630 | 1.0739 | 0 of 128 | 0.09120 | 1.4772 | 0 of 128 |
| diag | up | count-weighted | 0.06178 | 0.06633 | 1.0736 | 0 of 128 | 0.09023 | 1.4605 | 0 of 128 |
| diag | up | trace-weighted | 0.06178 | 0.06633 | 1.0736 | 0 of 128 | 0.09023 | 1.4605 | 0 of 128 |
| diag | both | unweighted | 0.06175 | 0.06631 | 1.0739 | 0 of 128 | 0.09101 | 1.4738 | 0 of 128 |
| diag | both | count-weighted | 0.06179 | 0.06634 | 1.0736 | 0 of 128 | 0.09004 | 1.4572 | 0 of 128 |
| diag | both | trace-weighted | 0.06179 | 0.06634 | 1.0736 | 0 of 128 | 0.09004 | 1.4572 | 0 of 128 |
| plain | gate | unweighted | 0.26189 | 0.26942 | 1.0288 | 0 of 128 | 0.33183 | 1.2670 | 0 of 128 |
| plain | gate | count-weighted | 0.26191 | 0.26946 | 1.0288 | 0 of 128 | 0.33036 | 1.2613 | 0 of 128 |
| plain | gate | trace-weighted | 0.26193 | 0.26948 | 1.0288 | 0 of 128 | 0.33042 | 1.2615 | 0 of 128 |
| plain | up | unweighted | 0.26177 | 0.26930 | 1.0288 | 0 of 128 | 0.33307 | 1.2724 | 0 of 128 |
| plain | up | count-weighted | 0.26178 | 0.26932 | 1.0288 | 0 of 128 | 0.33158 | 1.2667 | 0 of 128 |
| plain | up | trace-weighted | 0.26179 | 0.26934 | 1.0288 | 0 of 128 | 0.33164 | 1.2668 | 0 of 128 |
| plain | both | unweighted | 0.26183 | 0.26936 | 1.0288 | 0 of 128 | 0.33245 | 1.2697 | 0 of 128 |
| plain | both | count-weighted | 0.26185 | 0.26939 | 1.0288 | 0 of 128 | 0.33097 | 1.2640 | 0 of 128 |
| plain | both | trace-weighted | 0.26186 | 0.26941 | 1.0288 | 0 of 128 | 0.33103 | 1.2642 | 0 of 128 |
| gain | gate | unweighted | 0.99768 | 0.99868 | 1.0010 | 4 of 128 | 0.98657 | 0.9889 | 126 of 128 |
| gain | gate | count-weighted | 0.99761 | 0.99859 | 1.0010 | 4 of 128 | 0.98684 | 0.9892 | 126 of 128 |
| gain | gate | trace-weighted | 0.99761 | 0.99858 | 1.0010 | 4 of 128 | 0.98684 | 0.9892 | 126 of 128 |
| gain | up | unweighted | 0.99768 | 0.99869 | 1.0010 | 3 of 128 | 0.98651 | 0.9888 | 126 of 128 |
| gain | up | count-weighted | 0.99760 | 0.99860 | 1.0010 | 3 of 128 | 0.98677 | 0.9891 | 126 of 128 |
| gain | up | trace-weighted | 0.99761 | 0.99860 | 1.0010 | 3 of 128 | 0.98677 | 0.9891 | 126 of 128 |
| gain | both | unweighted | 0.99768 | 0.99869 | 1.0010 | 4 of 128 | 0.98654 | 0.9888 | 126 of 128 |
| gain | both | count-weighted | 0.99760 | 0.99859 | 1.0010 | 4 of 128 | 0.98680 | 0.9892 | 126 of 128 |
| gain | both | trace-weighted | 0.99761 | 0.99859 | 1.0010 | 4 of 128 | 0.98680 | 0.9892 | 126 of 128 |
| energy | gate | unweighted | 1.06394 | 1.06995 | 1.0057 | 0 of 128 | 1.08337 | 1.0183 | 0 of 128 |
| energy | gate | count-weighted | 1.06381 | 1.06979 | 1.0056 | 0 of 128 | 1.08288 | 1.0179 | 0 of 128 |
| energy | gate | trace-weighted | 1.06382 | 1.06979 | 1.0056 | 0 of 128 | 1.08292 | 1.0180 | 0 of 128 |
| energy | up | unweighted | 1.06388 | 1.06991 | 1.0057 | 0 of 128 | 1.08408 | 1.0190 | 0 of 128 |
| energy | up | count-weighted | 1.06374 | 1.06973 | 1.0056 | 0 of 128 | 1.08354 | 1.0186 | 0 of 128 |
| energy | up | trace-weighted | 1.06375 | 1.06974 | 1.0056 | 0 of 128 | 1.08358 | 1.0186 | 0 of 128 |
| energy | both | unweighted | 1.06391 | 1.06993 | 1.0057 | 0 of 128 | 1.08373 | 1.0186 | 0 of 128 |
| energy | both | count-weighted | 1.06377 | 1.06976 | 1.0056 | 0 of 128 | 1.08321 | 1.0183 | 0 of 128 |
| energy | both | trace-weighted | 1.06378 | 1.06976 | 1.0056 | 0 of 128 | 1.08325 | 1.0183 | 0 of 128 |
| ogain | gate | unweighted | 0.99390 | 0.99953 | 1.0057 | 0 of 128 | 1.00943 | 1.0156 | 0 of 128 |
| ogain | gate | count-weighted | 0.99403 | 0.99949 | 1.0055 | 0 of 128 | 1.00978 | 1.0158 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99404 | 0.99948 | 1.0055 | 0 of 128 | 1.00983 | 1.0159 | 0 of 128 |
| ogain | up | unweighted | 0.99676 | 0.99918 | 1.0024 | 1 of 128 | 1.00596 | 1.0092 | 0 of 128 |
| ogain | up | count-weighted | 0.99669 | 0.99909 | 1.0024 | 1 of 128 | 1.00608 | 1.0094 | 0 of 128 |
| ogain | up | trace-weighted | 0.99669 | 0.99909 | 1.0024 | 1 of 128 | 1.00613 | 1.0095 | 0 of 128 |
| ogain | both | unweighted | 0.99514 | 0.99938 | 1.0043 | 0 of 128 | 1.00795 | 1.0129 | 0 of 128 |
| ogain | both | count-weighted | 0.99518 | 0.99932 | 1.0042 | 0 of 128 | 1.00820 | 1.0131 | 0 of 128 |
| ogain | both | trace-weighted | 0.99518 | 0.99932 | 1.0042 | 0 of 128 | 1.00825 | 1.0131 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.1408e-02 | [-1.1714e-02, -1.1130e-02] |
| up | -1.5269e-02 | [-1.5527e-02, -1.5028e-02] |
| both | -1.3042e-02 | [-1.3325e-02, -1.2785e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.8425e-02 | [-1.8900e-02, -1.7976e-02] | -7.0177e-03 | [-7.3858e-03, -6.6768e-03] |
| full | up | -2.5814e-02 | [-2.6269e-02, -2.5388e-02] | -1.0545e-02 | [-1.0997e-02, -1.0118e-02] |
| full | both | -2.1552e-02 | [-2.2004e-02, -2.1127e-02] | -8.5104e-03 | [-8.9097e-03, -8.1403e-03] |
| reg | gate | -1.5948e-02 | [-1.6336e-02, -1.5580e-02] | -5.3914e-03 | [-5.6999e-03, -5.1067e-03] |
| reg | up | -2.2016e-02 | [-2.2373e-02, -2.1680e-02] | -8.1331e-03 | [-8.5048e-03, -7.7814e-03] |
| reg | both | -1.8543e-02 | [-1.8903e-02, -1.8202e-02] | -6.5639e-03 | [-6.8921e-03, -6.2560e-03] |
| bias | gate | -4.3437e-03 | [-4.4656e-03, -4.2211e-03] | -3.8124e-04 | [-4.2283e-04, -3.3995e-04] |
| bias | up | -6.2797e-03 | [-6.4141e-03, -6.1470e-03] | -1.0068e-03 | [-1.0660e-03, -9.4889e-04] |
| bias | both | -5.1629e-03 | [-5.2773e-03, -5.0472e-03] | -6.4567e-04 | [-6.8983e-04, -6.0154e-04] |
| rest | gate | -1.4082e-02 | [-1.4493e-02, -1.3695e-02] | -6.6364e-03 | [-6.9820e-03, -6.3153e-03] |
| rest | up | -1.9534e-02 | [-1.9983e-02, -1.9115e-02] | -9.5382e-03 | [-9.9722e-03, -9.1310e-03] |
| rest | both | -1.6390e-02 | [-1.6809e-02, -1.6004e-02] | -7.8648e-03 | [-8.2448e-03, -7.5122e-03] |
| diag | gate | +2.8051e-02 | [+2.7370e-02, +2.8742e-02] | +2.3501e-02 | [+2.2830e-02, +2.4182e-02] |
| diag | up | +2.8448e-02 | [+2.7760e-02, +2.9155e-02] | +2.3899e-02 | [+2.3221e-02, +2.4591e-02] |
| diag | both | +2.8250e-02 | [+2.7563e-02, +2.8948e-02] | +2.3700e-02 | [+2.3025e-02, +2.4386e-02] |

## Layer 3 (IQ2_S; UD `.scale` present, min 4.028e-05, max 0.0002603)

- Gram layout: expert 86: max |strict lower| = 0, max |strict upper| = 5.595e+04
- Gram layout: expert 94: max |strict lower| = 0, max |strict upper| = 5.727e+04
- Gram layout: expert 95: max |strict lower| = 0, max |strict upper| = 5.765e+04
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 2.19e-08; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.3.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.795206 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 603, median 8356, max 21651

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03556 | 0.02450 | 0.6889 | 128 of 128 | 0.01700 | 0.4781 | 126 of 128 |
| full | gate | count-weighted | 0.03466 | 0.02365 | 0.6824 | 128 of 128 | 0.01669 | 0.4814 | 126 of 128 |
| full | gate | trace-weighted | 0.03455 | 0.02358 | 0.6824 | 128 of 128 | 0.01668 | 0.4828 | 126 of 128 |
| full | up | unweighted | 0.04876 | 0.03356 | 0.6883 | 128 of 128 | 0.02187 | 0.4485 | 126 of 128 |
| full | up | count-weighted | 0.04780 | 0.03260 | 0.6820 | 128 of 128 | 0.02151 | 0.4500 | 126 of 128 |
| full | up | trace-weighted | 0.04769 | 0.03253 | 0.6820 | 128 of 128 | 0.02150 | 0.4508 | 126 of 128 |
| full | both | unweighted | 0.04108 | 0.02828 | 0.6886 | 128 of 128 | 0.01904 | 0.4635 | 126 of 128 |
| full | both | count-weighted | 0.04013 | 0.02738 | 0.6822 | 128 of 128 | 0.01870 | 0.4659 | 126 of 128 |
| full | both | trace-weighted | 0.04002 | 0.02730 | 0.6822 | 128 of 128 | 0.01869 | 0.4670 | 126 of 128 |
| reg | gate | unweighted | 0.03698 | 0.02675 | 0.7234 | 128 of 128 | 0.02096 | 0.5668 | 126 of 128 |
| reg | gate | count-weighted | 0.03609 | 0.02589 | 0.7175 | 128 of 128 | 0.02052 | 0.5687 | 126 of 128 |
| reg | gate | trace-weighted | 0.03598 | 0.02582 | 0.7175 | 128 of 128 | 0.02050 | 0.5698 | 126 of 128 |
| reg | up | unweighted | 0.04971 | 0.03594 | 0.7229 | 128 of 128 | 0.02687 | 0.5406 | 126 of 128 |
| reg | up | count-weighted | 0.04880 | 0.03500 | 0.7172 | 128 of 128 | 0.02639 | 0.5408 | 126 of 128 |
| reg | up | trace-weighted | 0.04870 | 0.03493 | 0.7171 | 128 of 128 | 0.02637 | 0.5414 | 126 of 128 |
| reg | both | unweighted | 0.04236 | 0.03063 | 0.7232 | 128 of 128 | 0.02346 | 0.5538 | 126 of 128 |
| reg | both | count-weighted | 0.04144 | 0.02973 | 0.7174 | 128 of 128 | 0.02300 | 0.5549 | 126 of 128 |
| reg | both | trace-weighted | 0.04134 | 0.02965 | 0.7173 | 128 of 128 | 0.02297 | 0.5557 | 126 of 128 |
| bias | gate | unweighted | 0.00473 | 8.827e-04 | 0.1866 | 128 of 128 | 6.239e-04 | 0.1319 | 126 of 128 |
| bias | gate | count-weighted | 0.00469 | 8.602e-04 | 0.1834 | 128 of 128 | 5.663e-04 | 0.1208 | 126 of 128 |
| bias | gate | trace-weighted | 0.00468 | 8.601e-04 | 0.1836 | 128 of 128 | 5.711e-04 | 0.1219 | 126 of 128 |
| bias | up | unweighted | 0.00650 | 0.00121 | 0.1858 | 128 of 128 | 2.745e-04 | 0.0422 | 126 of 128 |
| bias | up | count-weighted | 0.00648 | 0.00119 | 0.1833 | 128 of 128 | 1.627e-04 | 0.0251 | 126 of 128 |
| bias | up | trace-weighted | 0.00648 | 0.00119 | 0.1836 | 128 of 128 | 1.612e-04 | 0.0249 | 126 of 128 |
| bias | both | unweighted | 0.00547 | 0.00102 | 0.1861 | 128 of 128 | 4.792e-04 | 0.0876 | 126 of 128 |
| bias | both | count-weighted | 0.00543 | 9.959e-04 | 0.1833 | 128 of 128 | 3.989e-04 | 0.0734 | 126 of 128 |
| bias | both | trace-weighted | 0.00543 | 9.965e-04 | 0.1835 | 128 of 128 | 4.012e-04 | 0.0739 | 126 of 128 |
| rest | gate | unweighted | 0.03083 | 0.02362 | 0.7660 | 128 of 128 | 0.01638 | 0.5312 | 126 of 128 |
| rest | gate | count-weighted | 0.02997 | 0.02279 | 0.7605 | 128 of 128 | 0.01612 | 0.5379 | 126 of 128 |
| rest | gate | trace-weighted | 0.02987 | 0.02272 | 0.7606 | 128 of 128 | 0.01611 | 0.5394 | 126 of 128 |
| rest | up | unweighted | 0.04226 | 0.03235 | 0.7656 | 128 of 128 | 0.02159 | 0.5110 | 126 of 128 |
| rest | up | count-weighted | 0.04132 | 0.03141 | 0.7602 | 128 of 128 | 0.02135 | 0.5166 | 126 of 128 |
| rest | up | trace-weighted | 0.04122 | 0.03134 | 0.7603 | 128 of 128 | 0.02134 | 0.5178 | 126 of 128 |
| rest | both | unweighted | 0.03560 | 0.02727 | 0.7658 | 128 of 128 | 0.01856 | 0.5212 | 126 of 128 |
| rest | both | count-weighted | 0.03470 | 0.02638 | 0.7604 | 128 of 128 | 0.01830 | 0.5273 | 126 of 128 |
| rest | both | trace-weighted | 0.03459 | 0.02631 | 0.7605 | 128 of 128 | 0.01829 | 0.5287 | 126 of 128 |
| diag | gate | unweighted | 0.06215 | 0.06649 | 1.0699 | 0 of 128 | 0.09029 | 1.4528 | 0 of 128 |
| diag | gate | count-weighted | 0.06213 | 0.06647 | 1.0698 | 0 of 128 | 0.08946 | 1.4398 | 0 of 128 |
| diag | gate | trace-weighted | 0.06215 | 0.06648 | 1.0697 | 0 of 128 | 0.08936 | 1.4379 | 0 of 128 |
| diag | up | unweighted | 0.06211 | 0.06646 | 1.0699 | 0 of 128 | 0.09072 | 1.4607 | 0 of 128 |
| diag | up | count-weighted | 0.06210 | 0.06644 | 1.0699 | 0 of 128 | 0.08988 | 1.4474 | 0 of 128 |
| diag | up | trace-weighted | 0.06211 | 0.06645 | 1.0698 | 0 of 128 | 0.08978 | 1.4454 | 0 of 128 |
| diag | both | unweighted | 0.06213 | 0.06647 | 1.0699 | 0 of 128 | 0.09051 | 1.4567 | 0 of 128 |
| diag | both | count-weighted | 0.06212 | 0.06645 | 1.0698 | 0 of 128 | 0.08967 | 1.4436 | 0 of 128 |
| diag | both | trace-weighted | 0.06213 | 0.06647 | 1.0698 | 0 of 128 | 0.08957 | 1.4416 | 0 of 128 |
| plain | gate | unweighted | 0.26148 | 0.26857 | 1.0271 | 0 of 128 | 0.32744 | 1.2523 | 0 of 128 |
| plain | gate | count-weighted | 0.26162 | 0.26870 | 1.0271 | 0 of 128 | 0.32662 | 1.2485 | 0 of 128 |
| plain | gate | trace-weighted | 0.26165 | 0.26873 | 1.0271 | 0 of 128 | 0.32652 | 1.2479 | 0 of 128 |
| plain | up | unweighted | 0.26135 | 0.26845 | 1.0272 | 0 of 128 | 0.32889 | 1.2584 | 0 of 128 |
| plain | up | count-weighted | 0.26148 | 0.26857 | 1.0271 | 0 of 128 | 0.32810 | 1.2548 | 0 of 128 |
| plain | up | trace-weighted | 0.26151 | 0.26860 | 1.0271 | 0 of 128 | 0.32802 | 1.2543 | 0 of 128 |
| plain | both | unweighted | 0.26141 | 0.26851 | 1.0271 | 0 of 128 | 0.32816 | 1.2553 | 0 of 128 |
| plain | both | count-weighted | 0.26155 | 0.26863 | 1.0271 | 0 of 128 | 0.32736 | 1.2516 | 0 of 128 |
| plain | both | trace-weighted | 0.26158 | 0.26867 | 1.0271 | 0 of 128 | 0.32727 | 1.2511 | 0 of 128 |
| gain | gate | unweighted | 0.99750 | 0.99843 | 1.0009 | 3 of 128 | 0.98692 | 0.9894 | 126 of 128 |
| gain | gate | count-weighted | 0.99749 | 0.99840 | 1.0009 | 3 of 128 | 0.98719 | 0.9897 | 126 of 128 |
| gain | gate | trace-weighted | 0.99748 | 0.99839 | 1.0009 | 3 of 128 | 0.98724 | 0.9897 | 126 of 128 |
| gain | up | unweighted | 0.99749 | 0.99846 | 1.0010 | 3 of 128 | 0.98671 | 0.9892 | 126 of 128 |
| gain | up | count-weighted | 0.99747 | 0.99843 | 1.0010 | 3 of 128 | 0.98694 | 0.9894 | 126 of 128 |
| gain | up | trace-weighted | 0.99747 | 0.99842 | 1.0010 | 3 of 128 | 0.98697 | 0.9895 | 126 of 128 |
| gain | both | unweighted | 0.99750 | 0.99844 | 1.0009 | 3 of 128 | 0.98682 | 0.9893 | 126 of 128 |
| gain | both | count-weighted | 0.99748 | 0.99841 | 1.0009 | 3 of 128 | 0.98707 | 0.9896 | 126 of 128 |
| gain | both | trace-weighted | 0.99747 | 0.99840 | 1.0009 | 3 of 128 | 0.98710 | 0.9896 | 126 of 128 |
| energy | gate | unweighted | 1.06338 | 1.06900 | 1.0053 | 0 of 128 | 1.08116 | 1.0167 | 0 of 128 |
| energy | gate | count-weighted | 1.06342 | 1.06899 | 1.0052 | 0 of 128 | 1.08110 | 1.0166 | 0 of 128 |
| energy | gate | trace-weighted | 1.06342 | 1.06899 | 1.0052 | 0 of 128 | 1.08113 | 1.0167 | 0 of 128 |
| energy | up | unweighted | 1.06329 | 1.06898 | 1.0053 | 0 of 128 | 1.08170 | 1.0173 | 0 of 128 |
| energy | up | count-weighted | 1.06332 | 1.06899 | 1.0053 | 0 of 128 | 1.08157 | 1.0172 | 0 of 128 |
| energy | up | trace-weighted | 1.06332 | 1.06899 | 1.0053 | 0 of 128 | 1.08156 | 1.0172 | 0 of 128 |
| energy | both | unweighted | 1.06333 | 1.06899 | 1.0053 | 0 of 128 | 1.08143 | 1.0170 | 0 of 128 |
| energy | both | count-weighted | 1.06337 | 1.06899 | 1.0053 | 0 of 128 | 1.08133 | 1.0169 | 0 of 128 |
| energy | both | trace-weighted | 1.06337 | 1.06899 | 1.0053 | 0 of 128 | 1.08134 | 1.0169 | 0 of 128 |
| ogain | gate | unweighted | 0.99464 | 0.99937 | 1.0048 | 11 of 128 | 1.00984 | 1.0153 | 0 of 128 |
| ogain | gate | count-weighted | 0.99513 | 0.99935 | 1.0042 | 11 of 128 | 1.01041 | 1.0154 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99523 | 0.99933 | 1.0041 | 11 of 128 | 1.01049 | 1.0153 | 0 of 128 |
| ogain | up | unweighted | 0.99646 | 0.99902 | 1.0026 | 3 of 128 | 1.00519 | 1.0088 | 0 of 128 |
| ogain | up | count-weighted | 0.99638 | 0.99902 | 1.0027 | 3 of 128 | 1.00535 | 1.0090 | 0 of 128 |
| ogain | up | trace-weighted | 0.99634 | 0.99901 | 1.0027 | 3 of 128 | 1.00534 | 1.0090 | 0 of 128 |
| ogain | both | unweighted | 0.99543 | 0.99922 | 1.0038 | 3 of 128 | 1.00788 | 1.0125 | 0 of 128 |
| ogain | both | count-weighted | 0.99568 | 0.99921 | 1.0035 | 3 of 128 | 1.00828 | 1.0127 | 0 of 128 |
| ogain | both | trace-weighted | 0.99572 | 0.99920 | 1.0035 | 3 of 128 | 1.00833 | 1.0127 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.1006e-02 | [-1.1203e-02, -1.0819e-02] |
| up | -1.5201e-02 | [-1.5420e-02, -1.4975e-02] |
| both | -1.2752e-02 | [-1.2946e-02, -1.2565e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.7972e-02 | [-1.8456e-02, -1.7528e-02] | -6.9657e-03 | [-7.3692e-03, -6.5937e-03] |
| full | up | -2.6289e-02 | [-2.6798e-02, -2.5811e-02] | -1.1088e-02 | [-1.1564e-02, -1.0649e-02] |
| full | both | -2.1434e-02 | [-2.1909e-02, -2.0998e-02] | -8.6820e-03 | [-9.1090e-03, -8.2889e-03] |
| reg | gate | -1.5564e-02 | [-1.5960e-02, -1.5198e-02] | -5.3708e-03 | [-5.7091e-03, -5.0613e-03] |
| reg | up | -2.2410e-02 | [-2.2806e-02, -2.2036e-02] | -8.6068e-03 | [-8.9929e-03, -8.2515e-03] |
| reg | both | -1.8448e-02 | [-1.8827e-02, -1.8097e-02] | -6.7340e-03 | [-7.0865e-03, -6.4093e-03] |
| bias | gate | -4.1232e-03 | [-4.2169e-03, -4.0346e-03] | -2.9390e-04 | [-3.3467e-04, -2.5250e-04] |
| bias | up | -6.3154e-03 | [-6.4389e-03, -6.1955e-03] | -1.0246e-03 | [-1.0779e-03, -9.7123e-04] |
| bias | both | -5.0347e-03 | [-5.1240e-03, -4.9491e-03] | -5.9700e-04 | [-6.3602e-04, -5.5692e-04] |
| rest | gate | -1.3849e-02 | [-1.4282e-02, -1.3458e-02] | -6.6718e-03 | [-7.0458e-03, -6.3296e-03] |
| rest | up | -1.9973e-02 | [-2.0456e-02, -1.9529e-02] | -1.0063e-02 | [-1.0520e-02, -9.6473e-03] |
| rest | both | -1.6399e-02 | [-1.6845e-02, -1.5997e-02] | -8.0850e-03 | [-8.4883e-03, -7.7153e-03] |
| diag | gate | +2.7328e-02 | [+2.6787e-02, +2.7903e-02] | +2.2992e-02 | [+2.2459e-02, +2.3561e-02] |
| diag | up | +2.7782e-02 | [+2.7237e-02, +2.8366e-02] | +2.3444e-02 | [+2.2905e-02, +2.4024e-02] |
| diag | both | +2.7555e-02 | [+2.7013e-02, +2.8134e-02] | +2.3218e-02 | [+2.2684e-02, +2.3793e-02] |

## Layer 5 (IQ2_S; UD `.scale` present, min 4.09e-05, max 0.0001975)

- Gram layout: expert 20: max |strict lower| = 0, max |strict upper| = 1.804e+05
- Gram layout: expert 114: max |strict lower| = 0, max |strict upper| = 1.776e+05
- Gram layout: expert 55: max |strict lower| = 0, max |strict upper| = 1.543e+05
- Gram layout: diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = 7.08e-09; counts equal imatrix counts in 128 of 128 experts
- Gram layout: input means from blk.5.ffn_gate_up_exps.weight.in_sum (gguf shape [2304, 128]) / counts; counts equal the Gram's
- Gram layout: min over experts, files and halves of rest / full = 0.756003 (must be >= 0 up to rounding)
- Experts with tokens: 128 of 128; tokens per expert min 1169, median 8919, max 25222

| metric | half | mean | ours | UD | UD / ours | UD < ours | refined | refined / ours | refined < ours |
|---|---|---|---|---|---|---|---|---|---|
| full | gate | unweighted | 0.03455 | 0.02224 | 0.6437 | 128 of 128 | 0.01553 | 0.4495 | 126 of 128 |
| full | gate | count-weighted | 0.03504 | 0.02254 | 0.6432 | 128 of 128 | 0.01607 | 0.4585 | 126 of 128 |
| full | gate | trace-weighted | 0.03530 | 0.02273 | 0.6439 | 128 of 128 | 0.01627 | 0.4608 | 126 of 128 |
| full | up | unweighted | 0.04560 | 0.02951 | 0.6471 | 128 of 128 | 0.01985 | 0.4352 | 126 of 128 |
| full | up | count-weighted | 0.04533 | 0.02929 | 0.6462 | 128 of 128 | 0.02009 | 0.4432 | 126 of 128 |
| full | up | trace-weighted | 0.04520 | 0.02922 | 0.6465 | 128 of 128 | 0.02013 | 0.4453 | 126 of 128 |
| full | both | unweighted | 0.03884 | 0.02505 | 0.6449 | 128 of 128 | 0.01719 | 0.4424 | 126 of 128 |
| full | both | count-weighted | 0.03903 | 0.02514 | 0.6441 | 128 of 128 | 0.01760 | 0.4509 | 126 of 128 |
| full | both | trace-weighted | 0.03916 | 0.02524 | 0.6447 | 128 of 128 | 0.01774 | 0.4531 | 126 of 128 |
| reg | gate | unweighted | 0.03599 | 0.02457 | 0.6829 | 128 of 128 | 0.01951 | 0.5423 | 126 of 128 |
| reg | gate | count-weighted | 0.03648 | 0.02490 | 0.6824 | 128 of 128 | 0.02001 | 0.5484 | 126 of 128 |
| reg | gate | trace-weighted | 0.03674 | 0.02509 | 0.6830 | 128 of 128 | 0.02021 | 0.5502 | 126 of 128 |
| reg | up | unweighted | 0.04678 | 0.03209 | 0.6859 | 128 of 128 | 0.02477 | 0.5295 | 126 of 128 |
| reg | up | count-weighted | 0.04653 | 0.03187 | 0.6850 | 128 of 128 | 0.02488 | 0.5348 | 126 of 128 |
| reg | up | trace-weighted | 0.04641 | 0.03180 | 0.6853 | 128 of 128 | 0.02489 | 0.5364 | 126 of 128 |
| reg | both | unweighted | 0.04024 | 0.02752 | 0.6840 | 128 of 128 | 0.02157 | 0.5359 | 126 of 128 |
| reg | both | count-weighted | 0.04043 | 0.02762 | 0.6832 | 128 of 128 | 0.02190 | 0.5417 | 126 of 128 |
| reg | both | trace-weighted | 0.04055 | 0.02773 | 0.6837 | 128 of 128 | 0.02204 | 0.5434 | 126 of 128 |
| bias | gate | unweighted | 0.00553 | 9.828e-04 | 0.1776 | 128 of 128 | 6.027e-04 | 0.1089 | 126 of 128 |
| bias | gate | count-weighted | 0.00554 | 9.481e-04 | 0.1710 | 128 of 128 | 5.230e-04 | 0.0943 | 126 of 128 |
| bias | gate | trace-weighted | 0.00557 | 9.520e-04 | 0.1708 | 128 of 128 | 5.210e-04 | 0.0935 | 126 of 128 |
| bias | up | unweighted | 0.00717 | 0.00127 | 0.1773 | 128 of 128 | 3.578e-04 | 0.0499 | 126 of 128 |
| bias | up | count-weighted | 0.00707 | 0.00121 | 0.1705 | 128 of 128 | 2.596e-04 | 0.0367 | 126 of 128 |
| bias | up | trace-weighted | 0.00705 | 0.00120 | 0.1701 | 128 of 128 | 2.594e-04 | 0.0368 | 126 of 128 |
| bias | both | unweighted | 0.00618 | 0.00110 | 0.1776 | 128 of 128 | 5.035e-04 | 0.0815 | 126 of 128 |
| bias | both | count-weighted | 0.00615 | 0.00105 | 0.1708 | 128 of 128 | 4.142e-04 | 0.0674 | 126 of 128 |
| bias | both | trace-weighted | 0.00616 | 0.00105 | 0.1704 | 128 of 128 | 4.122e-04 | 0.0669 | 126 of 128 |
| rest | gate | unweighted | 0.02901 | 0.02126 | 0.7326 | 128 of 128 | 0.01493 | 0.5145 | 126 of 128 |
| rest | gate | count-weighted | 0.02950 | 0.02159 | 0.7320 | 128 of 128 | 0.01554 | 0.5270 | 126 of 128 |
| rest | gate | trace-weighted | 0.02972 | 0.02178 | 0.7326 | 128 of 128 | 0.01575 | 0.5297 | 126 of 128 |
| rest | up | unweighted | 0.03843 | 0.02824 | 0.7348 | 128 of 128 | 0.01949 | 0.5071 | 126 of 128 |
| rest | up | count-weighted | 0.03826 | 0.02809 | 0.7341 | 128 of 128 | 0.01983 | 0.5183 | 126 of 128 |
| rest | up | trace-weighted | 0.03815 | 0.02802 | 0.7346 | 128 of 128 | 0.01987 | 0.5208 | 126 of 128 |
| rest | both | unweighted | 0.03267 | 0.02395 | 0.7333 | 128 of 128 | 0.01668 | 0.5106 | 126 of 128 |
| rest | both | count-weighted | 0.03289 | 0.02409 | 0.7326 | 128 of 128 | 0.01718 | 0.5225 | 126 of 128 |
| rest | both | trace-weighted | 0.03300 | 0.02419 | 0.7332 | 128 of 128 | 0.01733 | 0.5252 | 126 of 128 |
| diag | gate | unweighted | 0.06372 | 0.06848 | 1.0747 | 0 of 128 | 0.09364 | 1.4696 | 0 of 128 |
| diag | gate | count-weighted | 0.06384 | 0.06857 | 1.0740 | 0 of 128 | 0.09235 | 1.4466 | 0 of 128 |
| diag | gate | trace-weighted | 0.06387 | 0.06859 | 1.0739 | 0 of 128 | 0.09216 | 1.4429 | 0 of 128 |
| diag | up | unweighted | 0.06372 | 0.06847 | 1.0746 | 0 of 128 | 0.09408 | 1.4766 | 0 of 128 |
| diag | up | count-weighted | 0.06383 | 0.06855 | 1.0739 | 0 of 128 | 0.09275 | 1.4531 | 0 of 128 |
| diag | up | trace-weighted | 0.06385 | 0.06857 | 1.0738 | 0 of 128 | 0.09254 | 1.4493 | 0 of 128 |
| diag | both | unweighted | 0.06372 | 0.06847 | 1.0746 | 0 of 128 | 0.09386 | 1.4731 | 0 of 128 |
| diag | both | count-weighted | 0.06383 | 0.06856 | 1.0740 | 0 of 128 | 0.09255 | 1.4498 | 0 of 128 |
| diag | both | trace-weighted | 0.06386 | 0.06858 | 1.0739 | 0 of 128 | 0.09235 | 1.4461 | 0 of 128 |
| plain | gate | unweighted | 0.25844 | 0.26665 | 1.0318 | 0 of 128 | 0.31888 | 1.2339 | 0 of 128 |
| plain | gate | count-weighted | 0.25846 | 0.26669 | 1.0318 | 0 of 128 | 0.31660 | 1.2249 | 0 of 128 |
| plain | gate | trace-weighted | 0.25849 | 0.26672 | 1.0319 | 0 of 128 | 0.31628 | 1.2236 | 0 of 128 |
| plain | up | unweighted | 0.25833 | 0.26652 | 1.0317 | 0 of 128 | 0.31961 | 1.2372 | 0 of 128 |
| plain | up | count-weighted | 0.25834 | 0.26655 | 1.0318 | 0 of 128 | 0.31727 | 1.2281 | 0 of 128 |
| plain | up | trace-weighted | 0.25837 | 0.26658 | 1.0318 | 0 of 128 | 0.31693 | 1.2267 | 0 of 128 |
| plain | both | unweighted | 0.25839 | 0.26658 | 1.0317 | 0 of 128 | 0.31925 | 1.2355 | 0 of 128 |
| plain | both | count-weighted | 0.25840 | 0.26662 | 1.0318 | 0 of 128 | 0.31693 | 1.2265 | 0 of 128 |
| plain | both | trace-weighted | 0.25843 | 0.26665 | 1.0318 | 0 of 128 | 0.31661 | 1.2251 | 0 of 128 |
| gain | gate | unweighted | 0.99566 | 0.99687 | 1.0012 | 2 of 128 | 0.98504 | 0.9893 | 126 of 128 |
| gain | gate | count-weighted | 0.99561 | 0.99683 | 1.0012 | 2 of 128 | 0.98545 | 0.9898 | 126 of 128 |
| gain | gate | trace-weighted | 0.99561 | 0.99683 | 1.0012 | 2 of 128 | 0.98554 | 0.9899 | 126 of 128 |
| gain | up | unweighted | 0.99566 | 0.99686 | 1.0012 | 2 of 128 | 0.98532 | 0.9896 | 126 of 128 |
| gain | up | count-weighted | 0.99561 | 0.99682 | 1.0012 | 2 of 128 | 0.98570 | 0.9900 | 126 of 128 |
| gain | up | trace-weighted | 0.99561 | 0.99681 | 1.0012 | 2 of 128 | 0.98577 | 0.9901 | 126 of 128 |
| gain | both | unweighted | 0.99566 | 0.99687 | 1.0012 | 2 of 128 | 0.98518 | 0.9895 | 126 of 128 |
| gain | both | count-weighted | 0.99561 | 0.99683 | 1.0012 | 2 of 128 | 0.98557 | 0.9899 | 126 of 128 |
| gain | both | trace-weighted | 0.99561 | 0.99682 | 1.0012 | 2 of 128 | 0.98565 | 0.9900 | 126 of 128 |
| energy | gate | unweighted | 1.05811 | 1.06485 | 1.0064 | 0 of 128 | 1.07191 | 1.0130 | 0 of 128 |
| energy | gate | count-weighted | 1.05802 | 1.06479 | 1.0064 | 0 of 128 | 1.07120 | 1.0125 | 0 of 128 |
| energy | gate | trace-weighted | 1.05803 | 1.06480 | 1.0064 | 0 of 128 | 1.07118 | 1.0124 | 0 of 128 |
| energy | up | unweighted | 1.05805 | 1.06475 | 1.0063 | 0 of 128 | 1.07293 | 1.0141 | 0 of 128 |
| energy | up | count-weighted | 1.05797 | 1.06469 | 1.0063 | 0 of 128 | 1.07211 | 1.0134 | 0 of 128 |
| energy | up | trace-weighted | 1.05798 | 1.06469 | 1.0063 | 0 of 128 | 1.07204 | 1.0133 | 0 of 128 |
| energy | both | unweighted | 1.05808 | 1.06480 | 1.0064 | 0 of 128 | 1.07242 | 1.0136 | 0 of 128 |
| energy | both | count-weighted | 1.05800 | 1.06474 | 1.0064 | 0 of 128 | 1.07166 | 1.0129 | 0 of 128 |
| energy | both | trace-weighted | 1.05801 | 1.06474 | 1.0064 | 0 of 128 | 1.07161 | 1.0129 | 0 of 128 |
| ogain | gate | unweighted | 0.99114 | 0.99856 | 1.0075 | 1 of 128 | 1.00896 | 1.0180 | 0 of 128 |
| ogain | gate | count-weighted | 0.99170 | 0.99851 | 1.0069 | 1 of 128 | 1.00889 | 1.0173 | 0 of 128 |
| ogain | gate | trace-weighted | 0.99192 | 0.99849 | 1.0066 | 1 of 128 | 1.00888 | 1.0171 | 0 of 128 |
| ogain | up | unweighted | 0.99475 | 0.99825 | 1.0035 | 1 of 128 | 1.00638 | 1.0117 | 0 of 128 |
| ogain | up | count-weighted | 0.99457 | 0.99822 | 1.0037 | 1 of 128 | 1.00617 | 1.0117 | 0 of 128 |
| ogain | up | trace-weighted | 0.99456 | 0.99822 | 1.0037 | 1 of 128 | 1.00616 | 1.0117 | 0 of 128 |
| ogain | both | unweighted | 0.99261 | 0.99845 | 1.0059 | 0 of 128 | 1.00791 | 1.0154 | 0 of 128 |
| ogain | both | count-weighted | 0.99284 | 0.99841 | 1.0056 | 0 of 128 | 1.00777 | 1.0150 | 0 of 128 |
| ogain | both | trace-weighted | 0.99296 | 0.99839 | 1.0055 | 0 of 128 | 1.00774 | 1.0149 | 0 of 128 |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:

| half | UD - ours | 95% CI |
|---|---|---|
| gate | -1.2501e-02 | [-1.2995e-02, -1.1972e-02] |
| up | -1.6041e-02 | [-1.6377e-02, -1.5713e-02] |
| both | -1.3890e-02 | [-1.4305e-02, -1.3461e-02] |

Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of refined minus ours and refined minus UD:

| metric | half | refined - ours | 95% CI | refined - UD | 95% CI |
|---|---|---|---|---|---|
| full | gate | -1.8974e-02 | [-1.9688e-02, -1.8219e-02] | -6.4728e-03 | [-6.8202e-03, -6.1371e-03] |
| full | up | -2.5243e-02 | [-2.5753e-02, -2.4753e-02] | -9.2021e-03 | [-9.5930e-03, -8.8331e-03] |
| full | both | -2.1434e-02 | [-2.1999e-02, -2.0850e-02] | -7.5440e-03 | [-7.8756e-03, -7.2357e-03] |
| reg | gate | -1.6474e-02 | [-1.7072e-02, -1.5837e-02] | -4.8885e-03 | [-5.1703e-03, -4.6217e-03] |
| reg | up | -2.1647e-02 | [-2.2054e-02, -2.1260e-02] | -6.9881e-03 | [-7.3050e-03, -6.6861e-03] |
| reg | both | -1.8532e-02 | [-1.9000e-02, -1.8047e-02] | -5.7242e-03 | [-5.9946e-03, -5.4775e-03] |
| bias | gate | -5.0212e-03 | [-5.2574e-03, -4.7726e-03] | -4.2502e-04 | [-4.9446e-04, -3.5506e-04] |
| bias | up | -6.8130e-03 | [-6.9347e-03, -6.6883e-03] | -9.4600e-04 | [-1.0033e-03, -8.9147e-04] |
| bias | both | -5.7309e-03 | [-5.9154e-03, -5.5346e-03] | -6.3510e-04 | [-6.9662e-04, -5.7575e-04] |
| rest | gate | -1.3953e-02 | [-1.4457e-02, -1.3427e-02] | -6.0478e-03 | [-6.3398e-03, -5.7706e-03] |
| rest | up | -1.8430e-02 | [-1.8872e-02, -1.8010e-02] | -8.2561e-03 | [-8.6153e-03, -7.9253e-03] |
| rest | both | -1.5703e-02 | [-1.6117e-02, -1.5284e-02] | -6.9089e-03 | [-7.1983e-03, -6.6448e-03] |
| diag | gate | +2.8509e-02 | [+2.7797e-02, +2.9267e-02] | +2.3783e-02 | [+2.3091e-02, +2.4518e-02] |
| diag | up | +2.8920e-02 | [+2.8221e-02, +2.9669e-02] | +2.4200e-02 | [+2.3522e-02, +2.4929e-02] |
| diag | both | +2.8714e-02 | [+2.8012e-02, +2.9467e-02] | +2.3991e-02 | [+2.3306e-02, +2.4724e-02] |

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

In layer 1 (both halves, count-weighted), UD's full error is 0.02472 against our 0.03898 (ratio 0.634; UD - ours -1.427e-02, 95% CI [-1.460e-02, -1.392e-02]) and its diagonal error 0.06724 against our 0.06236 (ratio 1.078); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 1 (added after registration, not part of the rule): the input-mean part is 0.00574 ours against 9.906e-04 UD (ratio 0.173), the centred rest 0.03325 against 0.02373 (ratio 0.714). In layer 2 (both halves, count-weighted), UD's full error is 0.02654 against our 0.03958 (ratio 0.670; UD - ours -1.304e-02, 95% CI [-1.333e-02, -1.279e-02]) and its diagonal error 0.06634 against our 0.06179 (ratio 1.074); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 2 (added after registration, not part of the rule): the input-mean part is 0.00554 ours against 0.00103 UD (ratio 0.185), the centred rest 0.03404 against 0.02551 (ratio 0.750). In layer 3 (both halves, count-weighted), UD's full error is 0.02738 against our 0.04013 (ratio 0.682; UD - ours -1.275e-02, 95% CI [-1.295e-02, -1.257e-02]) and its diagonal error 0.06645 against our 0.06212 (ratio 1.070); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 3 (added after registration, not part of the rule): the input-mean part is 0.00543 ours against 9.959e-04 UD (ratio 0.183), the centred rest 0.03470 against 0.02638 (ratio 0.760). In layer 5 (both halves, count-weighted), UD's full error is 0.02514 against our 0.03903 (ratio 0.644; UD - ours -1.389e-02, 95% CI [-1.431e-02, -1.346e-02]) and its diagonal error 0.06856 against our 0.06383 (ratio 1.074); UD is lower in 128 of 128 experts on full and 0 of 128 on diag. Split of the full error in layer 5 (added after registration, not part of the rule): the input-mean part is 0.00615 ours against 0.00105 UD (ratio 0.171), the centred rest 0.03289 against 0.02409 (ratio 0.733). Under the registered rule this is "supported" for both halves together, count-weighted; across the three means and three halves the branches are: supported.

## refined against ours and UD

In layer 1 (both halves, count-weighted), refined's full error is 0.01706 against our 0.03898 and UD's 0.02472 (refined - ours -2.192e-02, 95% CI [-2.239e-02, -2.148e-02]; (ours - refined) / (ours - UD) = 1.537; lower than ours in 125 of 128 experts). Bias part 3.353e-04 (ours 0.00574, UD 9.906e-04), rest 0.01672 (ours 0.03325, UD 0.02373). Regularised (the search objective) 0.02146 (ours 0.04034, UD 0.02720). Diagonal 0.09218 (ours 0.06236, UD 0.06724), plain 0.32570 (ours 0.25999, UD 0.26788). Gains (ours / UD / refined): plain 0.9966 / 0.9978 / 0.9856, energy 1.0608 / 1.0673 / 1.0775, output 0.9933 / 0.9991 / 1.0072. In layer 2 (both halves, count-weighted), refined's full error is 0.01803 against our 0.03958 and UD's 0.02654 (refined - ours -2.155e-02, 95% CI [-2.200e-02, -2.113e-02]; (ours - refined) / (ours - UD) = 1.653; lower than ours in 126 of 128 experts). Bias part 3.794e-04 (ours 0.00554, UD 0.00103), rest 0.01765 (ours 0.03404, UD 0.02551). Regularised (the search objective) 0.02236 (ours 0.04090, UD 0.02892). Diagonal 0.09004 (ours 0.06179, UD 0.06634), plain 0.33097 (ours 0.26185, UD 0.26939). Gains (ours / UD / refined): plain 0.9976 / 0.9986 / 0.9868, energy 1.0638 / 1.0698 / 1.0832, output 0.9952 / 0.9993 / 1.0082. In layer 3 (both halves, count-weighted), refined's full error is 0.01870 against our 0.04013 and UD's 0.02738 (refined - ours -2.143e-02, 95% CI [-2.191e-02, -2.100e-02]; (ours - refined) / (ours - UD) = 1.681; lower than ours in 126 of 128 experts). Bias part 3.989e-04 (ours 0.00543, UD 9.959e-04), rest 0.01830 (ours 0.03470, UD 0.02638). Regularised (the search objective) 0.02300 (ours 0.04144, UD 0.02973). Diagonal 0.08967 (ours 0.06212, UD 0.06645), plain 0.32736 (ours 0.26155, UD 0.26863). Gains (ours / UD / refined): plain 0.9975 / 0.9984 / 0.9871, energy 1.0634 / 1.0690 / 1.0813, output 0.9957 / 0.9992 / 1.0083. In layer 5 (both halves, count-weighted), refined's full error is 0.01760 against our 0.03903 and UD's 0.02514 (refined - ours -2.143e-02, 95% CI [-2.200e-02, -2.085e-02]; (ours - refined) / (ours - UD) = 1.543; lower than ours in 126 of 128 experts). Bias part 4.142e-04 (ours 0.00615, UD 0.00105), rest 0.01718 (ours 0.03289, UD 0.02409). Regularised (the search objective) 0.02190 (ours 0.04043, UD 0.02762). Diagonal 0.09255 (ours 0.06383, UD 0.06856), plain 0.31693 (ours 0.25840, UD 0.26662). Gains (ours / UD / refined): plain 0.9956 / 0.9968 / 0.9856, energy 1.0580 / 1.0647 / 1.0717, output 0.9928 / 0.9984 / 1.0078.

Full error (both halves) by Gram token count; an expert with fewer than 2304 tokens has a rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):

| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | refined (count-w.) | refined / ours | refined / UD |
|---|---|---|---|---|---|---|---|---|
| 1 | < 2304 | 3 | 0.004 | 0.04420 | 0.03053 | 0.04420 | 1.000 | 1.448 |
| 1 | >= 2304 | 125 | 0.996 | 0.03896 | 0.02470 | 0.01696 | 0.435 | 0.687 |
| 2 | < 2304 | 2 | 0.003 | 0.03960 | 0.02624 | 0.03960 | 1.000 | 1.509 |
| 2 | >= 2304 | 126 | 0.997 | 0.03958 | 0.02654 | 0.01796 | 0.454 | 0.677 |
| 3 | < 2304 | 2 | 0.002 | 0.03985 | 0.02604 | 0.03985 | 1.000 | 1.530 |
| 3 | >= 2304 | 126 | 0.998 | 0.04013 | 0.02738 | 0.01865 | 0.465 | 0.681 |
| 5 | < 2304 | 2 | 0.003 | 0.04361 | 0.02788 | 0.04361 | 1.000 | 1.564 |
| 5 | >= 2304 | 126 | 0.997 | 0.03902 | 0.02514 | 0.01753 | 0.449 | 0.697 |

Refinement log `models/refined/gu-refined-mix-1-2-3-5.gguf.json` (scripts/refine-codes.py; wall time 3913 s for 512 expert tensors, workers 4, sweeps <= 4, tol 0.0001):

| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 125 | 3 | 4: 125 | 60453477 | 2591430 of 2592000 | 4334 | 5.04e-10 | 3.87e-10 | yes |
| 2 | 126 | 2 | 4: 126 | 61602631 | 2612103 of 2612736 | 3414 | 4.74e-10 | 4.91e-10 | yes |
| 3 | 126 | 2 | 4: 126 | 61278501 | 2612143 of 2612736 | 3224 | 5.01e-10 | 4.82e-10 | yes |
| 5 | 126 | 2 | 4: 126 | 58499930 | 2611853 of 2612736 | 4571 | 6.25e-10 | 4.12e-10 | yes |

Search settings from the log: shrink 0.1, min tokens 2304; 'J never rose' is under the search objective; the max abs columns compare the log's own full-G relative errors with the ones computed here. layer 1: max abs(log - here) on reg 4.79e-10 before, 3.76e-10 after; layer 2: max abs(log - here) on reg 4.58e-10 before, 4.82e-10 after; layer 3: max abs(log - here) on reg 4.75e-10 before, 4.64e-10 after; layer 5: max abs(log - here) on reg 5.97e-10 before, 3.98e-10 after. Skipped experts keep the source bytes (reason: 'N tokens < 2304').
