# gate_up weight error, UD-IQ3_XXS against ours at the same type

Tables written by `scripts/gu-error.py --report` from `results/pilot/gu-error.tsv`. Sample: experts 0:128:8 (16 of 128): [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120].
rel = ||W_q - W|| / ||W|| against the BF16 file, rel_w the same weighted per input column by our imatrix (in_sum2 / counts), both averaged over the sampled experts. UD's dequantized values are multiplied by its per-expert `.scale` where present. Our row: cand where cand's tensor is its own encoding, else the pool file at that type (cand rows identical to a pool tensor are not repeated).

## IQ2_S

| layer | our file | UD rel | our rel | UD - ours | UD rel_w | our rel_w | UD - ours |
|---|---|---|---|---|---|---|---|
| 1 | (none of ours at IQ2_S) | 0.26775 | | | 0.25916 | | |
| 2 | (none of ours at IQ2_S) | 0.26948 | | | 0.25765 | | |
| 3 | (none of ours at IQ2_S) | 0.26871 | | | 0.25750 | | |
| 5 | (none of ours at IQ2_S) | 0.26650 | | | 0.26231 | | |
| 6 | cand | 0.26819 | 0.26141 | +0.00678 | 0.25671 | 0.24792 | +0.00879 |
| 7 | cand | 0.26967 | 0.26281 | +0.00686 | 0.25706 | 0.24811 | +0.00895 |
| 8 | cand | 0.26800 | 0.26213 | +0.00587 | 0.25583 | 0.24800 | +0.00783 |
| 12 | (none of ours at IQ2_S) | 0.26365 | | | 0.26052 | | |
| 18 | (none of ours at IQ2_S) | 0.26404 | | | 0.26266 | | |

UD higher than ours in 3 of 3 same-type layers (rel), 3 of 3 (rel_w).

## IQ2_XS

| layer | our file | UD rel | our rel | UD - ours | UD rel_w | our rel_w | UD - ours |
|---|---|---|---|---|---|---|---|
| 4 | (none of ours at IQ2_XS) | 0.32204 | | | 0.31003 | | |

UD higher than ours in 0 of 0 same-type layers (rel), 0 of 0 (rel_w).

## IQ2_XXS

| layer | our file | UD rel | our rel | UD - ours | UD rel_w | our rel_w | UD - ours |
|---|---|---|---|---|---|---|---|
| 0 | pool-2bit | 0.36803 | 0.35150 | +0.01653 | 0.35798 | 0.33921 | +0.01877 |

UD higher than ours in 1 of 1 same-type layers (rel), 1 of 1 (rel_w).

## IQ3_XXS

| layer | our file | UD rel | our rel | UD - ours | UD rel_w | our rel_w | UD - ours |
|---|---|---|---|---|---|---|---|
| 9 | pool-3bit | 0.19339 | 0.18723 | +0.00616 | 0.18468 | 0.17789 | +0.00679 |
| 10 | pool-3bit | 0.19194 | 0.18616 | +0.00578 | 0.18655 | 0.17976 | +0.00679 |
| 11 | pool-3bit | 0.19219 | 0.18554 | +0.00665 | 0.18788 | 0.18012 | +0.00776 |
| 13 | pool-3bit | 0.19091 | 0.18476 | +0.00615 | 0.18795 | 0.18071 | +0.00724 |
| 14 | pool-3bit | 0.19102 | 0.18473 | +0.00629 | 0.18877 | 0.18098 | +0.00779 |
| 15 | pool-3bit | 0.19131 | 0.18488 | +0.00643 | 0.18862 | 0.18081 | +0.00781 |
| 16 | pool-3bit | 0.19307 | 0.18637 | +0.00670 | 0.18748 | 0.17900 | +0.00848 |
| 17 | pool-3bit | 0.19292 | 0.18667 | +0.00625 | 0.18521 | 0.17747 | +0.00774 |
| 19 | pool-3bit | 0.19011 | 0.18418 | +0.00593 | 0.18859 | 0.18058 | +0.00801 |
| 20 | pool-3bit | 0.18961 | 0.18316 | +0.00645 | 0.18775 | 0.17958 | +0.00817 |
| 21 | pool-3bit | 0.18973 | 0.18366 | +0.00607 | 0.18642 | 0.17862 | +0.00780 |
| 22 | pool-3bit | 0.19046 | 0.18410 | +0.00636 | 0.18517 | 0.17752 | +0.00765 |
| 23 | pool-3bit | 0.19075 | 0.18465 | +0.00610 | 0.18344 | 0.17563 | +0.00781 |

UD higher than ours in 13 of 13 same-type layers (rel), 13 of 13 (rel_w).

## Types and bytes per layer (whole tensor, from the files)

| layer | UD type | cand type | UD bytes | cand bytes | UD - cand |
|---|---|---|---|---|---|
| 0 | IQ2_XXS | IQ2_XXS | 175,177,728 | 175,177,728 | +0 |
| 1 | IQ2_S | IQ2_XXS | 217,645,056 | 175,177,728 | +42,467,328 |
| 2 | IQ2_S | IQ2_XXS | 217,645,056 | 175,177,728 | +42,467,328 |
| 3 | IQ2_S | IQ2_XXS | 217,645,056 | 175,177,728 | +42,467,328 |
| 4 | IQ2_XS | IQ2_XXS | 196,411,392 | 175,177,728 | +21,233,664 |
| 5 | IQ2_S | IQ2_XXS | 217,645,056 | 175,177,728 | +42,467,328 |
| 6 | IQ2_S | IQ2_S | 217,645,056 | 217,645,056 | +0 |
| 7 | IQ2_S | IQ2_S | 217,645,056 | 217,645,056 | +0 |
| 8 | IQ2_S | IQ2_S | 217,645,056 | 217,645,056 | +0 |
| 9 | IQ3_XXS | IQ2_S | 260,112,384 | 217,645,056 | +42,467,328 |
| 10 | IQ3_XXS | IQ2_S | 260,112,384 | 217,645,056 | +42,467,328 |
| 11 | IQ3_XXS | IQ2_S | 260,112,384 | 217,645,056 | +42,467,328 |
| 12 | IQ2_S | IQ3_XXS | 217,645,056 | 260,112,384 | -42,467,328 |
| 13 | IQ3_XXS | IQ3_XXS | 260,112,384 | 260,112,384 | +0 |
| 14 | IQ3_XXS | IQ3_XXS | 260,112,384 | 260,112,384 | +0 |
| 15 | IQ3_XXS | IQ3_XXS | 260,112,384 | 260,112,384 | +0 |
| 16 | IQ3_XXS | IQ3_XXS | 260,112,384 | 260,112,384 | +0 |
| 17 | IQ3_XXS | IQ3_XXS | 260,112,384 | 260,112,384 | +0 |
| 18 | IQ2_S | IQ3_XXS | 217,645,056 | 260,112,384 | -42,467,328 |
| 19 | IQ3_XXS | IQ3_XXS | 260,112,384 | 260,112,384 | +0 |
| 20 | IQ3_XXS | IQ3_XXS | 260,112,384 | 260,112,384 | +0 |
| 21 | IQ3_XXS | IQ3_XXS | 260,112,384 | 260,112,384 | +0 |
| 22 | IQ3_XXS | IQ3_XXS | 260,112,384 | 260,112,384 | +0 |
| 23 | IQ3_XXS | IQ3_XXS | 260,112,384 | 260,112,384 | +0 |
| all | | | 5,711,855,616 | 5,478,285,312 | +233,570,304 |

Byte difference by type change (cand -> UD):

| cand -> UD | layers | UD - cand bytes |
|---|---|---|
| IQ2_XXS -> IQ2_S | 1, 2, 3, 5 | +169,869,312 |
| IQ2_XXS -> IQ2_XS | 4 | +21,233,664 |
| IQ2_S -> IQ3_XXS | 9, 10, 11 | +127,401,984 |
| IQ3_XXS -> IQ2_S | 12, 18 | -84,934,656 |

## Answer (hand-written from the tables above)

At the same type, UD's gate_up tensors are further from BF16 than ours in all 17 layers where both exist: 13 of 13 at IQ3_XXS (UD - pool-3bit +0.00578 to +0.00670 rel, +0.00679 to +0.00848 rel_w), 3 of 3 at IQ2_S (cand layers 6-8, +0.00587 to +0.00686 rel) and layer 0 at IQ2_XXS (+0.01653 rel). The gap holds in layers 20-23, where UD has no `.scale` tensor, so it is not an artifact of how we apply the scale. Their advantage on S is therefore not better quantization at a given type, at least by plain or our-imatrix-weighted weight error; it lines up with the choice of types per layer. UD's gate_up is 233,570,304 bytes larger than cand's in total. The difference comes from four changes: IQ2_XXS to IQ2_S in layers 1, 2, 3 and 5 (+169,869,312), IQ2_XXS to IQ2_XS in layer 4 (+21,233,664) and IQ2_S to IQ3_XXS in layers 9-11 (+127,401,984), partly paid for by dropping layers 12 and 18 from IQ3_XXS to IQ2_S (-84,934,656).

Inference, not measured here: since most of the extra bytes go to layers 1-5 and 9-11, the gu-ud KL drop is probably bought with bits in the early layers (1-5) and in looped layers 9-11, which run three times per token at P = 3, not with a better encoder. Two things this test cannot exclude: UD may have used an imatrix that weights columns differently from ours (for example one pooled over loop visits), and weight error is not output error. A splice that copies only UD's layer 1-5 or 9-11 gate_up tensors into cand would test the allocation reading directly.
