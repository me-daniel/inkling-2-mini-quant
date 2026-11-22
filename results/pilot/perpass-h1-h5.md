# Per-pass controls H1 to H5 (Addendum H)

Written by `scripts/perpass-summary.py` from files in `results/pilot/`. Exploratory pilot. KL against the IQ4_NL base on S (16 chunks of 4,096 tokens), P = 3. Intervals are 95% percentile bootstraps over chunks (10,000 resamples). r is the Fisher residual share ||d_all - sum_k d_k|| / ||d_all||; the norm ratio is ||d_all|| / ||sum_k d_k||; both are 0 and 1 when the parts add up exactly.

## Completeness

- H1: 4 of 4 conditions scored, `layersplit3bit-residual.json` present.
- H2: 4 of 4 conditions scored, `noise-tied-residual.json` present.
- H3: 4 of 4 conditions scored, `noise-indep-residual.json` present.
- H4: 6 of 6 conditions scored, no residual registered.
- H5: 4 of 4 conditions scored, `perpass-b3-residual.json` present.
- Nothing registered is missing.
- Not computable, because the logits were deleted before these were stored: per-chunk KL of the 3-bit pass split (only means are kept, `perpass2-kl.tsv`), so its KL ratio below is a point estimate; and per-chunk sums of squares for any group, so no bootstrap of a paired difference in r or norm ratio between two groups.
- H4's other orders (indep-231, indep-312) have no single-pass runs of their own (draw 2 on pass 1, draw 3 on pass 2 and so on were never run), so they have KL only, no residual. This is as registered.

## Findings

1. H1, layer split at the 3-bit rung. Registered reading: "the Addendum F rule at 3 bits, r and norm ratio of the 3-bit pass split (on record) against this split, beyond both 95% CIs". Pass split r 0.748 [0.732, 0.761], layer split r 1.108 [1.074, 1.145]; pass split norm ratio 0.788 [0.780, 0.794], layer split 0.640 [0.627, 0.652]. The r intervals do not overlap and the norm-ratio intervals do not overlap, so the rule is met. Per chunk, the layer split has the larger r in 16 of 16 chunks. KL all / sum of parts is 0.616 [0.591, 0.638] for the layer split and 0.875 (point only) for the pass split. At 2 bits (on record) the same order held: r 0.675 [0.657, 0.693] against 0.838 [0.822, 0.855]. Both partitions share the same all-on run (3b-all, KL 0.00809); the pass split is the closer to additive.
2. H2, tied Gaussian noise by pass (draw 1 on pass 1, 2 or 3 only, and on all passes). No reading rule is registered; descriptive. Single-pass KL p1 0.01883, p2 0.02031, p3 0.01652, all 0.06353. KL all / sum of parts 1.141 [1.119, 1.162]. r 0.680 [0.660, 0.702], norm ratio 0.836 [0.825, 0.847], pairwise Fisher cosines 0.320 [0.297, 0.343], 0.362 [0.348, 0.376], 0.138 [0.121, 0.154] (passes 1-2, 2-3, 1-3). The 2-bit quantization split on the same block (on record) gives r 0.675 [0.657, 0.693] and KL all / sum 1.135 [1.112, 1.162]: tied noise of the same relative error behaves like the quantization error.
3. H3, independent noise by pass (draw k on pass k only). Registered reading: "KL(noise-indep) / sum of the three single-pass KLs, paired chunk bootstrap; 'about additive' if its 95% CI contains 1". Ratio 0.953 [0.934, 0.971]: the CI excludes 1, so the registered reading does not call it about additive (sub-additive by 4.7%). Its single-pass KLs sum to 0.05572, against 0.05566 for the tied draws, so tied and independent differ in how the visits combine, not in what one visit costs. r 0.723 [0.709, 0.736], norm ratio 0.807 [0.798, 0.817], cosines 0.267 [0.240, 0.293], 0.216 [0.200, 0.232], 0.111 [0.098, 0.125].
4. H4, tied against independent noise over every pair of draws. Registered reading: "the tied/independent ratio is then reported for every pair of the 3 tied and 3 independent draws; 'tied costs more' is stated only if all 9 ratios are above 1". 9 ratios from 1.167 to 1.223; lowest lower CI bound 1.146. The rule is met: tied costs more. Tied KL 0.06353, 0.06488, 0.06493; independent KL 0.05311, 0.05444, 0.05388. The table follows the findings.
5. H5, the 2-bit pass split on B3 (layers 12-17). Registered: "same statistics as B2, descriptive". B3 against B2 (on record): r 0.539 [0.526, 0.553] against 0.675 [0.657, 0.693]; norm ratio 0.906 [0.897, 0.915] against 0.841 [0.829, 0.855]; KL all / sum 1.226 [1.209, 1.244] against 1.135 [1.112, 1.162]; share of the last pass ||d_3|| / ||d_all|| 0.629 [0.623, 0.634] against 0.514 [0.504, 0.525]. Single-pass KL on B3 p1 0.01661, p2 0.02069, p3 0.03244 (all 0.08548); on B2 p1 0.01885, p2 0.01983, p3 0.01591 (all 0.06198).
6. Across the visit splits, KL all / sum of the 3 single-visit KLs is B2 2-bit quantization (on record) 1.135 [1.112, 1.162]; B2 tied noise (H2) 1.141 [1.119, 1.162]; B2 independent noise (H3) 0.953 [0.934, 0.971]; B3 2-bit quantization (H5) 1.226 [1.209, 1.244]; 3-bit quantization on B2 0.875 (point only). To second order in the logit change, error that counts once gives about 0.33, independent per-visit error about 1, and 3 identical, aligned visit errors about 3.

H4, KL(tied) / KL(independent), paired bootstrap over chunks:

| tied \ independent | noise-indep | indep-231 | indep-312 |
|---|---|---|---|
| noise-tied | 1.196 [1.169, 1.225] | 1.167 [1.146, 1.188] | 1.179 [1.154, 1.202] |
| noise-tied2 | 1.222 [1.187, 1.256] | 1.192 [1.161, 1.219] | 1.204 [1.162, 1.246] |
| noise-tied3 | 1.223 [1.182, 1.262] | 1.193 [1.154, 1.230] | 1.205 [1.173, 1.238] |

Noise amplitude (`noise-t.tsv`, mean relative l2 error per expert on B2): ffn_gate_up_exps 2-bit 0.1288, draws 0.1289, 0.1289, 0.1289; ffn_down_exps 2-bit 0.5292, draws 0.5303, 0.5303, 0.5303.

Sources: H1 `layersplit3bit-residual.json`, `perpass3bit-residual.json`, `perpass2-kl.tsv`, `perpass3-kl.tsv`; H2 `noise-tied-residual.json`; H3 `noise-indep-residual.json`; H4 `perpass-kl.tsv` (tied draws 2 and 3) and `perpass3-kl.tsv`; H5 `perpass-b3-residual.json`, `perpass-residual.json`, `perpass-kl.tsv` (B2 runs p1, p2, p3, all).

## Limits

- One block per question: B2 (layers 6-11) for H1 to H4, B3 (layers 12-17) for H5. Nothing here covers layers 6-17 changed together.
- One amplitude per group: the 2-bit level (IQ2_XXS gate_up, stock Q2_0 down), the 3-bit level (gate_up only) and noise matched to the 2-bit relative error. r and the norm ratio depend on the level (finding 1).
- One capture: S, 16 chunks, one Q8_0 reference, one IQ4_NL base, one engine commit, P = 3 only. The intervals cover chunk sampling only, not seeds, documents or builds.
- Three noise draws. H4 permutes the same three draws, so its 9 ratios are not 9 replications; fresh seed triplets were not run.
- H1's layer split is not a control without weight sharing: each layer pair still runs on all three visits. It shows that the partition matters, not that the loop alone causes the difference.
- The H3 KL ratio mixes two things. With exactly additive logit changes it would still exceed 1 by the cross-pass Fisher terms, which are positive here; the logit-level residual r is large in every group.
- The logits files floor log-probs below max - 16 nats; the Fisher norm gives the floored tail little weight.
- No route replay and no tiny-amplitude rung, so the cause of the residual (router flips or smooth nonlinearity) is untested.
- Tied draws 2 and 3 come from the 21 Nov runs in `perpass-kl.tsv`. Their 22 Nov reruns (logs/chain12.out, logs/macgate-rerun.out) never entered a table and their logits are gone, so the two could not be compared.
- `analysis/per_pass.py` and `analysis/noise.py` were written by Claude and are not yet reviewed by Daniel; three analysis choices (Fisher norm, 10-nat head, pooled ratio) await their change-log approval.

## What it means for the thesis question

On these documents at P = 3, quantization error in a looped block counts roughly once per visit: changing all 3 visits costs 0.87 to 1.23 times the sum of the single-visit costs (finding 6), far from the 0.33 that counting once would give. Shared weights add a modest positive interaction: the same noise on every visit costs 1.17 to 1.22 times a fresh draw per visit, although the single-visit runs cost the same (KL sums 0.0557 and 0.0557). Nothing compounds strongly: the ratios stay far below the 3 of fully aligned visit errors, and the logit changes combine to less than their vector sum (norm ratio 0.84 on B2, 0.91 on B3). For an allocator, a bit in a looped layer is priced roughly per visit with a block-dependent surcharge, and the costliest visit (pass 2 on B2 at 2 bits, pass 1 at 3 bits, pass 3 on B3) changes with block and level, so one visit weight will not fit all of them.
