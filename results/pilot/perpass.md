# Per-pass test (pilot run 4, Addenda A, B, F)

Written by `analysis/perpass_report.py`. Exploratory. Block B2 (layers 6-11) experts at the 2-bit level (IQ2_XXS gate_up, stock Q2_0 down) on chosen passes, on the IQ4_NL base, at P = 3, on S (16 x 4,096 tokens). On these documents, at P = 3, changing B2's experts on pass k produced the measured logit changes below.

## Scope

- Amplitude-specific (Addendum F). The 2-bit level's mean relative l2 weight error is 0.5292 on ffn_down_exps, 0.1288 on ffn_gate_up_exps (`results/pilot/noise-t.tsv`, column t_2bit). Every 2-bit number below holds for this amplitude only.
- B2 only. B2 is layers 6-11, half of the loop core (layers 6-17). B3 (layers 12-17) was not perturbed.
- One noise draw per pass. noise-tied puts draw 1 on all three passes. noise-indep puts draws 1, 2 and 3 on passes 1, 2 and 3. The tied/independent CI covers chunk sampling only, not the draw.
- Loop-specific reading only by the Addendum F rule. Addendum F: a loop-specific claim needs the pass-split r and norm ratio to differ from the layer-split values beyond both CIs. The layer-split control has run; the outcome is under "What the controls allow".

## Parity

parity: pp-noop == pp-base and pp-allslots == pp-all, byte-identical (cmp, 2026-11-21 07:49)

## KL per condition

Mean KL over the 16 chunks from `results/pilot/perpass-kl.tsv` (`scripts/kldfile.py`).

| Condition | KL vs IQ4_NL base | KL vs Q8_0 reference |
|---|---|---|
| base | -0.00000 | 0.01606 |
| noop | deleted | deleted |
| p1 | 0.01885 | 0.02933 |
| p2 | 0.01983 | 0.03215 |
| p3 | 0.01591 | 0.02995 |
| all | 0.06198 | 0.07275 |
| allslots | deleted | deleted |
| noise-tied | 0.06353 | 0.07566 |
| noise-indep | 0.05311 | 0.06513 |

KL ratios against the base, paired bootstrap over chunks (seed 0):

| Ratio | Point [95% CI] | Per-chunk range |
|---|---|---|
| all / sum of single passes | 1.135 [1.112, 1.162] | 1.058-1.267 |
| all / mean single pass | 3.406 [3.335, 3.485] | 3.173-3.801 |
| p1 / p3 | 1.185 [1.096, 1.278] | 0.806-1.659 |
| p2 / p3 | 1.247 [1.169, 1.335] | 0.981-1.647 |

Three additivity comparators, all against the sum of the single passes: KL all / sum 1.135 [1.112, 1.162]; squared Fisher ||all||^2 / sum_k ||d_k||^2 1.088 [1.064, 1.117]; Fisher vector norm ratio ||all|| / ||sum_k d_k|| 0.841 [0.829, 0.855]. Values above 1 read super-additive and values below 1 sub-additive, so the comparators disagree. This report does not choose among them. That choice belongs to the H1 analysis.

## Residual statistics

From `results/pilot/perpass-residual.json` (`analysis/per_pass.py`). d_k = logp(pass k only) - logp(base), R = d_all - sum_k d_k, pooled over all scored tokens as a ratio of sums. Point estimate [95% percentile bootstrap over chunks].

Fisher is the primary norm. The registered prediction asks for centred Euclidean over the full vocabulary. That form cannot be computed faithfully from these files: the writer floors every value more than 16 nats below the top, and the floored entries hold the floor, not the model's log-probs. The head Euclidean column (entries within 10 nats of the base's top) is a head-restricted substitute for it. The full-vocabulary column, where present, keeps the floored entries as written. Three choices deviate from CHANGELOG.md: the Fisher primary norm, the 10-nat head, and a pooled ratio of sums instead of run 4's per-token ratio. They were fixed in code before the first per-pass logits existed but are not yet recorded in CHANGELOG.md (`reviews/2026-11-21-audit-perpass-opus.md`, finding 3).

| Statistic | Fisher | head Euclidean (substitute) |
|---|---|---|
| r = ||R|| / ||all|| | 0.675 [0.657, 0.693] | 0.740 [0.717, 0.762] |
| norm ratio ||all|| / ||sum_k d_k|| | 0.841 [0.829, 0.855] | 0.805 [0.790, 0.821] |
| ||d1|| / ||d3|| | 1.103 [1.059, 1.148] | 0.976 [0.942, 1.008] |
| ||d2|| / ||d3|| | 1.122 [1.082, 1.166] | 0.999 [0.971, 1.026] |
| ||d3|| / ||all|| | 0.514 [0.504, 0.525] | 0.584 [0.570, 0.596] |
| cos(d1, d2) | 0.320 [0.296, 0.341] | 0.319 [0.301, 0.337] |
| cos(d2, d3) | 0.347 [0.331, 0.360] | 0.333 [0.321, 0.346] |
| cos(d1, d3) | 0.134 [0.120, 0.147] | 0.152 [0.137, 0.166] |
| visit multiplier ||all||^2 / mean_k ||d_k||^2 | 3.264 [3.193, 3.350] | 2.986 [2.904, 3.080] |
| ||all||^2 / sum_k ||d_k||^2 | 1.088 [1.064, 1.117] | 0.995 [0.968, 1.027] |

Fisher r per chunk: min 0.592, median 0.669, max 0.746 (16 chunks).

Norm sensitivity on chunks 0 and 1 only, from `reviews/2026-11-21-audit-perpass-opus/audit_check.json` (the audit's CPU reread of the raw logits; no CI). A 16-chunk full-vocabulary column needs a CPU rerun of `analysis/per_pass.py` on the stored logits.

| Norm | r (c0, c1) | norm ratio (c0, c1) | p1/p3 (c0, c1) | p2/p3 (c0, c1) |
|---|---|---|---|---|
| Fisher | 0.622, 0.668 | 0.856, 0.899 | 1.019, 0.907 | 1.057, 0.990 |
| full vocabulary (floor included) | 0.672, 0.709 | 0.892, 0.873 | 0.915, 0.844 | 0.985, 0.929 |
| head 5 nats | 0.664, 0.669 | 0.836, 0.853 | 0.980, 0.870 | 1.027, 0.923 |
| head 10 nats (reported) | 0.697, 0.719 | 0.820, 0.823 | 0.907, 0.837 | 0.979, 0.875 |
| head 14 nats | 0.722, 0.745 | 0.812, 0.813 | 0.887, 0.832 | 0.954, 0.866 |

The head is unweighted. On these chunks a token's 10-nat head holds 1 to 6796 entries (median 154, 136), so tokens with flat distributions weigh more in the head norm than in the Fisher norm.

## Tied against independent noise (Addendum A)

Relative l2 error per expert tensor, mean over experts, from `results/pilot/noise-t.tsv`:

| Tensor | 2-bit | seed1 | seed2 | seed3 |
|---|---|---|---|---|
| ffn_down_exps | 0.5292 | 0.5303 | 0.5303 | 0.5303 |
| ffn_gate_up_exps | 0.1288 | 0.1289 | 0.1289 | 0.1289 |

From `results/pilot/noise-ratio.txt` (`analysis/noise.py`): KL tied 0.06353, independent 0.05311, ratio 1.196 [1.169, 1.225] (paired bootstrap over chunks).
KL tied / KL all (2-bit): 1.025 [0.997, 1.056].

Each condition is one draw per pass, so tying is confounded with which draws sit on passes 2 and 3. The CI covers chunk sampling only. Not yet run: noise-tied with draws 2 and 3 (two splices and two llama-perplexity runs); the spread of the three tied KLs bounds the draw effect.

Addendum A also predicts independent ~ sum of single-pass effects. This prediction is not tested: no Gaussian single-pass condition was run, and the 2-bit single passes cannot stand in for one (registered prediction, point 7).

## Registered predictions

From `reviews/2026-11-21-prediction-perpass-gpt-6.2-astra.md`, section 6. Quantization items scored on the head Euclidean substitute (the registered full-vocabulary form is not computable, see Residual statistics), the noise item on pooled mean KL. The head form is used only to score the registration.

| Prediction | Range | Measured [95% CI] | Point | CI |
|---|---|---|---|---|
| 6.1 pass-1/pass-3 ratio | 0.05-0.30 | 0.976 [0.942, 1.008] | outside | outside |
| 6.1 pass-2/pass-3 ratio | 0.25-0.65 | 0.999 [0.971, 1.026] | outside | outside |
| 6.2 r | 0.08-0.20 | 0.740 [0.717, 0.762] | outside | outside |
| 6.3 norm ratio | 0.90-1.10 | 0.805 [0.790, 0.821] | outside | outside |
| 6.4 tied / independent KL (one draw per pass) | 1.05-1.35 | 1.196 [1.169, 1.225] | inside | inside |

Pass ordering by measure. The registration predicts ||d3|| > ||d2|| > ||d1||, i.e. both ratios below 1. The pass ordering depends on the measure. KL ratios compare squared magnitudes (KL ~ 1/2 Fisher^2); the norm rows compare norms.

| Measure | p1 / p3 [95% CI] | p2 / p3 [95% CI] |
|---|---|---|
| KL (chunk means) | 1.185 [1.096, 1.278], CI above 1 | 1.247 [1.169, 1.335], CI above 1 |
| Fisher | 1.103 [1.059, 1.148], CI above 1 | 1.122 [1.082, 1.166], CI above 1 |
| head Euclidean (substitute) | 0.976 [0.942, 1.008], CI contains 1 | 0.999 [0.971, 1.026], CI contains 1 |

## Layer-split control

The 2-bit block on layers 6-7 / 8-9 / 10-11 only, all passes; the three sum to the `all` condition. Here d1, d2, d3 are the three layer pairs, not passes. Addendum F: a loop-specific claim needs the pass-split r and norm ratio to differ from the layer-split values beyond both CIs.

Residual statistics from `results/pilot/layersplit-residual.json` (95% bootstrap CI over chunks):

| Statistic | Fisher | head Euclidean (substitute) | full vocabulary (floor included) |
|---|---|---|---|
| r = ||R|| / ||all|| | 0.838 [0.822, 0.855] | 0.893 [0.872, 0.913] | 0.821 [0.798, 0.845] |
| norm ratio ||all|| / ||sum_k d_k|| | 0.759 [0.748, 0.770] | 0.735 [0.722, 0.748] | 0.799 [0.783, 0.815] |
| ||d1|| / ||d3|| | 0.818 [0.789, 0.846] | 0.797 [0.780, 0.814] | 0.782 [0.752, 0.817] |
| ||d2|| / ||d3|| | 1.073 [1.043, 1.103] | 1.038 [1.013, 1.063] | 1.036 [1.005, 1.067] |
| ||d3|| / ||all|| | 0.645 [0.635, 0.656] | 0.677 [0.666, 0.687] | 0.605 [0.590, 0.620] |
| cos(d1, d2) | 0.247 [0.226, 0.269] | 0.255 [0.238, 0.272] | 0.299 [0.277, 0.322] |
| cos(d2, d3) | 0.256 [0.233, 0.280] | 0.257 [0.237, 0.276] | 0.325 [0.301, 0.351] |
| cos(d1, d3) | 0.225 [0.206, 0.245] | 0.232 [0.212, 0.252] | 0.274 [0.256, 0.294] |
| visit multiplier ||all||^2 / mean_k ||d_k||^2 | 2.558 [2.506, 2.612] | 2.412 [2.359, 2.476] | 3.049 [2.915, 3.189] |
| ||all||^2 / sum_k ||d_k||^2 | 0.853 [0.835, 0.871] | 0.804 [0.786, 0.825] | 1.016 [0.972, 1.063] |

- Fisher r: pass split 0.675, this split 0.838; CIs do not overlap.
- Fisher norm_ratio: pass split 0.841, this split 0.759; CIs do not overlap.
- head Euclidean (substitute) r: pass split 0.740, this split 0.893; CIs do not overlap.
- head Euclidean (substitute) norm_ratio: pass split 0.805, this split 0.735; CIs do not overlap.

| Condition | KL vs base |
|---|---|
| ls67 | 0.01614 |
| ls89 | 0.02878 |
| ls1011 | 0.02489 |

Mean KL all / sum of the three parts: 0.888 (point only; the per-chunk KL of the controls is not stored).

## 3-bit rung

Pass 1 / 2 / 3 / all at the 3-bit level (IQ3_XXS gate_up, down IQ4_NL as in the base). On B2 the mean relative l2 error against the BF16 source is 0.0353 for IQ3_XXS gate_up and 0.1251 for IQ2_XXS gate_up (`results/pilot/t-blocks.tsv`). The comparison lines below set it against the 2-bit pass split.

Residual statistics from `results/pilot/perpass3bit-residual.json` (95% bootstrap CI over chunks):

| Statistic | Fisher | head Euclidean (substitute) | full vocabulary (floor included) |
|---|---|---|---|
| r = ||R|| / ||all|| | 0.748 [0.732, 0.761] | 0.851 [0.837, 0.864] | 0.878 [0.864, 0.893] |
| norm ratio ||all|| / ||sum_k d_k|| | 0.788 [0.780, 0.794] | 0.741 [0.735, 0.749] | 0.732 [0.724, 0.740] |
| ||d1|| / ||d3|| | 1.629 [1.572, 1.683] | 1.476 [1.440, 1.512] | 1.520 [1.484, 1.558] |
| ||d2|| / ||d3|| | 1.261 [1.244, 1.279] | 1.151 [1.139, 1.165] | 1.172 [1.159, 1.185] |
| ||d3|| / ||all|| | 0.467 [0.457, 0.477] | 0.527 [0.520, 0.534] | 0.522 [0.513, 0.530] |
| cos(d1, d2) | 0.255 [0.238, 0.273] | 0.277 [0.264, 0.291] | 0.288 [0.270, 0.305] |
| cos(d2, d3) | 0.269 [0.257, 0.281] | 0.275 [0.264, 0.285] | 0.277 [0.266, 0.288] |
| cos(d1, d3) | 0.125 [0.110, 0.141] | 0.157 [0.148, 0.167] | 0.163 [0.149, 0.176] |
| visit multiplier ||all||^2 / mean_k ||d_k||^2 | 2.620 [2.578, 2.671] | 2.393 [2.359, 2.431] | 2.351 [2.307, 2.393] |
| ||all||^2 / sum_k ||d_k||^2 | 0.873 [0.859, 0.890] | 0.798 [0.786, 0.810] | 0.784 [0.769, 0.798] |

- Fisher r: 2-bit 0.675, 3-bit 0.748; CIs do not overlap.
- Fisher norm_ratio: 2-bit 0.841, 3-bit 0.788; CIs do not overlap.
- head Euclidean (substitute) r: 2-bit 0.740, 3-bit 0.851; CIs do not overlap.
- head Euclidean (substitute) norm_ratio: 2-bit 0.805, 3-bit 0.741; CIs do not overlap.

| Condition | KL vs base |
|---|---|
| 3b-p1 | 0.00467 |
| 3b-p2 | 0.00281 |
| 3b-p3 | 0.00177 |
| 3b-all | 0.00809 |

Mean KL all / sum of the three parts: 0.875 (point only; the per-chunk KL of the controls is not stored).

## What the controls allow

Rule as applied: under the Fisher norm, for both r and the norm ratio, the pass-split and layer-split 95% CIs must not overlap (r 0.675 against 0.838, norm ratio 0.841 against 0.759; pass split first).

The controls license one loop-specific claim at the 2-bit amplitude on B2: splitting the change across passes on shared weights gives a different Fisher r and norm ratio than splitting it across layer pairs, which share no weights, and on these two statistics the pass split is nearer additive than the layer split (smaller r, norm ratio nearer 1). They do not license an H1 classification, any statement about B3 or other amplitudes, or a choice among the additivity comparators. The 3-bit rung (gate_up only, no layer-split control) licenses no loop-specific claim, and its Fisher r and norm ratio differ from the 2-bit pass split beyond both CIs, so the per-pass statistics depend on the level.
