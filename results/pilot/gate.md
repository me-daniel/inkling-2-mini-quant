# Release gate (pilot, P = 3)

Written by `scripts/gate-report.py` from `results/pilot/` and `logs/`. Do not edit by hand. Exploratory pilot (CHANGELOG.md, 21 Nov 2026).

**Verdict: FAIL.** Quality gate FAIL, stability not run. Uploads still need Daniel's go.

## Selection on S (Addendum F)

| candidate | predicted KL | measured KL on S (± SE) | measured / predicted | allocator bytes | file bytes | spliced and verified |
|---|---|---|---|---|---|---|
| sel-0 | 0.03900 | 0.03145 ± 0.00045 | 0.806 | 11,564,368,896 | 11,576,346,656 | yes |
| sel-1 | 0.03989 | 0.02963 ± 0.00039 | 0.743 | 11,587,011,456 | 11,598,989,600 | yes |
| sel-2 | 0.04001 | 0.02931 ± 0.00042 | 0.733 | 11,502,076,800 | 11,514,054,944 | yes |
| sel-3 | 0.04011 | 0.02945 ± 0.00044 | 0.734 | 11,438,375,808 | 11,450,353,952 | yes |

UD-IQ3_XXS on S (not used for selection): 0.02050.

Chosen: sel-2 (from `selection.md`).

## Files

| build | bytes | under the UD-IQ3_XXS limit | expert tensor types |
|---|---|---|---|
| cand | 11,514,054,944 | yes | ffn_down_exps IQ4_NL x24, ffn_gate_up_exps IQ2_S x6, ffn_gate_up_exps IQ2_XXS x6, ffn_gate_up_exps IQ3_XXS x12 |
| stock IQ3_XXS (larger than UD-IQ3_XXS) | 11,747,211,680 | no | ffn_down_exps IQ4_NL x24, ffn_gate_up_exps IQ3_XXS x24 |
| UD-IQ3_XXS | 11,663,310,784 | yes | ffn_down_exps IQ4_NL x4, ffn_down_exps NVFP4 x20, ffn_gate_up_exps IQ2_S x9, ffn_gate_up_exps IQ2_XS x1, ffn_gate_up_exps IQ2_XXS x1, ffn_gate_up_exps IQ3_XXS x13 |

## Quality gate on T1 and T2: FAIL

Paired difference cand minus baseline, 95% bootstrap CI over chunks (10,000 resamples, seed 0). Negative favours cand.

| set | baseline | mean KL cand vs baseline | diff [95% CI] | top-1 cand vs baseline | chunks | chunks where cand is worse |
|---|---|---|---|---|---|---|
| T1 | UD-IQ3_XXS | 0.02686 vs 0.01939 | +0.00747 [+0.00654, +0.00845] | 93.14% vs 94.42% | 16 | 16 |
| T2 | UD-IQ3_XXS | 0.01942 vs 0.01361 | +0.00581 [+0.00513, +0.00673] | 95.29% vs 95.95% | 5 | 5 |
| T1 | stock IQ3_XXS (larger than UD-IQ3_XXS) | 0.02686 vs 0.05421 | -0.02735 [-0.02938, -0.02537] | 93.14% vs 90.32% | 16 | 0 |
| T2 | stock IQ3_XXS (larger than UD-IQ3_XXS) | 0.01942 vs 0.04622 | -0.02680 [-0.02912, -0.02407] | 95.29% vs 92.67% | 5 | 0 |

Raw output of `analysis/compare.py`:

```
T1 cand vs uniform-IQ3_XXS: KL 0.02686 vs 0.05421, diff -0.02735 [-0.02938, -0.02537], top-1 93.14% vs 90.32% -> better
T1 cand vs UD-IQ3_XXS: KL 0.02686 vs 0.01939, diff +0.00747 [+0.00654, +0.00845], top-1 93.14% vs 94.42% -> not better
T2 cand vs uniform-IQ3_XXS: KL 0.01942 vs 0.04622, diff -0.02680 [-0.02912, -0.02407], top-1 95.29% vs 92.67% -> better
T2 cand vs UD-IQ3_XXS: KL 0.01942 vs 0.01361, diff +0.00581 [+0.00513, +0.00673], top-1 95.29% vs 95.95% -> not better
quality gate: FAIL
```

## Teacher check (Addendum F)

Not run: the quality gate did not pass, and the cand minus UD-IQ3_XXS gaps (T1 +0.00747, T2 +0.00581 KL) are not margins below 0.005 in cand's favour.

## Stability (cand against the Q8_0 reference): not run

Not run (phase 4 stops after a quality FAIL, Addendum F).

## Speed and Metal coverage (Addendum C)

`llama-bench` -ngl 99 -p 512 -n 128 -r 3, default loop.count = 3. Phase 4 runs it only after a quality PASS. The Metal count is read from each file's tensor types.

| build | pp512 t/s | tg128 t/s | expert tensors Metal cannot multiply |
|---|---|---|---|
| cand | not run | not run | 0 |
| stock IQ3_XXS (larger than UD-IQ3_XXS) | not run | not run | 0 |
| UD-IQ3_XXS | not run | not run | 20 |
