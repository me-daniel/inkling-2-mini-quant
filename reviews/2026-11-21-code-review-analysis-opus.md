# Code review: the delegated pilot analysis code

- Date: 2026-11-21. Reviewer: Opus 5.5 (workflow subagent). Independent of `reviews/2026-11-21-audit-perpass-opus.md`, which audited the per-pass report. This review covers the code.
- Files: `analysis/per_pass.py`, `analysis/noise.py`, `analysis/maptable.py`, `analysis/allocate.py`, `analysis/compare.py`, `analysis/perpass_report.py`, and `tests/`. None of them was edited.
- Rules checked against: `CHANGELOG.md` run 4 and Addenda A, B, E, F, G.
- Evidence: `reviews/2026-11-21-code-review-analysis-opus/`. Six `check_*.py` scripts, all CPU-only. Their output is in `numbers.txt`, and every number below is taken from it. They read the small result files, the logs, GGUF headers, 12 expert tensors, and the first 128 rows of chunk 0 of the five 2-bit pass-split logits files. They write only to temp dirs. Rerun them with `.venv/bin/python -I reviews/2026-11-21-code-review-analysis-opus/check_<name>.py`.

## Guide for Daniel (one page)

**Verdict.** The math is right. Every stored number I could recompute matches. Nothing I found changes the gate FAIL, the Mac-track comparisons or the per-pass tables. The findings are about failures that would give wrong numbers with no error, plus tests that cannot catch the main bugs.

**What each file does.**
- `per_pass.py` reads five logits files (base, pass 1, 2, 3, all). For each scored token it forms d_k = logp_k - logp_base, the sum of the d_k, the residual R = d_all - sum_k d_k, and the pairwise sums. It accumulates squared norms per chunk under three norms: Fisher (weighted by the base distribution, centred by the p-mean), head Euclidean (entries within 10 nats of the base's top, centred) and full vocabulary (centred). From pooled sums it gives r, the norm ratio, the pass ratios, the share of the last pass, the visit multiplier and the cosines (by polarisation). The CIs come from a percentile bootstrap over whole chunks.
- `noise.py` gives a paired chunk-bootstrap ratio of mean KLs, for tied/independent and for all/sum.
- `maptable.py` parses every `logs/ppl-*.log` into `results/pilot/kl.tsv`. It rebuilds per-chunk KL from the running means: chunk k = k m_k - (k - 1) m_(k-1).
- `allocate.py` predicts KL additively from the map (Addendum E), prices bytes exactly from the Q8_0 reference's tensor list, enumerates every combination under the budget, and picks 4 that differ in at least 2 of 11 choices.
- `compare.py` takes the paired difference of per-chunk KL (candidate minus baseline) with a percentile bootstrap. It reports "better" only when the upper bound is below 0 and the candidate's top-1 is not lower.
- `perpass_report.py` writes `results/pilot/perpass.md` from the stored result files and applies the Addendum F rule (disjoint CIs for both r and the norm ratio).

**The lines to read (about ten).**
1. `per_pass.py:48-49`: the Fisher norm. Centring under p, then a p-weighted sum of squares.
2. `per_pass.py:51-53`: the head and full forms. The head is unweighted, so flat tokens weigh more.
3. `per_pass.py:70-72`: p comes from the base and is renormalised with the floored entries kept. The head mask comes from the base only.
4. `per_pass.py:73-76`: the deltas, R and the pairwise sums. All five files go through the same decode.
5. `per_pass.py:86-94`: r, norm ratio, visit multiplier and the polarisation cosines.
6. `per_pass.py:101-104`: one index draw, whole chunks, the same indices for every term and norm.
7. `maptable.py:13` and `:22-25`: the chunk-line regex and the telescoping. Finding 1 is here.
8. `allocate.py:28`, `:63-69` and `:82`: the budget constant, the cost model and the byte test.
9. `compare.py:25-28` and `:40`: the paired bootstrap and the win rule.
10. `perpass_report.py:133-137` and `:178-179`: the Addendum F test and what it licenses.

**What the tests do and do not cover.** `pytest` gives 20 passed.
- Covered: the logits format round trip and KL on unclipped logits (`test_kldfile`). In `per_pass`: additive input gives r near 0, last-pass-only input gives share near 1, compounding input gives a large r, half the Fisher norm tracks KL, and the full form matches a direct formula with a floored quarter. In `noise`: the ratio identities, and that the bootstrap is paired. In the report: the prediction parser on the real registration, a full report on synthetic files, the forbidden words, and every verdict branch.
- Not covered: `maptable.py`, `allocate.py` and `compare.py` have no tests, and the gate verdict runs through all three (finding 2). In `per_pass`, no test notices if the Fisher centring or the head centring is removed, or if `HEAD_NATS` changes (finding 3). No test checks a value of the head norm, the cosines, the visit multiplier, the norm ratio or a CI. The tests put one block per chunk, so a block that crosses a chunk boundary is never exercised. My check with several blocks per chunk passes.

## Findings

### 1. Major: `maptable.py` silently drops chunk lines it cannot parse and gives the next chunk a wrong value

The regex at line 13 accepts only `[\d.]+` in the numeric fields. A line with `nan`, `inf` or a negative running KL does not match and is skipped without a warning. The telescoping at lines 22-25 then pairs the next line's k with the last matched mean: it computes k m_k - (k - 1) m_j for some j < k - 1. That is neither the next chunk nor the sum of the two. Nothing compares the kept lines with the log's "computing over N chunks" line.

It has fired once. `ppl-floor-q8-S-P3.log` declares 16 chunks and keeps 14. Chunks 1 and 2 print `nan` for the SE of ln(PPL ratio), so the stored row starts at chunk 3. The values are all near zero there, so nothing downstream changed. The other 53 logs keep every line, and mean(per_chunk) matches Mean KLD within 5.0e-06.

Synthetic log with 6 chunks, true per-chunk KL [0.02, 0.03, 0.025, 0.015, 0.04, 0.01]. Line 4 holds a `nan` SE, or a running KL printed as `-0.00000`. In both cases the parse gives [0.02, 0.03, 0.025, 0.03, 0.01]. That is 5 values, and the fourth is 0.03 where chunk 5 is 0.040 (0.055 for chunks 4 and 5 together). The mean drops from 0.02333 to 0.02300 with no error.

Why it matters: every gate and Mac-track CI comes from these per-chunk values (`compare.py`). If two runs lost different lines, compare would raise on the length mismatch. If they lost the same lines, the pairs would quietly be wrong.

Fix: match the chunk lines loosely (`\S+` per field) and parse `nan` as a value. Require the kept indices to be exactly 1..N, with N read from "computing over N chunks". Raise if they are not.

### 2. Major: the gate path has no tests

`tests/` has no test for `maptable.py`, `allocate.py` or `compare.py`. CLAUDE.md delegates `maptable.py` and `allocate.py` with the condition "Claude writes them with tests". `compare.py` produces the quality-gate and Mac-track verdicts. I checked their current outputs by hand:
- `compare.py`: 12 of 12 stored comparison lines in `gate-quality.txt`, `mac-quality.txt` and `mac-context-IQ2_M.txt` reproduce to the printed digit (`check_gate.py`).
- `allocate.py`: rerun on a copy of `kl.tsv`, it reproduces `allocation-top.json` exactly. Hand-computed Addendum E predictions equal the stored ones, for example sel-2 at 0.040006 against a measured 0.029309 on S. The priced bytes of sel-2 equal the tensor bytes of `cand.gguf`, 11,502,076,800 (`check_allocate.py`).

Nothing guards these properties when the code or the logs change. Fix: three small tests. maptable on a synthetic log, including the cases in finding 1. compare on fixed arrays with a known sign and a top-1 tie. allocate's cost and byte arithmetic on a two-block toy table.

### 3. Major: the `per_pass` tests cannot see the centring, the head norm or `HEAD_NATS`

I mutated `_sq_norms` and reran the 6 tests in `tests/test_per_pass.py` (`check_per_pass.py`):
- Fisher norm without centring: no test fails.
- Head norm without centring: no test fails.
- `HEAD_NATS` changed from 10 to 3: no test fails.

`test_shift_invariance` adds +3.0 and -7.0 to the logits. The writer stores log-softmax, so the shift is gone before `per_pass` reads the file: after the writer the two files differ by at most 1.1e-04 nats, less than the median code step of 2.44e-04. The test checks the writer, not the centring. Missing Fisher centring is also hard to see in principle (inference): for exact log-probs the p-mean of d is -KL, so centring changes the Fisher norm only at second order. The head norm is different. Its centring matters, and that norm is the one the registration is scored on (Addendum G.2).

Fix: one test that compares `head_euclid` and `fisher` per term with an explicit per-token formula on files with uneven head sizes. A second that checks `HEAD_NATS` reaches the mask. A third with known cosines (for example d2 = d1, so cos_12 = 1).

### 4. Minor: on T2 the percentile bootstrap rests on 5 chunks and is narrower than a t-interval

T2 is 10,428 tokens (release tokenizer). At context 2,048 that is 5 chunks, and 188 tokens past the last chunk are dropped. llama-perplexity scores only the second half of each chunk, so 5,115 positions are scored. Addendum F chose context 2,048 "so the bootstrap has enough chunks". With 5 chunks the percentile CI is 0.60 to 0.64 of the width of a Student-t interval on the same differences. On T1 (16 chunks) it is 0.88 to 0.89. No verdict changes. Every T2 comparison has the candidate on the same side on all 5 chunks (worse on 5/5 against UD-IQ3_XXS, better on 5/5 against every other file), and every t-interval keeps the bootstrap's sign. Example: cand minus UD-IQ3_XXS on T2 is +0.00581, bootstrap [+0.00513, +0.00673], t [+0.00451, +0.00711].

The 5-decimal running means in the logs do not matter. Re-rounding them at random 200 times gives CI widths with a standard deviation of at most 2.0e-05.

Fix: print the t-interval and the sign count next to the bootstrap for T2, and say the scored positions are 5,115.

### 5. Minor: `compare.py`'s last line reads like the upload gate

`mac-quality.txt` ends with "quality gate: PASS". The upload gate FAILed (`gate-quality.txt`). The same function also prints the Addendum G decision, and that decision differs from the upload gate in three ways:
- `compare.py:40` also demands top-1 not worse. The Addendum G rule asks only for the KL CI below 0. The extra condition is stricter, so it can only turn a PASS into a FAIL.
- It counts every baseline it is given. Addendum G counts only comparators confirmed fully Metal-supported. Counting more is again stricter.
- A missing run raises `KeyError` after some lines are already printed. In `macgate.sh:53` the output goes through `tee` under `sh` without pipefail, so the chain carries on and `mac-quality.txt` is left with no verdict line. That part is a script issue, outside these files.

Fix: let the caller name the rule (`--rule upload|mac`), print "upload gate" or "Mac decision rule (Addendum G)", and check that every run exists before printing anything.

### 6. Minor: `compare.paired` broadcasts arrays of unequal length

`paired(16 chunks of 0.02, 1 chunk of 0.03)` returns (-0.01, -0.01, -0.01): a zero-width CI, with no error. Other unequal lengths raise. Nothing checks that both logs used the same n_ctx or the same declared chunk count. Fix: assert equal shapes, and carry n_ctx and the declared chunk count into `kl.tsv` (finding 1).

### 7. Minor: the budget constant assumes 384 bytes too little overhead; the docstring miscounts the options

`BUDGET = 11_663_310_784 - 11_977_760`. The real overhead of `cand.gguf` is 11,978,144 bytes, and of `uniform-IQ3_XXS.gguf` also 11,978,144. The difference of 384 bytes is alignment padding, which depends on the tensor types. An allocation priced within 384 bytes of `BUDGET` would give a file over the UD-IQ3_XXS limit. Today's smallest margin in the top 8 is 2 MB and sel-2 has 149 MB, so nothing changed. Fix: subtract the largest overhead seen plus a margin of a few kB, and keep the final file-size check (`gate.md` has one).

The module docstring says "4 gate_up and 2 down types". The code has 3 (`IQ3_XXS`, `IQ2_S`, `IQ2_XXS`). Addendum D widened the candidate types to IQ2_XS, IQ3_S and IQ4_XS, but no map level prices them, so the allocator never considers them. Say so in the docstring.

### 8. Minor: missing map runs shrink the search space silently

The allocator skips an option whose map run is missing (`allocate.py:67`). It also drops Q6_K non-expert tensors if `map-NE-Q6K-S-P3` is missing. It exits only when a block has no options at all. With `map-B3-gu2s-S-P3` and `map-NE-Q6K-S-P3` removed from a copy of `kl.tsv`, it runs without error and returns 4 picks with NE types ['Q8_0'] only. All map runs exist today. Fix: list the expected runs and exit if any is missing.

### 9. Minor: `kl.tsv`'s `seconds` column is always `nan`

The docstring says the wall time is recorded "for the pass-count check". No log has a "total time" line, so the column is `nan` in all 53 rows. The pass-count check is done elsewhere (`phase2_report.py`, `p4-timing.sh`). Fix: drop the column or read the timestamps. The log list now has one KL log not yet collected, `ppl-upstream-cand-T1-P3.log`, newer than `kl.tsv`, so its 54 logs and 53 rows are not a mismatch.

### 10. Minor: the noise ratio uses draw 1 only, and duplicate rows resolve differently

- `noise-ratio.txt` and registration item 6.4 use noise-tied (draw 1) only. Draws 2 and 3 now exist. Against noise-indep they give 1.196 [1.169, 1.225], 1.222 [1.187, 1.256] and 1.223 [1.182, 1.262]. All three sit inside the registered 1.05-1.35. The report prints the three tied KLs but only the first ratio. Fix: print all three ratios.
- On duplicate (cond, against) rows, `noise.chunks` takes the first and `perpass_report.kl_chunks` and `kl_table` take the last (checked on a two-row file). A rerun appended to `perpass-kl.tsv` would then give `noise-ratio.txt` and the report tables from different rows. Today there are no duplicates, and the stored `noise-ratio.txt` equals a fresh recomputation. Fix: raise on duplicates in one shared reader.

### 11. Minor: stale and hard-coded text

- The `per_pass.py` docstring still says the three deviations "are not recorded in CHANGELOG.md". Addendum G section 2 now records them.
- `perpass_report.py` types "16 x 4,096 tokens", "the 16 chunks", "10 nats" and "16 nats" into the text. Read them from the data, `per_pass.HEAD_NATS` and the kldfile header, so the text cannot drift from the code.
- The verdict says the 3-bit rung shows that "the per-pass statistics depend on the level". The rung changes only gate_up (down tensors identical to the base in 6 of 6 layers). The 2-bit level changes gate_up and down (`check_real_logits.py`). The rung therefore differs in amplitude and in which projections change. Say "level (amplitude and projections changed)".

### 12. Nit: one wrong sentence in a rare branch of `control_section`

When the two CIs are disjoint but a point lies inside the other's CI, `control_section` prints "CIs do not overlap. Each point lies outside the other's CI." The second sentence is false in that case. It needs a percentile CI that excludes its own point. That is rare but possible (`check_report.py`). The verdict function handles the case correctly.

## Checked, no finding

- **Fisher algebra and statistics.** On synthetic files with a floored fifth of the vocabulary, correlated parts and several blocks per chunk, all 9 statistics match an independent per-chunk Fisher Gram matrix. The largest relative difference is 3.5e-15, and 2.5e-14 for r per chunk. The cosines from pairwise sums are valid, because each norm is a per-token quadratic form.
- **Bootstrap.** The r CI from `summarize` equals an independent loop with the same index draw to 12 digits. The same indices serve every term and norm. `noise.paired_ratio` and `compare.paired` resample whole chunks, paired across conditions. All seeds are fixed at 0.
- **Token alignment.** `accumulate` checks n_ctx, n_vocab and the full token arrays of all five files, and the header check pins the file sizes, so `zip` cannot run out early.
- **Float precision.** Decoding in float32 differs from float64 by at most 1.07e-06 nats, against a median writer step of 2.44e-04. All sums are float64. On real logits (2-bit pass split, chunk 0, rows 0-127) the writer's rounding noise is 1.1e-06 of ||R||^2 under Fisher, 3.2e-07 under head and 1.5e-07 under full. r is unchanged at 5 decimals after removing it, except full (0.67010 against 0.67009). The 3-bit rung's changes are smaller, so its noise share is larger, but still small (inference).
- **Division by near zero.** Each denominator is a pooled squared norm of a real perturbation. A no-op condition would give `nan`, not a silent number.
- **Telescoping.** It is valid because every llama-perplexity KL chunk scores the same number of positions. On 53 of 54 logs the per-chunk mean matches Mean KLD within 5.0e-06.
- **Allocator.** The cost model matches Addendum E: each change comes from the map level that differs from 3bit in that projection only. The bytes match the file exactly. The 2-of-11 distinctness rule is implemented as written.
- **Gate rule.** `win = hi < 0 and top-1 not worse` is the pilot's upload gate as registered. The FAIL against UD-IQ3_XXS holds under the t-interval too (T1 [+0.00639, +0.00855], T2 [+0.00451, +0.00711]).
- **Registered predictions.** The parser reads the four ranges in order from the registration. The report scores them on the head substitute, as Addendum G.2 says.
- **3-bit rung label.** "gate_up only" is true. See finding 11.

## Suggestion, not a finding

The pass-split and layer-split analyses share the base, the `all` file, the chunks and the bootstrap indices (seed 0, 16 chunks). A paired bootstrap of the difference (r_pass - r_layer, nr_pass - nr_layer) costs nothing with the current code. It would also report a measure of the effect, where "disjoint CIs" gives only yes or no. The registered rule stays the decision. The paired CI would be reported next to it, labelled post hoc.

## Outside these files

During this review a `scripts/kldfile.py` process building the per-pass KL table was resident at about 18 GB. That figure is a `ps` reading, not from a file. `kl_per_chunk` loads whole chunks in float64. Next to a GPU job, that can approach the one-heavy-run limit (inference).
