# Audit: per-pass result (pilot run 4, Addenda A, B, F)

- Date: 2026-11-21, before anything about the per-pass result goes public
- Auditor: Opus 5.5, effort xhigh (in-session subagent), as pre-registered
- Object: `results/pilot/perpass.md` and the prior step's summary of it
- Read: `analysis/per_pass.py`, `analysis/noise.py`, `analysis/perpass_report.py`, `scripts/kldfile.py`, `scripts/perpass.sh`, `scripts/perpass-kl.sh`, `scripts/splice.py`, `scripts/noise.py`, patches 0003 and 0006, the patched `src/models/inkling2-loop.cpp` and `inkling2.cpp`, `tools/perplexity/perplexity.cpp` (writer and KL), `tests/`, `CHANGELOG.md` (pilot, Addenda A-F), the registered prediction, splice manifests and run logs
- Evidence: `reviews/2026-11-21-audit-perpass-opus/`. `audit_check.py` and `audit_kl.py` reread chunks 0 and 1 of the raw logits on the CPU (no GPU job). `audit_numbers.py` prints every number below from the result files, the logs and those two outputs, into `numbers.txt`.

## Verdict

The numbers in `perpass.md` are computed correctly. I found no bug that changes them. An independent pass over the raw logits of chunks 0 and 1 reproduces the stored per-chunk Fisher r, head r and KL to every printed digit. The problems are in what the report and the summary say about the numbers. One finding is blocking: no loop-specific or "counts once per visit" wording before the layer-split comparator has run. Four are major: missing scope caveats, a substitute statistic labelled as the registered one, a norm-dependent pass ordering reported in its weaker form, and a noise ratio that rests on one draw per pass.

## Findings

### 1. Blocking: interpretive claims go beyond the design

The prior summary says "In KL the cost scales roughly once per visit", "Pass 3 does not dominate" and "The effects are far from additive as vectors". The first is H1 classification language. The CHANGELOG card wording allows the per-pass result only as "on these documents, at P = 3, changing B2's experts on pass k produced this measured logit change". Addendum F adds that a loop-specific claim needs the pass-split values to differ from the layer-split values beyond both CIs. The layer-split comparator and the 3-bit rung have not run.

The "once per visit" reading also depends on the comparator. In KL, all / sum of single passes is 1.135 [1.112, 1.162], and the CI excludes 1. In squared Fisher norm, ||all||^2 / sum_k ||d_k||^2 is 1.088. As vectors, ||all|| / ||sum_k d_k|| is 0.841. One comparator reads super-additive, another sub-additive. Choosing among them is the H1 analysis that Daniel writes. A pilot summary should not settle it.

Fix: no wording about the per-pass result goes public beyond the card sentence and the measured tables until `scripts/perpass2.sh` has run and the control sections are filled.

### 2. Major: perpass.md lacks caveats the addenda require

- Addendum F says the 2-bit results "are reported as amplitude-specific". The report names the types but never says this. It also omits the relative weight errors (0.5292 down, 0.1288 gate_up, from `noise-t.tsv`) next to the residual table.
- B2 is layers 6-11, half of the loop core (6-17). Nothing was measured on B3.
- The two control sections say only "Placeholder". They should already state Addendum F's rule, so a reader of the current file knows no loop-specific reading is licensed yet.
- The noise ratio rests on one draw per pass (finding 5). The report does not say so.

Fix: a short scope paragraph after the intro, written by `perpass_report.py`, with these four points.

### 3. Major: the registered statistic is replaced, and the report says it matches

The prediction registers "pooled Euclidean RMS" with "the same vocabulary centring everywhere", which is the full vocabulary. The report scores on the entries within 10 nats of the base's top and calls this "matches the registered prediction" and "(pooled RMS, as registered)".

The full-vocabulary form cannot be computed faithfully from these files. The writer floors every value more than 16 nats below the top. On chunks 0 and 1, every scored row has floored entries, and they hold 0.0120 and 0.0131 of the renormalised base mass. A substitute is justified. It is still a substitute.

The choice was fixed before data. `HEAD_NATS = 10` and the Fisher primary norm are in commit 9dd17b8 (05:09). The first per-pass logits were written at 06:36. Neither choice is in `CHANGELOG.md`. Nor is the switch from "a per-token residual ratio" (CHANGELOG, run 4) to a pooled ratio of sums.

Sensitivity on chunks 0 and 1 (`audit_check.json`):

| Norm | r (c0, c1) | norm ratio (c0, c1) | p1/p3 (c0, c1) | p2/p3 (c0, c1) |
|---|---|---|---|---|
| Fisher | 0.622, 0.668 | 0.856, 0.899 | 1.019, 0.907 | 1.057, 0.990 |
| full vocabulary (registered form, floor included) | 0.672, 0.709 | 0.892, 0.873 | 0.915, 0.844 | 0.985, 0.929 |
| head 5 nats | 0.664, 0.669 | 0.836, 0.853 | 0.980, 0.870 | 1.027, 0.923 |
| head 10 nats (reported) | 0.697, 0.719 | 0.820, 0.823 | 0.907, 0.837 | 0.979, 0.875 |
| head 14 nats | 0.722, 0.745 | 0.812, 0.813 | 0.887, 0.832 | 0.954, 0.866 |

Under every form the 6.1 bounds and 6.2 fail. 6.3 stays outside, but the literal full-vocabulary form sits nearer the 0.90 bound (0.892 and 0.873).

Fix: a dated CHANGELOG entry that records the Fisher primary norm, the 10-nat head and the pooled ratio, with the commit that fixed them before data. Relabel the column "head-restricted substitute; the registered full-vocabulary form is not computable (16-nat floor)". Add a full-vocabulary sensitivity row.

### 4. Major: the 6.1 ordering depends on the norm, and the report shows the weaker reading

- Head Euclidean: ||d1||/||d3|| 0.976 [0.942, 1.008], ||d2||/||d3|| 0.999 [0.971, 1.026]. The report says "point estimates hold".
- Fisher: 1.103 [1.059, 1.148] and 1.122 [1.082, 1.166]. Both CIs lie wholly above 1.
- KL: p1/p3 1.185 [1.096, 1.278], p2/p3 1.247 [1.169, 1.335] (paired bootstrap over chunks).

Under the norm that tracks KL, passes 1 and 2 move the output more than pass 3, and both CIs exclude 1. The report's Fisher line says only "both ratio CIs do not lie below 1, so pass 3 is not shown to dominate". That understates the result.

The norms disagree because the head is unweighted. On chunks 0 and 1 a token's head holds 1 to 6,796 entries (median 154 and 136). Tokens with flat distributions dominate the head sum. The prediction warned about this in its point 7. The ordering also moves between chunks: Fisher p1/p3 is 1.019 on chunk 0 and 0.907 on chunk 1.

Fix: use the head form only to score the registration. State the ordering under Fisher and KL, with the direction of the CIs written out.

### 5. Major: the tied/independent ratio rests on one draw per pass

noise-tied uses noise-1 on all three passes. noise-indep uses noise-1, noise-2 and noise-3. Tying is therefore confounded with which draws sit on passes 2 and 3. The CI [1.169, 1.225] covers chunk sampling only. The registration itself says "Generalizing its sign to other draws requires seed replication", and its point 7 says one draw confounds tying with draw variability.

The matching itself is fair per visit. Per expert and projection, achieved / target relative error is +0.07% on average (range -0.15% to +0.32%) for gate_up and +0.21% (range -0.12% to +0.54%) for down, over 768 experts x 3 seeds. Each visit gets the same relative error in both conditions, and pass 1 shares its draw, so the contrast is paired.

Addendum A also predicts "independent ~ sum of single-pass effects". No Gaussian single-pass condition was run, so this prediction is untested. The 2-bit single passes cannot stand in for it (prediction, point 7). The report leaves the prediction out instead of saying it is untested.

The summary's "Tied Gaussian noise costs about the same KL as the real all-pass 2-bit change" holds for this draw: tied / all = 1.025 [0.997, 1.056].

Fix: before 6.4 goes public, run noise-tied with seeds 2 and 3 (two splices and two runs of about 10 minutes each, when the GPU is free). The spread of the three tied KLs bounds the draw effect. State in the report that the Addendum A additivity prediction is not tested.

### 6. Minor: the parity gate has no scripted record

No repo script writes `perpass-parity.txt`. The cmp ran by hand, and the twin logits were then deleted. This check is the CHANGELOG stop rule for the whole test.

The logs back it up. The per-chunk PPL of pp-noop equals pp-base in all 16 chunks, and pp-allslots equals pp-all. As a control, pp-p1 differs from pp-base. The noop parity also tests the slot index end to end: noop puts base layer l into slot 24 + (p - 2)*12 + (l - 6), and a wrong mapping would change the logits.

Fix: run cmp and sha256 on both twins inside `perpass.sh`, and log them before any deletion. `perpass2.sh` deletes logits too.

### 7. Minor: the KL estimator differs from llama-perplexity's, and the floor gets weight

`kldfile.py` renormalises both sides over all entries, floors included. llama-perplexity keeps base entries with log p > -16, does not renormalise, and uses the other model's exact logits. On Q8_0 to IQ4_NL, llama-perplexity gives 0.016192 (`kl.tsv`) and kldfile gives 0.016060 (`perpass-kl.tsv`), 0.8% lower. A proxy that copies llama's base side raises the chunk-0 KLs by 0.17% to 0.63%. all/sum moves from 1.1962 to 1.2004 and tied/indep from 1.3116 to 1.3120 on chunk 0. The ratios are robust. The absolute KLs should not share a table with llama-perplexity's map numbers without a note.

`per_pass.py` says the Fisher norm "gives no weight to the clipped tail". Floored entries hold 1.2% to 1.3% of the base mass. Their effect on r is small. Fisher r over entries clean in all five files is 0.6215 against 0.6220 (chunk 0) and 0.6665 against 0.6683 (chunk 1). Entries floored in a perturbed file but not in the base carry at most 0.00069 of a Fisher norm.

Fix: correct the docstring. A clean-entry Fisher row in the report is optional.

### 8. Minor: build provenance of the per-pass runs

`pilot-chain.sh` rebuilt at 06:28. `build-llama.sh` then printed "loop tests failed", because its check still expected 6 OK lines and patch 0006 makes 12 (fixed in a2d539d at 08:01). It exited before writing `results/env.md`. `pilot-chain.sh` carried on, because `sh build-llama.sh 2>&1 | grep ...` returns grep's status. All 12 checks had printed OK, so the binary was built and passed its tests.

`env.md` was rewritten at 08:00:57. That rebuild relinked `libllama.dylib` (08:00:47) and `llama-perplexity` (08:00:50) while the noise-indep run (07:54:10 to 08:02:12) was using them. The run finished, and the patch files have not changed since 04:04, so the code is the same HEAD 9f5b9d6. The risk is low.

The `ppl-pp-*.log` files do not record the build commit. CLAUDE.md requires the commit with every run.

Fix: check `build-llama.sh`'s exit status in the chain scripts before filtering its output. Never rebuild while a llama-* job runs. Print the patched HEAD at the top of each run log.

### 9. Minor: diagnostics the registration asked for are missing

CHANGELOG run 4 names "a per-token residual ratio". The prediction asks for the median token ratio (point 2), the share of residual energy in the top 1% of tokens (point 4) and a leave-one-cluster-out range (point 5). The report has none of them.

On chunks 0 and 1, the median per-token Fisher r is 0.617 and 0.655 (pooled 0.622 and 0.668). The top 1% of tokens hold 0.130 and 0.226 of the residual energy. So r is not an artefact of a few tokens. Leaving one chunk out, all/sum stays within 1.127-1.140 and tied/indep within 1.189-1.201.

Fix: let `per_pass.py` keep per-token values for the median and the top-1% share, and report both with the leave-one-chunk-out range.

### 10. Minor: two ratios are printed without intervals

The report gives all / sum of single passes = 1.135 and all / mean single pass = 3.406 with no CI. With the paired bootstrap over chunks (seed 0) they are 1.135 [1.112, 1.162] and 3.406 [3.335, 3.485].

### 11. Minor: where the non-additivity sits

At the output the quadratic approximation holds. Half the Fisher norm over KL is 0.964 to 1.032 for single passes, and 0.958 and 0.939 for all passes (chunks 0 and 1). The residual is therefore not a higher-order KL effect at the output. It arises inside the network. The summary's "outside the local Taylor regime" applies to weight space and propagation, not to the output.

Most of r is angle, not length. From the pooled Fisher values, cos(all, sum_k d_k) is 0.823. With the measured norm ratio and no angle, r would be 0.189. This decomposition explains how r = 0.675 coexists with positive pairwise cosines, and the report could state it.

### 12. Minor: bootstrap units

S is 16 contiguous WikiText-2 test chunks. Adjacent chunks can share an article, and the prediction asked to group chunks by document when possible. A percentile bootstrap over 16 clusters tends to be narrow. Verdicts whose CIs sit far from the registered ranges do not hinge on this. The head-Euclidean ordering CIs, which straddle 1, do.

### 13. Minor: cosmetics

The base row prints "-0.00000". This is float rounding in `kldfile.py`; print 0 or the absolute value. The intro says "16 x 4,096 tokens", but 32,752 positions are scored (2,047 per chunk). Say so next to "pooled over all scored tokens".

### 14. Minor: test gaps

The synthetic logits in `tests/test_per_pass.py` never reach the 16-nat floor (normal, sd 1.5, 64 entries), so the floor path is untested there. `kldfile`'s KL is tested only on unclipped logits. No test covers the cosines, the visit multiplier or the head norm. The `test-llama-archs` slot fixture has a one-layer core, so the (l - 6) term of the slot formula is not exercised there. The noop parity covers it on the real model.

## Checked, no finding

- **Slot index.** The graph uses il = n_phys + (pass - 1)*(end - start) + (ip - start) with 0-based pass. The base has 24 layers, loop.start 6 and loop.end 18. For 1-based p this is 24 + (p - 2)*12 + (l - 6), and `layer_phys` maps it back to l. `slots()` in `perpass.sh` uses the same formula.
- **p1 perturbs pass 1 only.** Its splice manifest takes 12 tensors from `pool-2bit.gguf` (6 layers x 2 projections) and adds 24 base tensors as slots 24-29 and 36-41. `share_slots` copies the physical layer, then restores both projections from the slot. `build_layer` reads `model.layers[il]`. Blocks 6-11 of the base, the 2-bit pool and noise-1 hold no `.scale` tensors, so no physical scale leaks into a slot. p2 and p3 add slots for one pass, and the other pass shares the physical base tensors.
- **Decoding.** `kldfile.py` matches `log_softmax` in `perplexity.cpp`: log p = min_log_prob + scale * code, and code 0 is the floor at max - 16. The files hold n_vocab 200,058 (unpadded) and 2,047 scored positions per chunk.
- **Fisher algebra.** ||x||_p^2 is Var_p of the log-prob change, and KL(p || q) is about half of it. Centring removes the normaliser. The cosines come from polarisation on pooled sums, which is valid because each token's norm is a quadratic form. Rebuilt from the stored norms and cosines, the norm ratio (0.8413 Fisher, 0.8048 head) and the visit multiplier (3.2637, 2.9857) match the stored values.
- **Bootstrap.** It resamples whole chunks, uses the same indices for every term and both norms, and recomputes each ratio per replicate. `noise.py` pairs tied and independent the same way.
- **Recomputation.** The chunk-0 and chunk-1 Fisher r (0.6220, 0.6683), head r (0.6967, 0.7194) and the KLs of p1, p2, p3 and all match `perpass-residual.json` and `perpass-kl.tsv`.
- **Tests.** `.venv/bin/python -I -m pytest -q tests`: 9 passed.

## Response (2026-11-21, findings 1 to 5)

Scope of this response: analysis code, tests and `results/pilot/perpass.md` (regenerated by `analysis/perpass_report.py`). No GPU job was started. No CHANGELOG entry was written. Every number now in `perpass.md` comes from the result files through the script; this section quotes none that are not already in the audit above. Tests: `.venv/bin/python -I -m pytest -q tests`, 14 passed (9 before).

### 1. Accepted, fixed in the report

The prior summary was session output, not a repo file, so nothing in the repo carried its wording. The report now states measurements and the card sentence only. A "Scope" item states Addendum F's rule and says the report makes no loop-specific claim and no H1 classification while the layer-split control has not run. The old line "pass 3 is not shown to dominate" is gone. The three comparators named in the finding now sit side by side with CIs: KL all / sum, squared Fisher ||all||^2 / sum_k ||d_k||^2 (new row, derived as visit multiplier / 3) and the Fisher vector norm ratio. The report says they disagree and leaves the choice to the H1 analysis. `perpass_report.py` refuses to write the file if it contains a phrase from `FORBIDDEN` ("once per visit", "per visit", "dominat", "compounds", "loop-aware", "pass-aware"), and a test checks the output against that list. Nothing about the per-pass result should go public beyond the card sentence and the tables until `scripts/perpass2.sh` has filled both control sections.

### 2. Accepted, fixed in the report

A "Scope" section after the intro, written by the script, states: amplitude-specific with the 2-bit mean relative errors read from `noise-t.tsv`; B2 only (layers 6-11 of the 6-17 core, B3 not perturbed); one noise draw per pass, CI over chunks only; and the Addendum F rule. Both control placeholders now carry the rule ("no loop-specific reading is licensed" for the layer split, "specific to the 2-bit amplitude" for the 3-bit rung).

### 3. Accepted. Report and code fixed; the CHANGELOG entry needs Daniel

- The head column is now "head Euclidean (substitute)". The text says the registered full-vocabulary form is not computable faithfully because of the 16-nat floor. "As registered" is gone, and a test checks that.
- The report names the three deviations (Fisher primary norm, 10-nat head, pooled ratio of sums instead of run 4's per-token ratio) and says they are not yet in CHANGELOG.md. The `per_pass.py` docstring says the same.
- `per_pass.py` now computes a third norm, `full_euclid`: centred Euclidean over the full vocabulary, floored entries kept as written (new test with a quarter of the vocabulary below the floor). `scripts/perpass2.sh` will produce it for both controls. The stored `perpass-residual.json` predates it. Until it is recomputed, the report prints your chunk 0-1 sensitivity table from `audit_check.json`, labelled as chunks 0 and 1 only, with no CI.
- Needs a CPU rerun, not a GPU run: `.venv/bin/python -I analysis/per_pass.py models/logits/pp-{base,p1,p2,p3,all}.kld > results/pilot/perpass-residual.json`, then the report. It reads five 13 GB logits files, so it should run after the chain finishes. `perpass2.sh` does not delete these five files.
- Not written by me: the dated CHANGELOG entry. CHANGELOG.md records pre-registration changes and is outside this task's file scope, so Daniel writes or approves it. Proposed text: "Addendum G (2026-11-21, recorded after data; choices fixed in commit 9dd17b8 at 05:09, before the first per-pass logits at 06:36). The per-pass residual uses the Fisher norm under the base distribution as primary norm and a pooled ratio of sums over tokens instead of run 4's per-token ratio. The registered prediction is scored on centred Euclidean over the entries within 10 nats of the base's top, a substitute for the registered full-vocabulary form, which the 16-nat floor of the logits files makes uncomputable. A full-vocabulary form with floored entries kept is reported as a sensitivity check."

### 4. Accepted, fixed in the report

The registration scoring keeps the head form and says so ("used only to score the registration"). The old ordering bullets are replaced by a table of p1/p3 and p2/p3 under KL (paired chunk bootstrap, new `noise.paired_ratio`), Fisher and the head substitute. Each cell names the CI direction ("CI above 1", "CI below 1", "CI contains 1"). A note says KL ratios compare squared magnitudes. The KL ratio table adds per-chunk ranges, which show the ordering moving between chunks. Your head-size figures (entries per token, from `audit_check.json`) and the note that flat-distribution tokens weigh more in the head norm are printed under the sensitivity table. The KL ratios reproduce your `audit_numbers.py` values with the same seed. Finding 10 is covered by the same table (all / sum and all / mean now carry CIs).

### 5. Accepted for the report; the fix needs a GPU run

The report now says, next to 6.4 and in the noise section, that each condition is one draw per pass, that the CI covers chunks only, and that Addendum A's "independent ~ sum of single-pass effects" is not tested (no Gaussian single-pass condition; point 7 of the registration). It also prints tied / all with its CI. Not run, by design of this task: noise-tied with draws 2 and 3. That is two splices (`--take models/noise-2.gguf` and `--take models/noise-3.gguf` on B2, as `noise-tied` does with draw 1), two llama-perplexity runs on S at P = 3 when the chain is idle, and two new rows in `scripts/perpass-kl.sh`. The spread of the three tied KLs then bounds the draw effect. 6.4 should not go public before that.

### Findings 6 to 14

Not part of this task. Untouched, except that the `per_pass.py` docstring no longer says the Fisher norm gives no weight to the clipped tail (finding 7).
