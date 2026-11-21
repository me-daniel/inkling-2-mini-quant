# Direction review: is there an edge, and what is the file for?

- Date: 2026-11-21. Reviewer: Fable 5.2 (workflow subagent). Independent of the session that built the candidate.
- Question from Daniel: "Do we have any edge, or should we redo more? What is the point of our model at the moment?"
- Read: `results/pilot/` (gate.md, mac-quality.txt, bench-*.txt, types-*.txt, selection.md, phase2.md, perpass.md, kl.tsv, t-blocks.tsv, q2opt-errors.tsv, depth and memory files), `drafts/MODEL_CARD.md`, `drafts/note.md`, `drafts/release/README.md`, `reviews/2026-11-21-scoop-check.md`, `CHANGELOG.md` with Addenda A to G, the two code reviews, `scripts/hybrid.sh`, and the headers of `cand.gguf` and `UD-IQ3_XXS.gguf`. Not read: `private/`.
- Numbers: this file is written by `reviews/2026-11-21-direction-edge-fable/build.py`. Every measured number is read by that script from the result files, logs, GGUF headers, the llama.cpp tree, or the GitHub API record of PR #20456 saved in that folder (fetched tonight). Rerun with `.venv/bin/python -I reviews/2026-11-21-direction-edge-fable/build.py`.
- Labels: "estimate" and "guess" mark my own hours and odds. They are typed, not computed. "Inference" marks a conclusion that follows from the numbers but was not measured.
- No model run was started for this review.

## Short answer

1. **There is an edge. It is narrow, and it will not last.** Among the GGUFs under 11,663,310,784 bytes that run fully on Metal, the candidate has the lowest held-out KL. The next file is AtomicChat AD-IQ2_S, behind by 0.00811 KL on T1 and 0.00608 on T2. Speed is a tie with every other all-Metal file.
2. **It is not the best file at its size.** UD-IQ3_XXS is closer to the reference by 0.00747 KL on T1 and 0.00581 on T2. The candidate's only lead over it is Mac speed. That lead comes from a missing kernel in llama.cpp, not from anything the file does well.
3. **Do not redo the candidate.** It is frozen (Addendum G, point 1) and T1 and T2 have been seen. Inference: no reallocation inside Metal types will pass the original gate either. Our own build with IQ3_XXS gate_up in all 24 layers, IQ4_NL down and Q8_0 non-experts is 12.66 GB and scores 0.02213 on S. UD-IQ3_XXS scores 0.02050 at 11.66 GB. A file that is at least as fine as the candidate in every tensor, and a gigabyte larger, still does not reach it. Only a change of the down format can move the comparison with UD.
4. **The point of the model right now:** a stopgap for people who run llama.cpp on a Mac and want a file near 11.5 GB. The pilot is worth more than the file. It found the NVFP4 fact on Metal, the `.scale` bug on passes 2 and 3, the Q2_0 encoder defect, and it produced the per-pass method that the scoop check calls the novel part.
5. **Recommendation:** close the release track with one bounded step. Run a two-build diagnostic on S (about 21 minutes of scoring). Then put the hours into the per-pass work. Section 4 has the order.

## 1. The edge

### 1.1 What is measured

KL against the Q8_0 reference at P = 3. Differences are cand minus file with the 95% CI from `gate-quality.txt`, `mac-quality.txt` and `mac-context-IQ2_M.txt`. Speed is the `llama-bench` mean from `bench-*.txt` on the M5 Pro.

| File | Bytes | vs limit | Off-Metal expert tensors | KL T1 | KL T2 | cand minus file, T1 | cand minus file, T2 | Top-1 T1 | pp512 t/s | tg128 t/s |
|---|---|---|---|---|---|---|---|---|---|---|
| cand | 11,514,054,944 | under | 0 | 0.02686 | 0.01942 | n/a | n/a | 93.14% | 224.4 | 19.0 |
| Unsloth UD-IQ3_XXS | 11,663,310,784 | limit | 20 | 0.01939 | 0.01361 | +0.00747 [+0.00654, +0.00845] | +0.00581 [+0.00513, +0.00673] | 94.42% | 68.1 | 10.9 |
| AtomicChat AD-IQ2_S | 11,556,882,400 | under | 0 | 0.03497 | 0.02550 | -0.00811 [-0.00935, -0.00701] | -0.00608 [-0.00739, -0.00503] | 92.23% | 235.8 | 18.3 |
| bartowski IQ2_S | 11,068,886,080 | under | 0 | 0.04623 | 0.03632 | -0.01937 [-0.02139, -0.01748] | -0.01690 [-0.01907, -0.01421] | 91.32% | 235.2 | 18.6 |
| mudler APEX-I-Nano | 11,339,857,152 | under | 0 | 0.05935 | 0.05515 | -0.03249 [-0.03496, -0.03021] | -0.03573 [-0.03956, -0.03229] | 89.78% | 225.3 | 19.4 |
| bartowski IQ2_M | 11,942,949,952 | over | 0 | 0.03396 | 0.02671 | -0.00710 [-0.00827, -0.00614] | -0.00729 [-0.00973, -0.00536] | 92.39% | 232.1 | 20.5 |
| stock IQ3_XXS (ours) | 11,747,211,680 | over | 0 | 0.05421 | 0.04622 | -0.02735 [-0.02938, -0.02537] | -0.02680 [-0.02912, -0.02407] | 90.32% | 234.3 | 19.4 |

- Against UD-IQ3_XXS the candidate loses on 16 of 16 T1 chunks and 5 of 5 T2 chunks. Its KL is 1.39 times UD's on T1 and 1.43 times on T2. Top-1 agreement is 1.28 points lower on T1.
- Against AtomicChat AD-IQ2_S, the best other all-Metal file under the limit, it wins on 16 of 16 T1 chunks and 5 of 5 T2 chunks. Top-1 is 0.91 points higher on T1.
- The candidate sits about halfway between the two. It closes 52% of the T1 gap between AtomicChat AD-IQ2_S and UD-IQ3_XXS, and 51% of the T2 gap.
- It also beats two larger all-Metal files, bartowski IQ2_M (11.94 GB) and our stock IQ3_XXS (11.75 GB).
- Speed among the 6 all-Metal files: pp512 from 224.4 to 235.8 t/s, tg128 from 18.3 to 20.5 t/s. The candidate has no speed edge inside that group. Against UD-IQ3_XXS it is 3.30 times as fast on pp512 and 1.75 times on tg128.
- The file is 149,255,840 bytes under the limit. The selection left that budget unused.

So the measured claim holds: lowest KL among the all-Metal files under the limit that were tested, at equal speed. The gate FAIL holds too.

### 1.2 Who would use it

- A person on Apple Silicon, on llama.cpp or a front end built on it, who wants a file of about this size.
- The maximum resident set size in the 16K-depth bench run was 13.9 GB (`memory-cand.txt`). Inference: a 16 GB Mac is tight, an 18 or 24 GB Mac fits. A Mac with more memory can load a larger file and has no reason to pick this size class. None of this was measured on another machine.
- That person gets the same speed as from AtomicChat AD-IQ2_S and 0.00811 lower KL on T1. Compared with UD-IQ3_XXS they get 3.3 times the prompt speed and pay 0.00747 KL.
- The two margins are about the same size. The lead over AtomicChat AD-IQ2_S is 0.00811 on T1. The loss to UD-IQ3_XXS is 0.00747. If the lead is worth a release, the loss is worth the same weight in the text. The card cannot call one meaningful and the other small.
- We have no evidence that either margin is visible in use. KL is not task quality. The stability battery is at ceiling for both files (30 of 30 GSM8K for the candidate, 30 of 30 for the reference), so it cannot separate them.
- Not served: anyone on CUDA or CPU, where UD-IQ3_XXS runs every tensor and is closer. Anyone on MLX. The scoop check lists an MLX file of 8.87 GB whose own card claims KL 0.0208 against 0.0189 for UD-IQ3_XXS (its numbers, its teacher, not comparable with ours), and an MLX repack that runs the release's native NVFP4 on the Apple GPU at 19.53 GB.

Guess: the audience is small. It is real today.

### 1.3 What happens if Metal NVFP4 lands

- PR #20456 ("metal : add NVFP4 quantization support") is open, created 2026-03-12, last updated 2026-06-23, merge state "dirty", 0 comments, 3 review comments. It is 220 added lines in 5 files. 202 of them are in `ggml-metal.metal`.
- The patch already holds the MoE kernels: `kernel_mul_mm_id_nvfp4_f16`, `kernel_mul_mm_id_nvfp4_f32`, `kernel_mul_mv_id_nvfp4_f32`. Its body reports pp512 44.75 t/s on the CPU against 489 on Metal, and tg128 1.03 against 25.3 (the author's machine and model, not ours). The sibling review `reviews/2026-11-21-metal-nvfp4-feasibility.md` reads the patch in detail and reports that a code owner approved it in March. I did not verify that.
- Our pinned tree no longer has the single kernel file the PR edits (`ggml-metal.metal` present: no; `kernels/` now holds 38 `.metal` files). So the PR needs a port, not a rebase.
- If a kernel lands, UD-IQ3_XXS runs fully on Metal. Inference: its speed then joins the all-Metal group. At that point UD-IQ3_XXS is the better file on every axis we measure except 149,255,840 bytes, and the candidate has no use case left. The card says this already.
- How long: guess. The PR sat for months because nothing popular used NVFP4. Inkling 2 shipped on 20 Nov with NVFP4 experts, and Unsloth's files carry them. Demand is new. I would plan for weeks, not for a year.

The edge lasts until somebody ports about 220 lines. That is the honest size of the moat.

### 1.4 Does "lowest KL among all-Metal files at this size" matter when speed is equal?

Yes, for the narrow audience. With size and speed equal, closeness to the reference is the only thing left to choose on, and the candidate wins it on every chunk. Three limits:

- **The field is two days old and nobody else aimed at this niche.** The comparators are general-purpose uploads. The obvious rival does not exist yet: UD-IQ3_XXS's own layout with its 20 NVFP4 down tensors re-encoded to IQ4_NL. Anyone can build that in minutes. Section 2 shows that our data cannot say whether it would beat the candidate.
- **The size limit is UD-IQ3_XXS's byte count.** It was the right limit for the gate. A user does not have that limit.
- **Two comparators are still open.** IsValorum APEX-I-NanoPlus-V2.1 (10,981,551,648 bytes) and prithivMLmods Q2_K (10,863,738,816 bytes) joined under Amendment 2. Scored so far in `kl.tsv`: 0 of 2. Both are smaller than the candidate. Guess: neither beats it. The rule needs both.

### 1.5 Problems in the current release text

- `logs/hf-upload-v1.log` holds a Hugging Face commit URL (present). I did not open it and I do not know whether that repository is private or public. If it is public, option (a) has already happened. Daniel should confirm that this upload had his go at the time, as `CLAUDE.md` requires.
- Addendum G requires that "the FAIL against UD-IQ3_XXS is stated first, with both margins". `drafts/release/README.md`, as it stood when this file was built: first section "How we made and tested it"; the failure is first mentioned on line 20 of 64, under "How we made and tested it"; T1 margin (0.00747) in the text: no; T2 margin (0.00581) in the text: no. The release copy does not meet the rule. It mentions the failure and leaves both margins to DETAILS.md.
- `drafts/MODEL_CARD.md` gives both margins (T1: yes, T2: yes), under section 3 of 11. Its first section is "In short", which leads with the all-Metal claim. So it does not state the FAIL first either.
- The README says: "We measured the cost of shrinking each of 5 layer groups, including the core that runs three times." The card rules forbid "loop-aware" and "pass-aware". This sentence is not those words, but a reader will take it that way. The map measures the looped blocks with all passes tied.
- The README names T1 for its KL column: no.

## 2. What the data already says about the gap

"Should we redo more" depends on one thing we have not measured: is our gate_up and non-expert allocation any better than Unsloth's, once the down projections are equal?

### 2.1 Where the bytes go (GGUF headers)

| Tensor group | cand | UD-IQ3_XXS | UD minus cand |
|---|---|---|---|
| ffn_gate_up_exps | 5,478,285,312 | 5,711,855,616 | +233,570,304 |
| ffn_down_exps | 4,586,471,424 | 4,586,471,424 | +0 |
| non-expert | 1,437,320,064 | 1,352,983,296 | -84,336,768 |
| per-expert scale tensors | 0 | 20,480 | +20,480 |

- **The down projections cost the same bytes in both files.** Native NVFP4 in layers 0-19 is exact with respect to the release. IQ4_NL at the same bytes is lossy. So wherever NVFP4 can run, IQ4_NL down in those layers is a dominated choice. This matters for the thesis as well: an allocator that does not offer native NVFP4 as a level works on a dominated frontier.
- UD-IQ3_XXS spends 234 MB more than we do on gate_up and 84 MB less on non-experts.
- gate_up types. cand: IQ2_S x6, IQ2_XXS x6, IQ3_XXS x12, 2-bit in layers 0-11. UD: IQ2_S x9, IQ2_XS x1, IQ2_XXS x1, IQ3_XXS x13, 2-bit in layers 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 18.
- non-expert types. cand: Q6_K 1,290 MB, F32 148 MB. UD: Q8_0 476 MB, Q5_K 363 MB, Q4_K 272 MB, F32 148 MB, Q6_K 90 MB, BF16 4 MB.
- UD-IQ3_XXS also carries scale tensors for gate_up (20 of them), so its gate_up codes are stored in a different scale domain from ours. A "same type" layer in the two files is not the same encoding.

### 2.2 How much does IQ4_NL down cost? Two estimates that disagree

Both are inference from S scores. Neither is a measurement of the down format.

- **Cross-file estimate.** Our all-IQ3_XXS build scores 0.02213 on S. UD-IQ3_XXS scores 0.02050, lower by 0.00163, although 11 of its gate_up layers are 2-bit and its non-experts are smaller than Q8_0. Pricing those 11 layers from our block map (per-layer share of each block's gate_up step) gives 0.00590 to 0.00631 KL. Under additivity, native down in layers 0-19 is then worth 0.00753 to 0.00794 on S, plus whatever UD's non-expert types cost. Our own picks came in below the additive sum (measured over additive for the candidate's gate_up and non-expert steps: 0.61). With that factor the range is 0.00522 to 0.00547. The whole cand minus UD gap on S is 0.00881.
- **Weight-error estimate (Addendum A style).** The mean relative weight error of IQ4_NL down in layers 0-19 is 0.0054, against 0.0347 for IQ3_XXS gate_up (`t-blocks.tsv`). A slope fitted per block from the optimized-Q2_0 down level prices IQ4_NL down in layers 0-19 at 0.00172 KL. The same linear model prices the five 3-bit map blocks at 0.02028 in total, where the map measured 0.02821. So it underprices small errors by a factor of 1.4, and its figure for the down format is a floor at best.

Reading: the cross-file estimate says the down format explains most or all of the gap (59% to 90% of it before non-experts). The weight-error estimate says it explains 19%, or 27% after correcting by its own underpricing factor. The first rests on file scores and I trust it more. But it has a hole: it assumes Unsloth's gate_up encoding is no better than ours at the same type. If theirs is better, the down format is worth less and our allocation is worse than it looks.

### 2.3 Whose allocation is better at equal down? The map cannot say

Under additivity the hybrid beats UD-IQ3_XXS on S only if our gate_up and non-expert steps cost less than theirs.

- Our gate_up step costs 0.00826 by the map. Theirs costs 0.00590 to 0.00631. Theirs is cheaper, which fits the 234 MB more they spend there.
- Our non-experts (all Q6_K) cost 0.00353 by the map. Theirs have no map entry.
- Break-even: their non-expert types would have to cost 0.00548 to 0.00589, which is 1.6 to 1.7 times our Q6_K figure. They use Q4_K and Q5_K on part of those tensors, so that is possible. It is not shown.

Guess: the hybrid lands within about 0.002 KL of UD-IQ3_XXS on S, either side. I would not bet on the sign.

### 2.4 The diagnostic that settles it: a 2 x 2 on S

| | down IQ4_NL (runs on Metal) | down native NVFP4 in layers 0-19 |
|---|---|---|
| our gate_up and non-experts | cand: 0.02931 (have) | **hybrid** (`scripts/hybrid.sh`: drafted, not run) |
| UD's gate_up and non-experts | **mirror** (new) | UD-IQ3_XXS: 0.02050 (have) |

- Rows compare the allocations. Columns price the down format. Two new builds fill it.
- **Mirror** is the build nobody has made: UD-IQ3_XXS's gate_up and non-experts, with our IQ4_NL down in layers 0-19. It runs fully on Metal. It is also the rival from section 1.4. If mirror scores below 0.02931 on S, then Unsloth's layout beats ours on our own ground, and "lowest KL among all-Metal files" is true only until someone re-encodes 20 tensors.
- Build note for mirror: use `cand.gguf` as base and take every UD tensor except the down weights of layers 0-19, plus UD's gate_up scale tensors with `--add`. Using UD as base would keep its down scale tensors and apply them to weights that are already scaled.
- Cost from the logs: a UD-type run on S took 57.2 s per chunk (ETA 15.25 min) because NVFP4 runs on the CPU. An all-Metal run took 19.9 s per chunk (ETA 5.30 min). So hybrid needs about 15 minutes and mirror about 5.
- Caveat: the candidate was selected on S from 4 picks (S scores 0.02931 to 0.03145), UD's layout was not. That bias favours us and is small next to the gap. S is still the right set. T stays reserved for release comparisons.
- Addendum C already allows a native-down build "as a quality reference only". Mirror is new and needs its own dated entry before it is scored.

## 3. Options

Hours are my estimates.

| Option | Cost (estimate) | Risk | What it would show | Thesis fit | Verdict |
|---|---|---|---|---|---|
| (a) Upload as is | 1 to 2 h | Low. A file that loses to UD, with a short shelf life | Nothing new | None | Do once, bounded, after fixes. No upkeep |
| (b) Hybrid, plus mirror | 1 to 2 h, about 21 min of scoring | Low. Loader must accept a down scale without a gate_up scale | Whether the gap is the down format, and whose allocation is better | Medium. Tests the map-to-allocator chain. Tells the thesis whether NVFP4 must be a level | **Do first** |
| (c) Port the Metal NVFP4 kernel | 8 to 16 h plus Daniel's review. Upstream merge time unknown | Medium. Removes our own file's edge. Review latency upstream | UD-type files at Metal speed on this Mac | Indirect. Speeds every native-down run here by about 2.9x | Decide after (b). Worth a 4 h spike |
| (d) Better NVFP4 to IQ4_NL re-encode | Unknown. Daniel's task | High. No exact mapping exists | How much of the down cost an encoder can recover | High for the thesis tool. Low for this release | Park until (b) gives the ceiling |
| (e) Stop the release track, write up per-pass | 20 to 40 h | Medium. Result is narrow today (one block, two levels) | The part nobody else has | Direct | **Main line after (b)** |
| (f1) Publish the measurements | 2 to 3 h | Low | An independent KL table of the public files under one reference | Low | Fold into (a) |

### (a) Upload as is

The work is done and the claim is true as scoped. The cost is small. The return is small too: a narrow audience, a lead that depends on an upstream gap, and no task evidence. The original pilot rule said a failed gate means "publish the measurements without a model". Addendum G changed that after the fact, with approval and in the open. A strict reader will still see moved goalposts. The defence is the rule Addendum G itself set: the FAIL first, with both margins. The release README does not do that yet (section 1.5).

Do it once if Daniel wants a public artifact. Conditions: both pending comparators scored and beaten, README fixed, the llama.cpp commit next to the Metal claim, and a line that the file is retired when Metal gains NVFP4. Then stop. No second version, no tuning.

### (b) Hybrid, and mirror

This is the cheapest experiment with the most to say. One of three things comes out.

1. **Hybrid clearly below UD-IQ3_XXS, and cand below mirror.** Our allocation is better at equal down in both columns. The whole gate loss was the down format. The map-to-allocator chain works. Then (c) has a prize: with a Metal kernel, a native-down file with our allocation could be the best file under the limit on every backend. That needs a new registered gate and a T score for the hybrid. Its allocation was fixed before T was seen, so T is still a fair test for it, but the decision to try is post hoc and must be said so.
2. **A tie.** The allocation is a wash against Unsloth's heuristic. The measured map bought nothing at this granularity. That is a useful negative for H3 planning: five blocks and tied passes are too coarse to beat a good prior.
3. **Hybrid above UD-IQ3_XXS, and mirror below cand.** Unsloth's allocation is better than ours. The Mac claim then rests only on the fact that nobody has re-encoded their file. I would not upload in that case, or I would upload with that sentence in the card.
4. **The two columns disagree.** Then the allocation interacts with the down format. Report it as that. For the Mac claim only the IQ4_NL column counts: mirror below cand is still bad news for it.

The same run prices the down format (cand minus hybrid, mirror minus UD). That number is the ceiling for (d) and the quality prize for (c).

Check before the run: the hybrid has a down scale and no gate_up scale on 20 layers. The generic scale pass in `llama-model.cpp` creates each projection's scale on its own and marks each optional (found in the source by this script: yes). So the file should load and scale correctly. This was read, not run.

### (c) Metal NVFP4 kernel

- What exists: 220 lines against a file layout that is gone. The sibling feasibility review puts the port at about a day plus testing and names one bug to fix on the way. My estimate agrees with it. The MoE variants are in the patch. `CLAUDE.md` assigns llama.cpp patches to Claude, each with a test that fails without it, and Daniel reviews.
- What it would show here: UD-IQ3_XXS and the hybrid at Metal speed. Today a native-down scoring run costs 57.2 s per chunk against 19.9 s. If thesis candidates use native NVFP4 as their 4.5-bit level, which section 2.1 says they should, every such run on this Mac pays that factor until the kernel exists.
- What it does to the release: it ends the candidate's reason to exist. That is the right outcome for users. A file whose value depends on a bug staying open is a weak thing to stand behind. A kernel helps every Mac user of every NVFP4 GGUF.
- Risk: correctness on 1152-wide rows and on the expert-indexed path, numerical parity with the CPU path, and upstream review time we do not control. A local patch is enough for our own runs.
- It is not thesis science. Guess: one to two working days to a tested local patch.

Verdict: not ahead of (b) and not ahead of G0 work. If (b) gives outcome 1, or if native NVFP4 becomes a thesis level, do a 4-hour spike: port, bit-compare against the CPU path, bench UD-IQ3_XXS.

### (d) A better NVFP4 to IQ4_NL re-encode

- Reserved for Daniel under `CLAUDE.md`. I give no design.
- Structure, from `ggml-common.h`: NVFP4 has one scale per 16 weights, IQ4_NL one per 32 with a fixed non-linear codebook. Inference: an exact re-encode is not possible in general, so some loss stays.
- The current IQ4_NL down already has a low weight error (0.0054). Guess: an encoder that knows the source grid recovers a part of the down cost, not all of it. If the down cost is near the gap (0.00881 on S), a part of it does not flip the gate.
- It fits the thesis tool, so it will be written anyway. It should not be written for this release.

Verdict: park. Let (b) give the ceiling first.

### (e) Stop the release track and write up the per-pass result

- This is what the scoop check found nowhere else: the same quantized block on one pass at a time, an additivity residual, and a layer-split control without weight sharing.
- State today: Fisher r is 0.675 [0.657, 0.693] for the pass split and 0.838 [0.822, 0.855] for the layer split. The controls license one narrow claim on B2 at the 2-bit level. The three additivity comparators disagree. The 3-bit rung has no layer-split control. B3 is untouched. The Gaussian single-pass conditions were not run, so one Addendum A prediction is untested.
- That is a method and a first result, not yet a finding that carries a note. Each missing piece is a handful of all-Metal runs at about 5 minutes each.
- Open items that block publication: Daniel's approval of the deviations in Addendum G section 2, his review of the delegated analysis code, and the major findings of `reviews/2026-11-21-code-review-analysis-opus.md` (the chunk parser and the missing tests on the gate path).
- The novelty has a clock too. One Mini repo already claims a visit-level sensitivity measurement, and another states the accumulation hypothesis as untested.

Verdict: this is where the hours should go.

### (f) Other things I see

- **f1. Publish the comparison table itself.** 7 files (5 public, 2 ours) scored on held-out text under one reference, with types and bytes read from the files. That has value without our model, and it is what the original rule promised after a FAIL.
- **f2. Mirror build.** Part of (b) above. It is the test of whether the Mac claim survives the obvious rival.
- **f3. Offer native NVFP4 as a level in the thesis allocator.** Same bytes as IQ4_NL down, no loss (section 2.1). This is a design note for Daniel, not a pilot run.
- **f4. Do not chase the unused 149 MB.** Spending it needs a reselection after T was seen.
- **f5. No task evaluation in the pilot.** It would be the right way to learn whether 0.00811 KL matters to a user. GSM8K test and all task sets are reserved for the thesis, so leave it.

## 4. Recommendation and order for the next 48 hours

The file has a real, small, short-lived edge. It is not worth more engineering. The pilot's lasting value is the per-pass method and what it taught about the down format.

1. **Tonight, no action.** Let the running chain finish the depth bench and the two pending comparators. If either beats the candidate on T1 and T2, the Mac-track rule fails and no model card goes out.
2. **Daniel, about 30 minutes.** Check what `logs/hf-upload-v1.log` points to and whether it is public. Approve or reject the Addendum G section 2 deviations. Write the dated entry for the mirror build and the 2 x 2 reading rule before any score exists.
3. **Run the 2 x 2 on S** (hybrid, then mirror; about 21 minutes of scoring, 1 to 2 h in all). One heavy run at a time, after the chain is idle.
4. **Close the release track, at most 2 h.** Fix the README order and the "looped core" sentence. Then one of:
   - outcome 1 or 2 of (b) and comparators beaten: publish once, with the 2 x 2 in the details, and the retirement line.
   - mirror below cand (outcome 3 or 4): publish the measurements, and say that a re-encode of UD's layout does better. No model, or the model with that sentence.
5. **Hours 6 to 40: per-pass work (e), each step registered before its data.** In this order: the layer-split control at the 3-bit rung, the Gaussian single-pass conditions, two more independent-noise draws, then the pass split on B3. Fix the chunk parser finding first, since every CI runs through it.
6. **Hours 40 to 48: turn the per-pass section of `drafts/note.md` into a standalone draft** for Daniel's review. The release section shrinks to the table and the FAIL.
7. **Only if (b) gives outcome 1, or Daniel decides native NVFP4 is a thesis level:** the 4-hour kernel spike (c). Otherwise leave a note that PR #20456 exists and is small.
8. **(d) stays with Daniel** and waits for the ceiling from step 3.

What I would not do: reselect the candidate, build a second release, or spend the G0 run-up on a file that one upstream merge retires.

## 5. What I did not check

- Whether the Hugging Face repository in the upload log is public, and what it contains.
- Whether the hybrid loads and scores correctly. I read the scale pass in the loader and ran nothing.
- The body of PR #20456 beyond its metadata, file list and kernel names. I did not read the review comments.
- Any speed or quality figure on a machine other than this M5 Pro.
- `private/`, including the handoff, so thesis-fit judgments rest on `CLAUDE.md` and `CHANGELOG.md` only.
- The pending comparators. They were not scored when this was written (0 of 2 in `kl.tsv`).
