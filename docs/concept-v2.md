# Concept v2: test further, compare further, improve

Written 21 Nov 2026, evening, after v1 went public. It turns what the pilot taught us into a plan and a way of working. Every run below is registered in `CHANGELOG.md` before its data exists (Addendum H onwards).

## Where we stand

- v1 is public: an 11.51 GB GGUF that runs fully on Metal. It failed the registered gate against Unsloth's UD-IQ3_XXS and passed the post hoc Mac comparison.
- The gap to UD is probably the down projections. UD keeps the release's NVFP4 down tensors exactly. We re-encode them to IQ4_NL at the same bytes and lose information. This is an inference until the 2 x 2 below has run.
- The per-pass test works and gives a loop-specific difference against a layer-split control. It is one block, one amplitude and one noise draw so far.
- The additive allocator overpredicted KL by 19 to 27% and ranked its own picks wrongly. Measurement picked the candidate, not the model.

## Lessons for the workflow

1. Register first, then run: every result that went public was registered before its data, and that kept the FAIL honest. Keep doing it, in small dated amendments.
2. Numbers only from scripts: the card, figures and report are rendered from result files. Keep the large inputs optional (file sizes now live in `results/pilot/file-bytes.tsv`).
3. Chains need a dry run: two chain failures today came from shell mistakes (a helper used before it was defined, a zsh variable that did not split). Every new chain gets `sh -n`, a stub run with `DRY=1`, and helpers defined at the top.
3b. Never edit a shell script while a chain is running it. `sh` reads scripts as it goes, so an insertion shifts what it reads next. Edit before launch, or put new steps in a new script.
3c. One heavy job at a time on the shared Mac, GPU and CPU alike: the GPU lock (`scripts/gpu-lock.sh`, shared with the sibling repo) for runs, and at least 30% free memory before splices or quantize jobs. Swap reached its limit once when an unlocked run met the sibling's 22 GB capture.
4. Logits are the disk bottleneck: 13 GB per S run. Analyse right after the run, keep only the base, delete the rest (regenerable bit for bit, run-to-run KL is exactly 0).
5. Downloads are slow here (about 2 to 8 MB/s): start them first, in parallel with GPU work.
6. Speed needs an idle machine: KL is deterministic and can run next to other work. llama-bench cannot. Speed runs go in a separate, announced slot.
7. The comparison set moves daily: about 70 Inkling repos changed in 8 hours. Rescan Hugging Face right before any claim and date the claim.
8. The additive model is a ranking hint, not a decision: decide by measurement on S, confirm once on held-out text.
9. Reviews at checkpoints, from different families: Opus for audits, Fable for direction, Astra for wording. Not after every step.
10. Held-out text is used up by looking: T1 and T2 were seen once for v1. A v2 candidate needs a fresh set T3, built before the candidate exists.

## Tracks

### A. Diagnose the gap (tonight)

The 2 x 2 on S, from the direction review:

| | our layout | UD's layout |
|---|---|---|
| IQ4_NL down (Metal) | cand (0.02931) | mirror: UD with our IQ4_NL down in layers 0-19 |
| NVFP4 down (CPU on Mac) | hybrid: cand with UD's NVFP4 down | UD (0.02050) |

Reading rule: if hybrid is at or below UD, our layout is as good as theirs and the whole gap is the down format. If mirror is below cand, their layout is better even at equal down, and a re-layout is worth more than a kernel.

### B. Compare further (tonight, KL only)

1. The two registered comparators (IsValorum NanoPlus, prithivMLmods Q2_K), already queued.
2. bartowski Q2_K (11.88 GB), the last standard-type file under 12 GB, so the card can say "under 12 GB" if we stay ahead.
3. Speed for new files only in an idle slot.

### Update, 21 Nov ~23:50: what A found

Measured on S (`results/pilot/addendum-h.md`):

| Build | KL on S |
|---|---|
| cand | 0.02931 |
| cand with UD's NVFP4 down (hybrid) | 0.02813 |
| cand with UD's non-expert tensors | 0.02941 |
| cand with UD's gate_up experts | 0.02121 |
| UD with our IQ4_NL down (mirror, fully on Metal) | 0.02149 |
| UD-IQ3_XXS | 0.02050 |

- The down format explains little. UD's gate_up experts explain the whole gap, at about 0.22 GB more bytes.
- So the kernel (C1) no longer looks like the way to beat UD with our file. It would let UD's own file run fully on Metal, which helps Mac users and ends our speed edge.
- The open question is whether UD's gate_up wins by type choice per layer or by better quantization at the same type (a CPU comparison against BF16 is running, `scripts/gu-error.py`). That decides C2 to C4.
- The mirror is a fully-Metal file at 0.02149 on S. Releasing it would be a Metal-native re-encode of Unsloth's layout, with clear credit; that is Daniel's call.

### C. Improve (after A; v2 candidate)

Order depends on the 2 x 2:
1. Metal NVFP4 kernel (patch 0008, about 200 to 250 lines, mostly from MXFP4; `reviews/2026-11-21-metal-nvfp4-feasibility.md`). With it, a native-down file runs fully on Metal, and so do Unsloth's files. It is a contribution upstream as much as a release lever. Daniel reviews the diff before any result depends on it, and he decides on an upstream PR.
2. Non-expert bytes: UD keeps output at Q5_K and token_embd at Q4_K, we use Q6_K everywhere. Two map entries on S price the saving, which can go to gate_up.
3. Measured search instead of the additive allocator: start from cand, try one block change at a time on S, keep what helps under the byte limit. Each step is one S run (about 6 min).
4. NVFP4 to IQ4_NL re-encode: Daniel's tool (CLAUDE.md). The hybrid gives its ceiling.
5. T3 before any v2 candidate exists; v2 is scored once on T3, T1 and T2 are reported but marked as seen.

### D. The per-pass science (tonight and tomorrow)

Each condition is registered before it runs; logits are deleted after analysis.
1. Layer-split control at the 3-bit rung (does the pass vs layer difference survive at small amplitude?).
2. Gaussian noise on one pass at a time (Addendum A point 7: is independent noise about the sum of single-pass effects?).
3. Two more independent-noise draws (the tied/independent ratio still rests on one independent draw).
4. The pass split on B3 (layers 12-17).
5. A tiny-amplitude noise rung: r stayed at 0.75 at 3 bits although KL fell 7.7-fold. If r stays high at very small noise, the cause is not amplitude (routing flips are one guess).

### E. Release v2 (tomorrow, after Daniel's go)

Card v2 only changes if a result changes: new comparators, the 2 x 2 line, depth and memory are already in. A new file only if C produces one that beats cand on S and then on T3.

## Tonight's run order

1. GPU, KL only: the registered comparators, then the 2 x 2 (mirror, hybrid), then D1 to D3, then bartowski Q2_K.
2. In parallel, CPU or worktree: the Metal NVFP4 patch spike with `test-backend-ops` (no result depends on it until Daniel reviews), T3 data preparation, the wording pass on the GitHub README and report.
3. Morning: one brief with what ran, what changed, and the decisions.

## Decisions for Daniel

- Approve Addenda F, G section 2 and H.
- Patch 0008: review the diff, then decide whether to open an upstream PR.
- v2: only if A and C give a better file on T3.
