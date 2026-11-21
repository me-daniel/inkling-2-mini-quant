# Synthesis: plan review of "Counting Passes", 21 Nov 2026

Inputs. Four independent reviews of the same redacted brief and prompt (`2026-11-21-plan-prompt.md`): GPT-6.2 Sol (xhigh), GPT-6.2 Astra (high), Fable 5.2 (high), Opus 5.5 (xhigh, **still running, added below when back**). Plus a primary-source fact-check of the brief's Verify items (`2026-11-21-factcheck-g0-sonnet.md`, Sonnet). Reviewers saw no conclusions from Claude and no fact-check results.

Checked by Claude on 21 Nov 2026 (not only reported by a reviewer):
- All 26 arXiv ids cited across the reviews exist, with matching titles and dates (arXiv API).
- llama.cpp PR states, tags and source via `gh`: #32217 merged 14 Nov (`5c008e0`, b12406); #32694 merged 20 Nov (`29b52d3`, b12600); #32695 (Unsloth) and #32704 (scale bug) open; `imatrix.cpp` accumulates with `+=` by weight name; `llama-quant.cpp` fallback table (see 1.4).
- Hugging Face API and cards: Dory1.0-4B-BF16 exists (created 16 Oct); hermitdave RCO 3.5 bpw, araysuter mixed 2-bit and netprojects 2.5-bit Mini MLX builds exist (all 20 Nov); ggml-org card says BF16 52.2 GB, Q8_0 KL 0.0030 and IQ4_NL 0.0095 against BF16 on 40 × 512 WikiText-2; Unsloth's protocol is "seven domains using identical llama.cpp settings" against UD-Q8_K_XL.
- weft-loom issue #39 exists (open, 20 Nov): a team migration plan that mentions a varloop pass sweep, not a study.

Not checked: reviewer cost estimates (labelled Est. by them), GCP prices, the Goodfire post, LOOM's router claim, the #32704 PPL figure (130.4 vs 7.65 is quoted from the PR text).

## 1. Facts that change the plan now

1.1 **Build target.** b12406 cannot load Mini. Pin b12600 (`29b52d3`) or later. P is the load-time key `inkling2_loop.loop.count` and honours `--override-kv` (code reading; ggml-org documents 1 to 6). Not yet run.

1.2 **NVFP4 scale bug in passes 2 and 3** (open PR #32704). Any number from looped layers on current master is suspect until it is merged or applied. This blocks every looped-layer measurement in G0.

1.3 **Two GGUF lineages.** Unsloth's files follow the unmerged fork (#32695: arch `inkling2`, `block_count 48`, native NVFP4 experts with scale tensors) and reportedly fail on upstream. Their KLD table uses an unspecified seven-domain corpus against UD-Q8_K_XL. "Reproduce 0.0595 ± 15%" has no defined target. All three reviews and the fact-check agree.

1.4 **Down projections cannot take the planned levels** (Sol point 16; confirmed in source, no other reviewer caught it). `ffn_down_exps` has row width 1152, not a multiple of 256. `llama-quantize` (`tensor_type_fallback`, `src/llama-quant.cpp`, master `29b52d3`) silently maps IQ1/IQ2/IQ3/IQ4_XS to IQ4_NL, Q2_K/Q3_K to Q4_0 and Q4_K to Q5_0. A block "at IQ2_XXS" therefore has 4.5-bit down projections, which hold a third of its expert weights. The level grid must name the type per projection, use 32-block types for down (Q2_0, IQ4_NL, Q4_0) or quantize gate_up only, and check the fallback warnings in every build log. Unsloth's Q2 file uses Q2_0 for down (fact-check header dump).

1.5 **The reference does not fit on upstream.** ggml-org's BF16 GGUF is 52.2 GB against 48 GB RAM, because the converter decodes the experts. Options: a Q8_0-experts reference with its floor measured once against BF16 on a larger machine, or the fork's native NVFP4 type if it can be made to load with P control.

1.6 **imatrix already pools over visits.** One tensor name per physical weight (`layers[il] = layers[layer_phys(il)]`), accumulation by `+=`. Pooled is the default; first-visit-only is the variant that needs a patch.

1.7 **Disk.** `llama-perplexity` stores about 2 bytes per vocabulary entry per scored token. At vocab 201,024, 64 × 4,096 chunks give about 52.7 GB per reference logits file, per P and per corpus. Five P values and two corpora approach 0.5 TB.

1.8 **Smaller corrections to the brief.** The HF key is `text_config.loop_config.num_passes`, not `loop_num_passes`. Embeddings and unembedding are untied. Varloop and Mini share the layout. The "last-sixth rule" is a training-time choice (NVIDIA's 15%, rounded to their 6-layer ladder), not a sensitivity claim (Fable, from the TML blog). The perplexity-trap example mixes two papers: PPL 11.51 to 14.38 is from Feedback Exposure, GSM8K 73% to 7% is Tilted Bowl's per-channel INT4 row on n = 100 (Fable, Astra).

## 2. Where all three reviewers agree

2.1 **H1's curve-shape reading is not identifiable.** KLD is roughly quadratic in the logit perturbation. The same weight error applied on every visit gives aligned perturbations, amplitude ∝ P and KLD ∝ P² with no amplification at all. The reference also changes with P, and varloop at P ≠ 3 is off-distribution (RL at 3 only; the card warns behaviour "may be unpredictable" at other counts). "Flat = once, linear = per visit, superlinear = compounds" cannot be read off a KLD-against-P curve.

2.2 **The central test should be a per-visit intervention at fixed P = 3.** Quantize the same weights on visit 1 only, 2 only, 3 only, then all three, on one core block. Compare the all-visit result with the sum of the singletons. Ranked first or second by every reviewer. It answers "which visit carries the damage" and "does it add up" without off-distribution P. Keep a small P = 1/3/5 sweep on varloop for deployment relevance. Fable adds: fit the exponent on hidden-state error at the loop exit, not on KLD. Requires untying weights per visit in the llama.cpp graph (a patch to `inkling2-loop.cpp`, est. 3 to 5 days).

2.3 **B1 cannot test Tilted Bowl's contraction claim.** Each pass restarts from layer 5's output, so B1's error is re-injected every pass. Fable: the loop reads the previous pass only through rank-64 gated messages, so the prior expectation is "last pass dominates plus a small leak". Drop the Tilted Bowl framing or test it only with the per-visit design.

2.4 **Gates keep their dates, pass conditions change.** G0: P verified (24/48/72 layer visits at P = 1/3/5), one runtime, a reference that fits, measured run-to-run and conversion floors, one audited intervention. Unsloth becomes a quoted external number. G1: per-visit result with an interval, null allowed. G2: an internal contrast on held-out data, win or lose. G3: "no significant reversal" or bounded disagreement, not "ordering agrees".

2.5 **H3 must be an internal contrast.** Same candidate tensors, quantizer, imatrix, byte cap and solver; only the visit weighting differs. Allocate on one split, test on another. Block-level first (five block measurements cannot identify 24 layer scores). Check that the three policies actually choose different builds at the budget, or H3 is vacuous. Unsloth and community builds are external anchors only. Loopy (2610.05265) is the closest precedent: rankings change with depth, selection by loss at deployment depth.

2.6 **H2 needs matched exposure, or it shrinks.** Pooled versus first-visit confounds visit diversity with three times the rows and better rare-expert coverage. Sol and Astra: compare first-only against visit-stratified at equal row counts, log per-expert counts. Fable: a strawman, since nobody ships first-only and Feedback Exposure already showed summed Hessians win on 9 of 9 checkpoints; cut to a one-day pre-check (cosine of per-visit second moments, drop H2 if above 0.99).

2.7 **H4 and H5 are cheap, descriptive, and need margins.** H4: same layer across visits at fixed P, reference-vs-reference noise floor first, stratify by the top-8/top-9 margin; the first affected router cannot differ. H5: an equivalence test with a pre-registered margin, B5 alone, plus an equal-size earlier-layer control.

2.8 **Tier 1 is the budget risk.** Six builds × three seeds × 2,191 items = 39,438 generations. Estimates differ (2.10) but all say: run a 30 to 50 item pilot per task first, then size the panel. McNemar over pooled seeds violates independence; cluster by item.

2.9 **Integrations: stay narrow.** No GSQ, RCO, LittleBit-2, MoESQ/QuTLASS or QTIP integration. Use any released compatible build as an external baseline. Borrow two zero-cost ideas: Loopy's selection at deployment depth (allocate at P = 3, test transfer to P = 5) and LoopQuant's pairwise interaction checks. LittleBit-2 (2603.00042) is sub-1-bit factorization plus QAT, validated on dense Llama only; for Inkling it means 4 to 12+ weeks and likely over 200 GPU-h. GSQ needs a per-family wrapper and lists 4 × H200 for a 30B MoE.

2.10 **Novelty is the controlled visit measurement on this regime, not a method.** Not novel: pooling over loops, loop-wise allocation, expert ILP (MODE), routing consistency (VSRAQ), exact budgets (RCO), nested lower precisions (Matryoshka Quantization 2502.06786, MatGPTQ). Possibly novel: which visits carry shared-weight error in a restart-plus-message looped MoE trained in FP4, and whether that changes the best sub-4-bit allocation at fixed bytes. Requantizing below 4 bits from a native FP4 grid with no BF16 master (Verify, one search each).

## 3. Where they disagree

| Topic | Sol | Astra | Fable | Claude's call |
|---|---|---|---|---|
| Tier-1 cost | 137 to 219 GPU-h decode at 50 to 80 tok/s aggregate | ceiling 646 M tokens; 200 GPU-h needs 56 tok/s sustained | 40 to 75 GPU-h | The spread is aggregate throughput, which nobody has measured on a G4. Pilot decides. Plan the reduced panel (reference plus two allocations, one seed, GSM8K full plus one code set) and expand only if the pilot allows. |
| H2 | keep, equal-exposure | keep, visit-stratified | cut to pre-check | Run Fable's one-day pre-check first. If per-visit statistics differ, run the equal-exposure contrast. If not, report it and drop H2. |
| Second model | costs more than Mini confirmation | only as fallback if Inkling fails | add Dory1.0-4B (dense, 1 to 4 loops, BF16 master) | Dory as the G0 fallback instrument and an optional generality check after G2. Its llama.cpp support is an unmerged PR (#30306, Verify) and its licence is "other" (OpenMDW). |
| Nested E2M1 level | bounded ablation, drop if round-trip exceeds 2 days | keep only after identity round-trip; scales cost 0.5 bit/weight at 2 to 3 bits | prior art is Matryoshka; novel only for native FP4 | Ablation only, accuracy claim only, no matched-size claim without a packed format. |
| Transformers path | open, lm_head claim unconfirmed | open, could not confirm blocker | float32-only, reporter calls it non-blocking; SGLang and vLLM support pass overrides | A native-engine sanity check (vLLM or SGLang on a G4, shared KV and DSpark off) is a G0 fallback. Not the main path. |

## 4. Scoop risk

**Next 2 days (all reviewers, Est.).** More GGUF ladders (bartowski, mradermacher, lmstudio-community), Unsloth re-uploads that may move their table, more MLX 2 to 3-bit builds, loader fixes. All baselines. No reviewer expects a method paper from the loop-quant groups or IST-DASLab in this window.

**2 to 6 weeks.** The Loopy, LoopQuant and LoopQ groups are the main direct threat to H1 to H3 (a revision adding Inkling or Dory). IST-DASLab GSQ-RCO GGUFs are plausible (they shipped them for a 384-expert MoE) and would beat any visit-blind allocator on H3's practical outcome, though not the visit decomposition. AutoRound, llm-compressor and GPTQModel recipes follow the transformers merge.

**Already out (verified).** Visit-blind sub-4-bit Mini builds: hermitdave "RCO" 3.5 bpw, araysuter mixed 2-bit (2-bit gate/up, 3-bit down, 4-bit tail, a partial H5 overlap), netprojects 2.5-bit, Unsloth down to IQ1_S. No arXiv paper on quantizing Inkling (reviewers' searches, 20 to 21 Nov). NariQ (2611.26157, 19 Nov) already calibrates on reference-generated agent traces, so the trace corpus has precedent.

**Implication.** The visit intervention is the defensible part and the one most exposed to the loop groups. Publish the G1 map note quickly, as planned, and make it the per-visit result.

## 5. Recommended changes, ranked

1. **G0 rebuilt** (2 to 4 days, under 5 GPU-h): pin b12600 plus the #32704 fix; verify layer visits at P = 1/3/5 and cached-decode against full-prefix logits; pick one reference that fits 48 GB and measure its floor against BF16 once; measure run-to-run floor; dump the tensor types of every build and fail on unexpected fallbacks. Drop Unsloth ± 15%.
2. **Per-visit intervention at P = 3 as the primary H1 test** (1 week plus the graph patch, under 10 GPU-h). Varloop P = 1/3/5 sweep secondary. Decide who writes the untying patch (6.1).
3. **Fix the level grid** (1 day): per-projection types, 32-block types for down, extra free levels (IQ1_M, IQ2_XS, IQ2_S, IQ2_M, IQ3_XS, IQ3_S on gate_up). Count stored bytes from the file, not from labels.
4. **H3 as an internal, held-out, block-level contrast** with a check that the policies differ (3 to 7 days, Mac only).
5. **Data and disk plan** (1 day): reasoning traces primary, WikiText through the chat template as the comparability arm; split calibration, selection and test by prompt; 32 chunks or top-k logits; build, evaluate, delete.
6. **Tier-1 pilot before committing the panel** (1 to 2 days, 3 to 8 GPU-h); cluster bootstrap by item; output length and truncation as registered outcomes.
7. **H2 pre-check, H4 and H5 as bounded side results**, each with a pre-registered margin; nested E2M1 as an ablation.

Every change to H1 to H5, metrics or gates goes into `CHANGELOG.md` with a date before any G0 data exists. The pre-registration has not been mirrored publicly yet, so this is the cheapest moment.

## 6. Open decisions for Daniel

6.1 **Who writes the per-visit untying patch** in `inkling2-loop.cpp`. It is the instrument for the main test. The core-functions rule covers the analysis but not this patch.
6.2 **Reference choice**: Q8_0-experts on the Mac with a floor measured on GCP, or a native NVFP4 path via the fork.
6.3 **Whether to apply #32704 locally** or wait for the merge.
6.4 **Dory1.0-4B**: fallback only, or a planned generality check.
6.5 **When to mirror `PREREG.md` publicly**: before or after folding in these changes. Public push needs your go.

## 7. Opus 5.5 (xhigh)

Pending. To be added when it returns, with any point that changes sections 1 to 6.
