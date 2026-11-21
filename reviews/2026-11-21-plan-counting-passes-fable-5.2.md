# Review: plan, Counting Passes design

- Date: 2026-11-21
- Checkpoint: plan (extended: hypotheses, prior art, scoop forecast, novelty, feasibility, integrations)
- Model: Fable 5.2, effort high, Claude Code subagent with WebSearch/WebFetch
- Input: redacted design brief from `private/counting-passes-claude-code-handoff.md` (sections 3-8, no personal details), repo HEAD a0e251f (no design files tracked yet)
- Prompt: shared review prompt, saved as `reviews/2026-11-21-plan-prompt.md`

## Reply (verbatim)

# Review: "Counting Passes" design brief

Reviewer: Fable 5.2. Date: 21 Nov 2026. All web sources were fetched on 21 Nov 2026; the source list with dates is at the end, cited as [S#].
Labels: **V** vendor or authors on their own work, **I** independent (including my own reading of public code), **Est.** my estimate, **Verify** could not confirm.

## 0. Facts in the brief that are already stale or wrong

0.1 **Pass count is settable.** `--override-kv inkling2_loop.loop.count=int:N`, N = 1 to 6, default 3, fixed at load (V, [S3]). Loop support is llama.cpp PR #32694, merged 20 Nov (V, [S5]). PR #32217 is the non-looped Inkling 2 base, merged 14 Nov [S7]. The loop range is half-open `[core_start, core_end)` (V, [S6]), so layers 6 to 17.

0.2 **Two incompatible GGUF lineages.** Unsloth's files need the Unsloth fork (draft PR #32695, release `b12588-mix-c0c3de8`): `block_count = 48`, native NVFP4 experts with per-expert scale tensors (V, [S8][S6]). Users report they fail on upstream with `blk.24.attn_q.weight not found` (I, [S6]). Upstream stores experts as BF16, Q8_0 or IQ4_NL only (V, [S4]).

0.3 **Unsloth's KLD protocol is not 64 × 4,096 WikiText-2.** Their page says "across seven domains using identical llama.cpp settings" and names no dataset, token count or context (V, [S8]). The brief's protocol is Verify. "0.0595 ± 15%" is therefore not a well-defined target.

0.4 **The native reference does not fit the Mac on upstream.** The BF16 GGUF is 52.2 GB against 48 GB RAM (V, [S4]). Experts are about 24.5B of 26.4B parameters (Est. from the brief's dimensions), so any BF16-expert reference is over 49 GB.

0.5 **The "perplexity trap" number is stitched from two papers.** PPL 11.51 to 14.38 comes from Feedback Exposure's table (Verify; found via search snippet). GSM8K 73 to 7 is Tilted Bowl's per-channel INT4 row, n = 100, Wilson half-width about 10 points (V, [S19]). Different quantizers. Tilted Bowl's own statement is about teacher-forced NLL on Recurrent-OLMo-2.

0.6 **The "last-sixth rule" is not a sensitivity claim.** TML chose 1/6 because NVIDIA's recipe uses 15% and their layer ladder scales by 6 (V, [S11]). It is also a training-time rule.

0.7 **transformers is less blocked than stated.** PR #51050 is open, but the lm_head bug affects only float32 loading and the reporter calls it non-blocking (I, [S9]). SGLang lists RTX PRO 6000 and RTX 5090 as verified with a BF16 GEMM backend and has `--inkling2-loop-num-passes` (V, [S13]).

## A. Hypotheses

**A1. H1 (ΔKLD(P) grows faster for B2/B3 than B4/B5).** Not identifiable as posed.

1. The mapping "flat = once, linear = per visit, superlinear = compounds" is wrong for KLD. KLD is roughly quadratic in the logit perturbation. Independent per-visit errors give KLD ∝ P. The same weight error applied on every visit gives aligned perturbations, amplitude ∝ P and KLD ∝ P², with no amplification at all. Fit the exponent on hidden-state error norm at the loop exit, not on KLD (Est.).
2. Loop topology undercuts the trichotomy. Each pass restarts from layer 5's output and reads the previous pass only through gated rank-64 messages (V, [S1][S6]). Error from pass k < P reaches layer 18 only through that bottleneck. The prior expectation is "last pass dominates plus a small leak", i.e. near flat. B1 error is re-injected on every pass, so B1 does not test contraction. Tilted Bowl concerns iterated maps that settle and says it "says little" about non-settling four-loop models (V, [S19]). Drop the Tilted Bowl framing.
3. P ≠ 3 is off-distribution. The card says behaviour "may be unpredictable at pass counts other than 3", and later training differs from Mini (V, [S2]). The HF blog says the model runs "1, 2, or 3" passes while the card says 1 to 5 (V, conflicting). A P-slope mixes training exposure (1:2:6:2:1) with recurrence dynamics.
4. The reference changes with P. At P = 1 the model is much weaker (SWE-bench 56.3 against 78.7, V [S2]), so its output entropy differs and ΔKLD is not on a common scale across P.
5. Blocks differ in size (6, 6, 6, 2, 4 layers) and source grid (NVFP4 against MXFP8). Compare ΔKLD per GB saved.
6. Falsifier: a pre-registered exponent b in error ∝ P^b with a bootstrap CI. b ≤ 0.3 for B2/B3 kills "per visit".
7. **Cheaper and cleaner design:** untie weights at inference on Mini at P = 3. Quantize experts of B2 or B3 on visit k only (k = 1, 2, 3), two levels: 12 configurations plus "all visits". Then test additivity: E(all) against ΣE(k). This needs no off-distribution P and answers which visit carries the damage. Add matched-norm Gaussian noise per visit as a control. Cost: a llama.cpp patch (3 to 5 days, Est.) or hooks on the transformers PR branch on one 96 GB GPU (about 10 GPU-h, Est.).

**A2. H2 (pooled imatrix beats first-visit imatrix).** A strawman.

1. llama.cpp's collector keys statistics by weight name and adds with `+=`, so looped tensors are already pooled over visits; MoE counts are per expert (I, code reading [S14]). ggml-org's imatrix was computed at 3 passes (V, [S3]). Nobody ships first-visit-only.
2. Feedback Exposure already shows summed Hessians beat one-step GPTQ on 9 of 9 checkpoints (V, [S18]). H2 is a diagonal-Hessian replication.
3. Pre-check, one day: dump per-visit second-moment vectors and compare. If cosine similarity is above 0.99, drop H2 (Est.).
4. Better questions: visit-weighted pooling using A1.7's weights, and calibration-depth mismatch (imatrix at P = 3, deployed at P = 5), which is Loopy's finding on Ouro only (V, [S16]).
5. Falsifier: paired chunk-bootstrap CI of ΔKLD(first-only minus pooled) includes zero at IQ2_XXS on B2+B3.

**A3. H3 (measured > summed-per-visit > counted-once, at matched size).**

1. It may be vacuous. With 24 layers and three levels, the three weightings can return the same allocation. Show first that they differ at both budgets.
2. The levels are too coarse. At 9.14 GB, with non-experts at Q8_0, experts get about 2.3 bpw; only about a quarter can sit above IQ2_XXS (Est.). Add IQ1_M, IQ2_XS, IQ2_S, IQ2_M, IQ3_XS, IQ3_S; they are free in `llama-quantize`.
3. "Measured end-to-end" wins in-sample by construction. Allocate on one split, test on another domain.
4. "Beats Unsloth" confounds allocation with their undisclosed imatrix, native-NVFP4 source, lower-bit non-expert tensors and a different runtime (0.2). The identifying contrast is internal: same quantizer, same imatrix, same non-expert treatment, only the visit weighting differs. Unsloth is an external anchor.
5. An end-to-end loss-driven allocator is implicitly visit-aware. RCO does exactly this under an exact size budget and already ships per-layer expert types in GGUF (V, [S22][S23]). The claim must be the decomposition (how much visit multiplicity explains), not the solver.
6. Ten pairwise configurations test additivity near the clean point; the allocator operates at KLD 0.03 to 0.09. Add leave-one-block-out at the operating point (all low, one block native): 5 to 10 extra builds.
7. Falsifier: visit-aware and visit-blind allocations differ, yet the held-out paired CI includes zero at both budgets.

**A4. H4 (router Jaccard falls with pass index).**

1. Confounded with routing margin: later passes may have smaller top-8 to top-9 gaps regardless of error. Condition flips on the margin.
2. Confounded with depth: compare the same layer across passes (12 layers × 3 passes, repeated measures).
3. LOOM reports routers re-select the same experts across loops (V, [S26]); Goodfire has an observational write-up on information flow across loops (Verify; primary not reachable).
4. Feasible cheaply: the imatrix collector already reads the `ids` tensor of `MUL_MAT_ID` (I, [S14]); dumping it needs a small patch plus a visit counter.
5. Falsifier: within-layer slope over pass index ≥ 0.

**A5. H5 (B5 tolerates about 4 bits).** Well posed only after "negligible" gets an equivalence margin (TOST against the run-to-run floor, F3). It tests inference-time tolerance, not the vendor's rule (0.6). Community builds already run B5 at 3 or 4 bits without isolating it (I, [S31]). Falsifier: ΔKLD per GB saved at Q4_K for B5 is not below that of B1 to B4. Cost: one build.

**A6. Controls missing across H1 to H5.** A run-to-run floor (same file twice, CPU against Metal, two batch sizes). An RTN arm without imatrix. A uniform allocation at each budget. The Q8_0 floor: ggml-org reports Q8_0 at KLD 0.0030 against BF16 (40 × 512 WikiText-2, V [S4]), which is 28% of Unsloth's "4-bit floor" of 0.0107. The brief's Est. that the grid switch causes the 4-bit loss is untested; the nested level and a Q8_0-experts-only build test it directly.

**A7. Float against NVFP4 activations.** Low concern: SGLang's verified consumer-GPU path also uses BF16 GEMMs and reports GSM8K 93.6 to 94.2% (V, [S13]). Report it as a limitation; do not spend compute on it.

**A8. WikiText against traces.** Raw WikiText without the chat template is off-distribution for a reasoning model; ggml-org rendered calibration data through the template (V, [S3]). Make traces primary and WikiText the comparability arm.

## B. Prior art and scoop risk

**B1. Looped-model PTQ (all V, all confirmed on arXiv).** LoopQ (8 May 2026, W4A4). Loopy (4 Oct; rankings change with depth, shown on Ouro-1.4B at T = 1 to 4; no MoE, no sub-4-bit weight-only, no allocation). LoopQuant (12 Oct; ILPA allocates activation bits per loop with Shapley sensitivities and pairwise interactions; weights fixed). Feedback Exposure (25 Sep; nine checkpoints up to 4.17B; 17 to 102% gap closed). Tilted Bowl (30 Sep; 33 models, no MoE, no layer-wise mixed precision). KV side: ResidualQuant, LoopVQ, LoopKV (3 Nov; per-(loop, layer) planning). None uses MoE, a natively FP4-trained model, a restart-plus-message loop, or weights below 4 bits.

**B2. New since the brief.** Dory1.0-4B (NVIDIA; arXiv 2 Nov, weights 16 Oct): dense looped model, 8 input, 20 recurrent, 8 output layers, 1 to 4 loops, BF16, MXFP8 and NVFP4 checkpoints, OpenMDW-1.1, llama.cpp key `dory.num_loops` in unmerged PR #30306 (V, [S27]). It is a second variable-loop instrument with a BF16 master.

**B3. Inkling-specific, one day after release (I, HF listing [S31]).** 22 quantized repos. Sub-4-bit: Unsloth down to IQ1_S; InterpreticAI IQ3_XXS; netprojects MLX 2.5-bit (KL 0.19 on 488 positions) and 3-bit; araysuter mixed 2-bit; hermitdave "RCO" 3.5 bpw (a budgeted mixed-precision allocator, all 48 expert tensors at the 3-bit floor, KLD 0.0669). All are visit-blind. No EXL3, AutoRound, bartowski or ISTA-DASLab repo yet. No arXiv paper on quantizing Inkling (I, arXiv search).

**B4. Nearest plan overlap.** Moissanite-Labs/weft-loom issue #39 (20 Nov) plans a varloop pass sweep and a "loop-agnostic vs loop-accumulated error" note, with no error-against-P experiment (I, [S32]). Low threat; it shows the idea is obvious.

**B5. Forecast, next 2 days (Est.).** bartowski, mradermacher and lmstudio-community imatrix ladders on upstream. Unsloth re-uploads to match upstream (the maintainer said he would check a shard rewrite, V [S5]), which may move their KLD table. More MLX 2 to 3-bit builds. All baselines.

**B6. Forecast, 2 to 6 weeks (Est.).** transformers PR merges; then AutoRound, llm-compressor and GPTQModel recipes. EXL3 once turboderp adds the architecture. ISTA-DASLab GSQ+RCO GGUFs are plausible: they shipped them for a 384-expert MoE and have NVFP4 MoE work (MoESQ) (V, [S23][S24]). A loop-group revision adding Inkling or Dory is plausible within 1 to 3 months.

**B7. Overlap.** H1: nobody has it; moderate risk over 6 weeks. H2: already taken (A2). H3: a GSQ+RCO release would beat Unsloth and subsume the allocator, though not the visit decomposition. H4: partly Goodfire and LOOM. H5: anyone can do it in an afternoon. Everything in B5 is a baseline only.

## C. Novelty

**C1. Not novel.** Pooling calibration over loops; loop-wise allocation; expert-level ILP (MODE); routing consistency (VSRAQ); exact-budget type assignment (RCO); a KLD ladder; nesting lower precisions in a higher one (Matryoshka Quantization, Feb 2025, V [S29]).

**C2. Novel after B.** (a) An error law against pass count, and against which visit, for shared weights in a production-scale looped MoE trained in FP4. (b) The restart-plus-gated-message topology, where prior art's iterated-map intuition may not hold. (c) Requantizing below 4 bits from a native FP4 grid with no BF16 master, including nested E2M1 subsets; I found no prior work on this (Verify; one targeted search).

**C3. One sentence.** "In a looped MoE whose passes restart from a shared prelude, we measure which visits carry shared-weight quantization error and how it scales with pass count, and show whether that law changes the best sub-4-bit allocation at matched size under an identical quantizer."

**C4. Enough for a thesis plus workshop paper.** The per-visit untied experiment (A1.7); a second topology (Dory or Ouro) showing the law depends on loop design; a held-out prediction (law fitted at P ≤ 3 predicts P = 4, 5); task-level confirmation; a released llama.cpp per-visit statistics patch. A null (flat law, visit-blind allocation is fine) is publishable if the per-visit evidence is clean. For a Cognitive Science thesis, register one more outcome: does quantization lengthen chain of thought more at low P? TML reports GPQA reasoning tokens falling from 3,703 to 1,661 between 1 and 5 passes (V, [S1]); Lotfi et al. report quantization lengthens CoT (V, [S30]).

## D. Getting it done

**D1. Toolchain split (0.2).** Test on days 1 to 2: load Unsloth's Q8_K_XL on upstream and on the fork; check the override works on the fork. Kill: not loadable on a build with P control by 6 Dec. Fallback: self-built upstream baselines; Unsloth's numbers quoted, not reproduced.

**D2. Reference does not fit (0.4).** Test: measure KLD of a Q8_0-experts reference against BF16 once on a 96 GB or larger machine. Kill: above 0.001. Fallback: produce reference logits on GCP, or use the fork's native type.

**D3. Disk.** `llama-perplexity` writes 2 × (n_vocab + 4) bytes per scored token and scores the second half of each chunk (I, [S14]). At vocab 201,024 that is 52.7 GB per 64 × 4,096 reference, about 263 GB for five P values, and the same again for traces (Est.). Single-block builds with the rest at Q8_0 are about 22 to 27 GB each. Build, evaluate, delete; use 32 chunks; or patch to top-k logits.

**D4. Flat law.** If A1.7 shows last-pass dominance, H3 has nothing to exploit. Test in weeks 3 to 4. Pivot: the contribution is "visit-blind is sufficient here, and why", plus Dory as contrast.

**D5. Tier-1 power.** GSM8K-500 at about 90% accuracy detects roughly 4-point paired differences at 80% power; 2 points needs about 2,000 items (Est.). G3 ("ordering agrees") will fail by noise. Six builds × 2,191 items × 3 seeds is 39,438 generations, about 40 to 75 GPU-h (Est.). Fallback: two builds per budget, full GSM8K (1,319) plus one code set. McNemar over pooled seeds violates independence; use a cluster bootstrap by item.

**D6. Moving targets.** Model repos, Unsloth files and PRs changed within the last hours [S31]. Pin HF revisions and commit hashes on day 1.

**D7. Memory at long context.** A user reports compute buffers that scale with micro-batch × KV length because the relative-position bias forces the non-flash path (I, [S5]). Test `-ub 512` at 4,096 context on day 1.

**D8. Scope.** Five hypotheses, a nested-grid tool, an allocator and four benchmarks is too much for 24 weeks. See G.

**D9. Minimum viable thesis.** Block × level map at P = 3 on Mini, per-visit attribution, the B5 test, one visit-aware against visit-blind allocation at tier 0, GSM8K only.

**D10. Gates.** G0 (6 Dec): realistic once the ± 15% criterion is replaced by "run-to-run floor measured; reference fits; override verified". G1 (3 Jan): realistic; add the per-visit result as its criterion. G2 (31 Jan): redefine as internal contrast; "beats Unsloth" is a stretch. G3 (28 Feb): redefine as "no significant reversal".

## E. Integrations against de-risking

**E1. GSQ.** Adds a stronger quantizer. Needs a per-family wrapper; lists 4× H200 for a 30B MoE, no runtimes given (V, [S21]). Blocked by transformers and by 8-GPU nodes at $51 to 53/h. **No.** Evaluate their GGUFs as an external baseline if they appear (zero cost).

**E2. RCO.** Adds an exact-budget solver. Needs gradients through a PyTorch forward: blocked. A DP or knapsack over llama.cpp-measured sensitivities is the same budget logic at zero cost. **Cite, do not integrate.**

**E3. LittleBit-2** (arXiv 2603.00042, 9 Feb 2026). Sub-1-bit low-rank binary factorization with latent rotation and Joint-ITQ, then QAT; tested on Llama-2 and Llama-3 only, no MoE (V; the QAT step is from a secondary summary, Verify [S28]). Expert matrices are 2304 × 1152, with little low-rank slack (Est.); QAT on a 26B looped MoE with no master is far beyond 200 GPU-h. **No.**

**E4. LoopQuant, Loopy, LoopQ.** Their problem is W4A4 activations, which Inkling solves by training. Take two ideas at zero cost: select at deployment depth (allocate at P = 3, test transfer to P = 5) and the pairwise interaction term.

**E5. Feedback-Exposure Hessian summing.** llama.cpp already does the diagonal version (A2.1). Full GPTQ with summed Hessians needs PyTorch and cannot be exported to K-quant GGUF without a projection step. About 2 weeks and 5 to 15 GPU-h (Est.). Stretch after G2, only if the PR is merged by mid-January.

**E6. QTIP, trellis, VQ.** No kernels in llama.cpp or MLX. EXL3 will be a free baseline later. **No.**

**E7. Worth adding.** The per-visit untied probe (1 week, under 10 GPU-h). Dory1.0-4B as a second instrument (1 to 2 weeks; 4B fits the 3080 Ti in BF16; the transformers loop argument is undocumented, Verify).

**E8. Recommendation.** Stay narrow. Add E7, nothing else. The asset is a clean measurement on a new architecture, not a better quantizer.

## F. What else

**F1.** Reference and test arms must share the runtime, the non-expert Q8_0 tensors and P. Add a second KLD against native P = 3 as a common scale across P.
**F2.** Bootstrap over chunks, paired across conditions. llama.cpp's ± is a token-level standard error and understates.
**F3.** Publish the run-to-run floor; use it as the H5 margin.
**F4.** One primary contrast per hypothesis, Holm across five; the rest exploratory.
**F5.** Matched size: exclude mmproj and DSpark; report expert bpw and file bytes; match within 0.5%.
**F6.** Split traces by prompt into allocation and test sets; cluster by trace.
**F7.** AUP (updated 15 Jul 2026): covers "any modified version"; you are responsible for downstream users you serve; no benchmarking or publication limits; it can change at any time (V, [S12]). Archive the version with each release.
**F8.** Mirror the pre-registration publicly before any G0 data, with corrections 0.1 to 0.6 as dated changelog entries.
**F9.** GCP G4 spot: I found $0.45 to $2.34/h depending on page and region, not $1.58 (Verify, [S33]).
**F10.** Keep DSpark off; SGLang's shared-cache mode changes accuracy (90.0 against 88.5 on GSM8K, V [S13]), so log KV mode.
**F11.** Imatrix coverage: 128 experts, top-8; check per-expert counts before trusting IQ2 on rarely routed experts.

## G. Ranked top 7 changes

1. **Replace the P-sweep as primary H1 evidence with per-visit untied quantization on Mini at P = 3; keep the varloop sweep secondary; analyse hidden-state error, not KLD shape.** Cost: 1 week, under 10 GPU-h. Affects H1, H3, G1.
2. **Fix the reference on day 1: one runtime, a reference that fits 48 GB, measured run-to-run and Q8_0 floors; drop "Unsloth ± 15%".** Cost: 2 to 3 days, about 5 GPU-h. Affects G0, all hypotheses.
3. **Redefine H3 and G2 as an internal contrast (same quantizer and imatrix, only visit weighting differs), on held-out data, with six extra quant levels and a check that the allocations differ.** Cost: 3 days, Mac time only. Affects H3, G2.
4. **Cut H2 to a one-day pre-check; replace it with visit-weighted pooling and calibration-depth transfer if the pre-check passes.** Cost: saves 1 to 2 weeks. Affects H2.
5. **Plan disk and logits: 32 chunks or top-k logits, build-evaluate-delete.** Cost: 1 day. Affects G0, G1.
6. **Shrink tier 1 to two builds per budget on full GSM8K plus one code set, cluster bootstrap by item; G3 becomes "no significant reversal"; register CoT length against P as an outcome.** Cost: saves 20 to 40 GPU-h. Affects G3, H3.
7. **Add Dory1.0-4B as a second topology with a BF16 master; reject GSQ, RCO, LittleBit-2 and QTIP integration.** Cost: 1 to 2 weeks, local hardware. Affects H1, novelty, scoop risk.

## Sources

- [S1] TML, "Introducing Inkling 2 Mini", 20 Nov 2026: https://thinkingmachines.ai/news/inkling-2-mini/
- [S2] varloop model card, 20 Nov 2026: https://huggingface.co/thinkingmachines/TML-d24-varloop-research
- [S3] ggml-org varloop GGUF card, 20 Nov 2026: https://huggingface.co/ggml-org/TML-d24-varloop-research-GGUF
- [S4] ggml-org Mini GGUF card, 20 Nov 2026: https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF
- [S5] llama.cpp PR #32694, merged 20 Nov 2026: https://github.com/ggml-org/llama.cpp/pull/32694
- [S6] llama.cpp PR #32695, draft, 20 Nov 2026: https://github.com/ggml-org/llama.cpp/pull/32695 ; stacked scales PR: https://github.com/danielhanchen/llama.cpp/pull/10
- [S7] llama.cpp PR #32217, merged 14 Nov 2026: https://github.com/ggml-org/llama.cpp/pull/32217
- [S8] Unsloth guide and repo, fetched 21 Nov 2026: https://unsloth.ai/docs/models/inkling-2-mini ; https://huggingface.co/unsloth/Inkling-2-Mini-GGUF
- [S9] transformers PR #51050, open, 20 Nov 2026: https://github.com/huggingface/transformers/pull/51050
- [S10] mlx-lm PR #2086, draft, last activity 20 Nov 2026: https://github.com/ml-explore/mlx-lm/pull/2086
- [S11] TML, "End-to-End 4-Bit Training", 13 Nov 2026: https://thinkingmachines.ai/blog/end-to-end-4-bit-training/
- [S12] Model AUP, updated 15 Jul 2026: https://thinkingmachines.ai/model-acceptable-use-policy/
- [S13] SGLang cookbook, fetched 21 Nov 2026: https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini
- [S14] llama.cpp master, fetched 21 Nov 2026: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/imatrix/imatrix.cpp ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/perplexity.cpp
- [S15] LoopQ, 8 May 2026 (v2 12 Oct): https://arxiv.org/abs/2605.16343
- [S16] Loopy, 4 Oct 2026: https://arxiv.org/abs/2610.05265
- [S17] LoopQuant, 12 Oct 2026: https://arxiv.org/abs/2610.16368
- [S18] Feedback Exposure, 25 Sep 2026: https://arxiv.org/abs/2609.30820
- [S19] Tilted Bowl, 30 Sep 2026: https://arxiv.org/abs/2609.39277
- [S20] GSQ, 20 Apr 2026 (v3 10 Nov): https://arxiv.org/abs/2604.18556
- [S21] GSQ code, fetched 21 Nov 2026: https://github.com/IST-DASLab/GSQ
- [S22] RCO, 1 May 2026: https://arxiv.org/abs/2605.00649
- [S23] ISTA-DASLab GSQ-RCO GGUF card: https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
- [S24] MoESQ, 29 Sep 2026: https://arxiv.org/abs/2610.02241
- [S25] MODE, 15 Jun 2026: https://arxiv.org/abs/2606.17118 ; VSRAQ, 4 Jun 2026: https://arxiv.org/abs/2606.05688
- [S26] LOOM, 1 Oct 2026: https://arxiv.org/abs/2610.01153 ; KV: https://arxiv.org/abs/2610.10381 (7 Oct), https://arxiv.org/abs/2611.05304 (3 Nov), https://arxiv.org/abs/2611.04137 (3 Nov)
- [S27] Dory1.0-4B, paper 2 Nov 2026, weights 16 Oct 2026: https://arxiv.org/abs/2611.03998 ; https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
- [S28] LittleBit-2, 9 Feb 2026 (v2 3 May): https://arxiv.org/abs/2603.00042
- [S29] Matryoshka Quantization, 10 Feb 2025: https://arxiv.org/abs/2502.06786
- [S30] Lotfi et al., 29 May 2026: https://arxiv.org/abs/2606.00206
- [S31] HF derivative listing, 21 Nov 2026: https://huggingface.co/models?other=base_model:quantized:thinkingmachines/Inkling-2-Mini&sort=created ; https://huggingface.co/hermitdave/Inkling-2-Mini-MLX-RCO-3.5bpw ; https://huggingface.co/netprojects/Inkling-2-Mini-MLX-2.5bit ; https://huggingface.co/araysuter/Inkling-2-Mini-MLX-mixed-2bit
- [S32] weft-loom issue #39, 20 Nov 2026: https://github.com/Moissanite-Labs/weft-loom/issues/39
- [S33] GCP pricing, fetched 21 Nov 2026: https://cloud.google.com/products/compute/pricing/accelerator-optimized ; https://gcloud-compute.com/g4-standard-48.html
