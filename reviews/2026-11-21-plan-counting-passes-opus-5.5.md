# Review: plan, Counting Passes design

- Date: 2026-11-21
- Checkpoint: plan (extended: hypotheses, prior art, scoop forecast, novelty, feasibility, integrations)
- Model: Opus 5.5, effort xhigh, Claude Code subagent with WebSearch/WebFetch
- Input: redacted design brief from `private/counting-passes-claude-code-handoff.md` (sections 3-8, no personal details), repo HEAD 3c8947f (no design files tracked yet)
- Prompt: shared review prompt, saved as `reviews/2026-11-21-plan-prompt.md`

## Reply (verbatim)

# Review of "Counting Passes" (21 Nov 2026)

Labels: V = vendor/authors about their own work, I = independent (third-party code, PR pages, source files), Est. = my estimate, Verify = not confirmed. [Sn] = source list at the end (URL + date). Everything was checked live on 21 Nov 2026.

## 0. Fact corrections that change the plan

0.1 **Wrong llama.cpp PR and build.** PR #32217 (merged 14 Nov) adds the non-looped `inkling2` arch, and its commit still says "Real checkpoint conversion remains untested" (I) [S9]. The looped `inkling2_loop` graph is PR #32694, merged 20 Nov as 29b52d3 (I) [S10]. It first ships in **b12600** (20 Nov, 23:30) (I) [S12]. So "≥ b12406" will not load Mini.

0.2 **The pass count is already settable.** The command is `--override-kv inkling2_loop.loop.count=int:N`, valid N is 1–6, and the count is fixed at load (V) [S14]. In HF, varloop's config field is `loop_config.num_passes` with `is_varloop: true`, not `text_config.loop_num_passes` (V) [S4]. The transformers kwarg `loop_num_passes` comes from the HF blog example (I) [S8].

0.3 **Unsloth is not on mainline and not on WikiText.** The UD files need Unsloth's fork `b12588-mix-c0c3de8`, built for "looped-layer support and per-expert NVFP4 scales" (V) [S16]. A commenter on #32694 reports the UD files fail to load on mainline (block_count 48 vs 24 physical blocks) (I) [S10][S11]. Unsloth's KLD is measured "across seven domains", which are undisclosed (V) [S16]. So "64×4096 WikiText-2 (Unsloth-comparable)" is false.

0.4 **ggml-org did not ship ladders.** It published BF16 (52.2 GB), Q8_0 and IQ4_NL only (V) [S13][S14]. **The Q8_0 floor is large:** Q8_0 vs BF16 gives KLD 0.0030 and 97.6% same top-1 (WikiText, 40×512) (V) [S13]. AtomicChat measured 0.0033 and 97.7% on its own corpus (V) [S18].

0.5 **The format mismatch on ffn_down breaks the level design.** In `ffn_down_exps` the row size is ne0 = 1152 = 4.5×256 (I, shapes in the source) [S9]. When ncols % block ≠ 0, llama-quant falls back: IQ2_XXS and IQ3_XXS become IQ4_NL, and Q4_K becomes Q5_0. This happens **even under a `--tensor-type` override** (I) [S20]. One third of every "IQ2_XXS block" is therefore 4.5 bpw.

0.6 **imatrix already pools visits.** Every MUL_MAT/MUL_MAT_ID that reads a weight adds to the same `m_stats[wname]`, and counts accumulate. There is no "already seen" check (I) [S21].

0.7 **Disk.** A `--kl-divergence-base` file stores the full vocabulary as uint16 for every scored token. Only the second half of each window is scored (I) [S22]. With a 201,024 vocab, 64×4096 tokens is ≈52.7 GB per reference per dataset (Est.). Five varloop P values plus Mini, times two datasets, is ≈630 GB, against ~690 GB free.

0.8 **The BF16 GGUF (52.2 GB) does not fit in 48 GB of unified memory** (V size [S13], Est.).

0.9 **Architecture.** Attention residuals (depth-wise mixing, every layer) are separate from the cross-pass message controllers (`tanh(gain)`, `tanh(forget)` on Q/K/V and the FFN input) (I, code) [S9]. Passes restart from the prelude, and the message gate starts closed (V) [S6]. MTP reads visit indices 5/23/41/48 (V) [S3], so it is tied to P = 3 (Est.). The loop is half-open, 6–17 (V) [S2][S6].

0.10 **The "last-sixth rule" is not a sensitivity claim.** TML says 16.6% because NVIDIA's recipe uses 15% and "our pretraining ladder scales layer count by 6" (V) [S7]. 4/6 scale selection applies to activations, not weights (V) [S7].

0.11 **The Tilted Bowl citation is half wrong.** Ouro-1.4B 73% → 7% GSM8K is at w4c (4-bit per-channel), and PPL 11.51 → 14.38 is not in the paper (Verify). Ouro "does not settle" like Huginn, and INT8 final loops on Ouro-1.4B gave +0.2 (p = 1.0) (V) [S27].

0.12 **transformers is less blocked than stated.** PR #51050 is open, but the lm_head bug hits only float32 loads and a maintainer called it non-blocking (I) [S23]. In practice: PR branch plus a ≥80 GB GPU.

0.13 **Varloop "may behave unpredictably at pass counts other than 3"**, and its benchmarks are non-monotone: GPQA 84.2 at P3 vs 81.9 at P4 (V) [S2].

0.14 **The config conflict is resolved:** the same modelopt vs `inkling2` clash broke SGLang on Inkling-2-Small, and `hf_quant_config.json` is authoritative (I) [S26].

## A. Hypotheses

**H1 (count once / per visit / compound)**

1. **It is not identifiable from KLD curve shape.** KLD is second order in the logit error, KLD ≈ ½·eᵀFe. Suppose per-visit errors e_k are additive but correlated. That is likely here: passes restart from the same prelude state, and cross-loop states are "highly similar" (V) [S33][S34]. Then KLD grows roughly as P² with no compounding at all. So "superlinear means compounding" is wrong. Define the three outcomes in logit (or final hidden-state) space instead:
   - "once": e_all ≈ e_last;
   - "per visit": e_all ≈ Σe_k;
   - "compound": ‖e_all‖ > ‖Σe_k‖.
2. **P ≠ 3 is off-distribution.** P = 1/2/4/5 had 1/12–1/6 of looped midtraining each and no post-training (V) [S2]. A sensitivity that rises at P ≠ 3 can come from off-distribution brittleness, not visit counting.
3. **The reference changes with P.** Reference entropy changes with it, and KLD scales with it. Report KLD/H(ref), and run difference-in-differences within each P against a non-looped block.
4. **The controls are unbalanced.** B4 is 2 layers. B5 is 4 layers of native MXFP8, i.e. a different starting precision. B2/B3 are 6 NVFP4 layers each. Use **B1** as the matched control (6 layers, NVFP4). The design actually has three cells:
   - B1: computed once, consumed P times (restart);
   - B2/B3: computed P times;
   - B4: computed once, consumed once.

   State this explicitly.
5. **Drop "tests Tilted Bowl".** Their theory concerns loops that settle. A 3-pass restart loop is not a fixed-point iteration. Reframe B1 as "loop gain on prelude error".
6. **Falsifier (pre-register).** Quantize B2 in pass k only, at P = 3. If ‖e_all − Σe_k‖/‖e_all‖ < 0.1 and the ΔKLD(B2)/ΔKLD(B1) ratio does not vary across P ∈ {2,3,4}, then the error is additive per visit and there is no compounding.
7. **A cheaper and cleaner design: a per-pass causal ablation on Mini at P = 3** (on-distribution, RL'd). Use 7 pass subsets × {B1, B2, B3} × 2 levels ≈ 42 runs. Add a **message ablation**: quantize pass k but feed pass k+1 the native message. That splits error injection from cross-pass transmission. Read `tanh(gain)` and `tanh(forget)` from the weights for free. Keep the varloop sweep as a transfer check at P ∈ {2,3,4}.

**H2 (pooled vs first-visit imatrix)**

8. **Pooled is already the llama.cpp default** [S21]. "First visit only" needs a patch and is a degraded variant. The Hessian-summing result already exists on 9 looped checkpoints (V) [S26b].
9. **Confound: expert coverage.** Routing differs by pass, so a first-visit imatrix leaves some experts with zero counts. Any gain would come from coverage, not from visit weighting.
10. **Cheaper test (1 day):** compute the cosine similarity of per-visit imatrix vectors for each expert. If the median is > 0.99, drop H2.
11. **Better replacements:** a visit-weighted imatrix (weights from the item-7 sensitivities) vs pooled, or the calibration source (self-traces vs bartowski v6) vs pooled. RECAP shows calibration history matters for reasoning models (V) [S36].

**H3 (allocator beats Unsloth at matched size)**

12. **The ordering is partly tautological.** An allocator built on measured end-to-end ΔKLD includes visits by construction. The question worth asking is whether a free **×visits correction** to the proxy closes most of the gap between visit-blind and measured allocation.
13. **Unsloth is a confounded comparator.** It differs in imatrix data (undisclosed), non-expert recipe, fork build and eval data. Make the primary contrast **same pipeline, visit-corrected vs visit-blind**. Copy Unsloth's non-expert tensor types (`gguf-dump` shows them) so that only expert allocation differs. Keep Unsloth, bartowski and GSQ-RCO as external references.
14. **Feasibility at 9.14 GB.** Experts are ≈24.5B params (Est. from shapes; 92.6–94% per [S19][S18]). B5 alone is ≈4.1B params, ≈4.1 GB at 8 bits, or 21% of the native file (Est.). With non-experts at Q8_0 (≈1.6 GB), experts must average ≈2.45 bpw. With the down projection forced to IQ4_NL by fallback, IQ2_XXS gate_up gives ≈2.9 bpw and ≈10.4 GB in total, which does not fit (Est.). You need block-32 down types (Q4_0, IQ4_NL, Q2_0; Q2_0 block size Verify), and B5 must go well below 4 bits.
15. **~10 pairwise configs is too few** to test additivity across 5 blocks × 4 levels. Evaluate every greedy step end-to-end instead, and prune with Loopy-style successive halving over chunks (V) [S25].
16. **Falsifier:** at both budgets, the paired CI of ΔKLD for the visit-corrected allocator overlaps the visit-blind one.

**H4 (router Jaccard falls with pass)**

17. **It is near-trivially true.** With only B2 quantized, the layer-6 router input in pass 1 is error-free (restart), so Jaccard starts at 1 and falls. Make H4 a **mediation test**: route-lock to the reference top-8 and measure the fraction of ΔKLD removed (falsifier: < 20%). Use mass-weighted overlap, since near-ties in a top-8 of 128 flip cheaply.

**H5 (B5 tolerates ~4 bits)**

18. **Define "negligible"**, e.g. below the Q8_0 floor (0.003) or below the CI. Confounds:
    - B5 goes from 8 to 4 bits, a larger step than NVFP4 to Q4_K;
    - down falls back from Q4_K to Q5_0;
    - the vendor made no inference claim (0.10).

    H5 matters mainly as a **budget input to H3**. Falsifier: B5@Q4 has ΔKLD per GB saved ≥ that of B4@Q4.

**Cross-cutting**

19. **Activations.** Training used per-token NVFP4 activations with `fouroversix_mae` (V) [S5]; llama.cpp runs float, so the "native" reference is not the trained numerics. Measure it once by fake-quantizing activations in the transformers or MLX reference on 8 chunks, and flag it if the KLD exceeds your smallest effect.
20. **No master weights.** The weights are RTN NVFP4 (`nvfp4_nearest`) from a high-precision master that TML did not release (V, inferred from their figure) [S7]. Q4_K "new grid" is double rounding, so grid change is confounded with re-rounding. Make it 2×2: {native E2M1 grid, new grid} × {4 bits, ≤3 bits}. The nested level fills the "native grid, ≤3 bits" cell.
21. **Data.** WikiText is off-domain for an RL'd reasoning model. Self-traces are on-policy only at P = 3. Keep the imatrix data disjoint from the eval data. ggml-org's imatrix used bartowski v6 (V) [S13], so do not evaluate on v6.

## B. Prior art and scoop risk

22. **Looped PTQ.** All of it is dense, with no MoE and no sub-4-bit weight-only work on FP4-native checkpoints:
    - LoopQ, W4A4 (V) [S24]; Loopy, choose configs at the deployment depth (V) [S25]; LoopQuant's ILPA, a loop-wise *activation* allocation (V) [S26].
    - Feedback Exposure, 9 checkpoints: the summed Hessian closes 17–102% of the gap, and Huginn INT3 goes 5.84 → 20.70 (V) [S26b].
    - Tilted Bowl, 33 models at w2–w4 (V) [S27]; Hyperloop (V) [S28].
    - KV cache: ResidualQuant and LoopVQ (V) [S33][S34].
23. **MoE allocation:** MODE, expert ILP at W3A16 and 2-bit (V) [S29]; VSRAQ, routing consistency, with no bit widths in the abstract, so "4-bit only" needs Verify [S30]; RCO, exact budget, non-decomposable loss, HF-based (V) [S31][S32]; GSQ, 2–3 bits with GGUF K-quant refinement (V) [S27b]. GGUF cannot express per-expert types, so MODE-style per-expert allocation does not deploy (V) [S35].
24. **Inkling-specific work.** I found no arXiv paper as of 21 Nov. Within 24 h the Hub already had 26 Mini repos (I) [S17], including:
    - AtomicChat's ladder with KLD vs BF16. It explicitly guesses that "rounding accumulates across these passes… we have not tested it" (V) [S18].
    - hermitdave's MLX "RCO" 3.53 bpw: KL 0.0669 vs native on WikiText (V) [S19].
    - araysuter's mixed 2-bit build (V) [S20b].

    The community is primed for H1.
25. **Forecast for the next 2 days (Est.):** a bartowski imatrix ladder on b12600 (70%; he did Inkling-2-Small, search listing, Verify); Unsloth re-uploads (90%) and varloop UD GGUFs (40%); more MLX 2/3-bit builds (80%); a #32695 compatibility or per-expert-scale merge (60%). A post with llama-perplexity KLD at P = 1–6 for varloop IQ4_NL is 30% likely now and 60% by 6 weeks. That is the cheapest scoop of H1's headline curve.
26. **Forecast for the next 2–6 weeks (Est.):**
    - **IST-DASLab GSQ-RCO GGUF of Mini (35%, gated on the transformers merge).** They already publish GSQ-RCO vs UD at matched size for Qwen3.8-27B (V) [S35]. That is exactly H3, and RCO optimises the real loss through the looped forward, so it is implicitly visit-aware.
    - Baselines only: AutoRound mixed-bit/GGUF (50% after the merge); Red Hat W4A16/FP8 (40%); EXL3 (40%; stock exllamav3 lacks Inkling-2, Verify). ModelOpt is already native.
    - Loop-PTQ groups adding Mini in a revision (15–25%). Likely W4A4, which is native here, so low overlap. The biggest H2 risk is per-step GPTQ from Feedback Exposure's author.
27. **Overlap summary.** High: the H1 aggregate P-curve; H3 vs UD (GSQ-RCO). Medium: H2 (already in the literature). Low: H4, H5, the per-pass causal ablation, message and route-lock mediation, and the visit-count proxy. Everything else is baselines.

## C. Novelty

28. **Not novel:** loop-aware calibration, depth-specific config choice, loop-wise precision allocation, expert ILP, routing consistency, measured allocators that beat UD at matched size, nested quantization (Matryoshka, Any-Precision) [S37][S38], NVFP4 scale-bit reuse (MixFP4) [S39].
29. **Novel:** (i) a per-pass causal decomposition of weight-quantization error in a looped MoE; (ii) a restart-from-prelude loop with low-rank messages, a channel unlike the iterative loops studied so far; (iii) requantizing an FP4-native model with no released master; (iv) visit scaling on a production-scale variable-pass checkpoint; (v) deployable GGUFs.
30. **Contribution sentence:** "In Inkling 2 Mini, a restart-from-prelude looped MoE trained natively in FP4, sub-4-bit error in shared looped experts [adds per pass / compounds] in logit space, travels between passes mainly via [message / routing], and a one-line visit-count correction to imatrix allocation recovers X% of the gap to measured allocation at matched GGUF size."
31. **Workshop bar:** a clean per-pass result with CIs; one transfer check (varloop P ∈ {2,4}, or a dense looped model; Ouro in llama.cpp, Verify); task confirmation on ≥2 tasks; released per-pass patch and allocator. Target an ICLR/ICML 2027 efficient-ML workshop (dates Verify).
32. **The nested E2M1 level is a diagnostic, not a contribution.** A 2-bit subset with E4M3/16 scales costs 2.5 bpw, more than IQ2_XXS at ≈2.06, and has no kernel.

## D. Getting it done

33. **Fork split.** Day-1 test: run UD-Q8_K_XL, UD-Q2_K_XL, ggml-org Q8_0 and IQ4_NL (4 chunks each) on mainline b12600 and on Unsloth's fork. Pivot: one pinned build for everything. Fallback: re-measure Unsloth yourself and ignore their published numbers.
34. **Reference.** Test whether varloop converts with NVFP4 experts preserved (fork, per-expert scales [S15]). If not, use a Q8_0 reference with its ≈0.003 floor. Fallback: BF16 reference logits on a G4 for 32 chunks.
35. **Pass override.** Test P = 1–6 on 8 chunks with MTP off, and check KV memory scales with P. If the logits do not change, patch (≤2 days) or go Mini-only; item 7 needs no override.
36. **ffn_down fallback.** Check `n_fallback` in the llama-quantize log and the tensor types with `gguf-dump`, then fix the levels (item 14).
37. **Disk.** Use 32 WikiText + 32 trace chunks per P (≈53 GB). Run one P at a time and delete as you go. Block GGUFs are ≈20–25 GB each (Est.), so generate, evaluate, delete. Better: quantize with llama-quantize, dequantize with gguf-py (IQ support Verify) and inject into a Python graph, which needs no file per configuration.
38. **Per-pass instrument.** Target: transformers PR branch on a G4 (≈10–20 GPU-h, Est.) or the MLX fork (0 GPU-h). Gate: within 1 week the instrument must reproduce llama.cpp reference logits to KLD < 0.005. Pivot: a llama.cpp C++ patch that loads a second weight set for pass k. Kill: H1 falls back to the varloop sweep with difference-in-differences.
39. **Power.**
    - G1's "between-block ΔKLD > CI width" will pass trivially for B2 vs B4. Replace it with "the H1 per-pass contrast CI excludes the 'once' prediction".
    - Tier-1 MDE at 80% power (Est.): GSM8K-500 ≈3.5–5 points, IFEval ≈4–5, MMLU-Pro-1K ≈3, LCB-150 ≈10–12.
    - So G3 "ordering agrees" will fail by noise for close builds. Pre-register only the 2 primary contrasts.
40. **Tier-1 engine.** inkling2 on CUDA hit graph-reserve memory blowups at large `-ub` on a 5090 (I) [S10], and vLLM lists H100/H200/GB10, not SM120 (V) [S40]. Run GSM8K-50 on G4 spot ($1.36–3.46/h, avg $2.24; I, 18 Nov [S41]) before booking hours. Compute: 7 builds × 2,191 items × 3 seeds ≈ 46k generations, ≈10–50 GPU-h (Est.).
41. **Scope.** Five hypotheses, an allocator, the nested level, tier 1 and a public note are too much for one person in 24 weeks with writing. Cut H2, H4 and H5 to secondary.
42. **Minimum viable thesis:** the per-pass and per-block ΔKLD map with logit additivity on Mini at P = 3, plus one visit-corrected vs visit-blind allocation at 11.66 GB, confirmed on GSM8K + IFEval.
43. **Gates.**
    - G0 (6 Dec): fine once the "Unsloth ±15%" criterion goes (the eval data differs). Replace it with reproducing UD ratios: Q2_K_XL/IQ3_XXS = 2.21 and IQ3_XXS/Q4_K_XL = 2.51 on your own data, ±15% (V numbers [S16]).
    - G1 (3 Jan): crosses the holidays; move the note to ~14 Dec using item 7.
    - G2 and G3: realistic if re-targeted per item 13.

## E. Integrations vs de-risking

44. **GSQ.** It adds rounding quality, not visit insight. Cost: 3–5 weeks plus ≈32–80 GPU-h. H100/H200 come only as 8-GPU nodes at $51–53/h, so ≈$200–500 per run (Est.). Blockers: the transformers branch, and K-quants cannot be used on the 1152-wide down projection. **No.**
45. **Other IST-DASLab work.** RCO (Apache-2.0, HF-based) is useful as an external allocator baseline; cost 1–2 weeks if the PR branch works. **Optional.** MoESQ (B200 sparse NVFP4) [S42] and MR-GPTQ are irrelevant on Mac/GGUF. **No.**
46. **LittleBit-2.** Sub-1-bit low-rank binary factorisation, tested only on Llama-2/3 dense, with training cost and kernel unspecified (V) [S43]. This model has 3,072 expert matrices behind loops and there is no GGUF kernel. Months of work. **No.**
47. **Loopy/LoopQuant/LoopQ.** Their W4A4 rotations target activations, which llama.cpp keeps in float. Borrow only Loopy's successive halving (≈1 week, Mac). **Yes.**
48. **Feedback-Exposure Hessian summing.** Already the imatrix default. Swap in a visit-weighted imatrix patch instead (2–3 days). **Yes.**
49. **QTIP/trellis/VQ.** EXL3 will supply this as a community baseline; building it yourself takes months. **No.**
50. **Something else:** gguf-py dequantization injected into a Python graph for the per-pass, route-lock and message ablations (1–1.5 weeks). **Yes.** Overall recommendation: **stay narrow.** The contribution is the measurement, and every method integration is either blocked or off-axis.

## F. Also add or beware of

51. **KLD reference.** Declare one primary: the NVFP4-preserving GGUF at the same P. Measure three floors once: Q8_0 vs native, engine (MLX or transformers vs llama.cpp) and activation numerics. Keep non-experts at BF16 or Unsloth-copied types, not Q8_0 by default; the Q8_0 floor is roughly ⅓ of UD-Q4_K_XL's KLD (Est.; different data and references).
52. **Statistics.** Paired bootstrap over identical chunks. Report mean, median, 99% and 99.9% KLD and same-top-p, all printed by llama.cpp [S22]. Fit one pre-registered model over the ~100 tier-0 cells, not 100 tests. In tier 1, McNemar over 3 seeds is ill-defined: use per-item mean correctness with a paired bootstrap or a mixed logistic model, then Holm.
53. **Matched size.** Count only the text model: exclude mmproj (0.64 GB) and MTP/DSpark. Report expert bpw separately. Include the E4M3 and FP32 scales in the nested-level bpw.
54. **AUP.** It covers "any modified version", makes you responsible for anyone you let use the model, and bans circumventing safety "measures" (V) [S44]. Ship the AUP notice with released GGUFs, never use the "Uncensored" derivatives [S17], and add a cheap 100-prompt refusal check, since quantization can erode refusals (Est.).
55. **Reproducibility.** Pin HF revisions (repos changed within hours [S17]), both llama.cpp commits and GGUF sha256 hashes. Log `--override-kv`, the effort plumbing (`--chat-template-kwargs`, Verify), slot count and seeds. Continuous batching makes T = 1 nondeterministic.
56. **Compute traps:** the macOS wired-memory limit (≈36 GB default, Est.); P = 5 costs ≈1.4× P = 3 per token (7.4B vs 5.4B active); KV grows with P; spot preemption mid-generation. Tier-0 wall-clock is ≈10–25 min per run on the M5 Pro, ≈17–42 h in total (Est.).
57. **Missed by the brief:** the model is multimodal (state text-only); raw WikiText vs chat-templated traces; fixed KV type and flash-attention setting across runs. The HF blog says mlx-lm support was upstreamed, but PR #2086 is still a draft (I) [S8][S24b].

## G. Top 7 changes, ranked

1. **Make H1 a per-pass causal ablation on Mini at P = 3, measured in logit space.** Add message and route-lock ablations, and use the varloop sweep P ∈ {2,3,4} as transfer. Cost: 1–1.5 weeks plus 0–20 GPU-h. Affects H1, H4, G1.
2. **Fix the ffn_down level set.** At 1152 width, IQ types fall back to IQ4_NL and Q4_K to Q5_0. Use block-32 down types, recompute budget feasibility, and log the actual bpw. Cost: 1–2 days. Affects H1, H3, H5, G0, G2.
3. **Fix the reference.** Use an NVFP4-preserving GGUF at the same P. Measure the Q8_0, engine and activation floors once. Keep non-experts at BF16 or Unsloth-copied types. Cost: 2–4 days. Affects all hypotheses and G0.
4. **Amend G0 now, before any data.** Require build ≥ b12600 plus the pinned Unsloth fork. Set P via `inkling2_loop.loop.count`. Replace "UD ±15% on WikiText" with ratio reproduction on your own data. Add the disk plan. Cost: 0.5 day plus a changelog entry. Affects G0.
5. **Re-target H3.** Make the primary contrast visit-corrected vs visit-blind in one pipeline (same imatrix, same non-expert types). Keep UD, bartowski and GSQ-RCO as references, and use successive halving. Cost: ≈1 week. Affects H3, G2.
6. **Cut H2, H4 and H5 to secondary.** Replace H2 with a visit-weighted imatrix (or calibration source) vs pooled, make H4 a mediation test, and use H5 only as a budget input. Saves ≈2 weeks; the patch costs 2–3 days. Affects H2, H4, H5.
7. **Publish the note by ~14 Dec with the P = 3 per-pass result and the timestamped prereg.** Tighten tier-1 statistics: paired bootstrap or mixed model, Holm, stated MDEs, LCB-150 secondary. Cost: 1–2 days. Affects scoop risk, G1, G3.

## Sources (accessed 21 Nov 2026)

- S1 Mini card, 20 Nov 2026 — https://huggingface.co/thinkingmachines/Inkling-2-Mini
- S2 varloop card, 20 Nov 2026 — https://huggingface.co/thinkingmachines/TML-d24-varloop-research
- S3 Mini config.json — https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/config.json
- S4 varloop config.json — https://huggingface.co/thinkingmachines/TML-d24-varloop-research/blob/main/config.json
- S5 hf_quant_config.json — https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/hf_quant_config.json
- S6 TML announcement, 20 Nov 2026 — https://thinkingmachines.ai/news/inkling-2-mini/
- S7 TML "End-to-End 4-Bit Training", 13 Nov 2026 — https://thinkingmachines.ai/blog/end-to-end-4-bit-training/
- S8 HF blog, 20 Nov 2026 — https://huggingface.co/blog/inkling2mini
- S9 llama.cpp PR #32217 (merged 14 Nov 2026) and `src/models/inkling2.cpp` on master — https://github.com/ggml-org/llama.cpp/pull/32217
- S10 llama.cpp PR #32694 (merged 20 Nov 2026) — https://github.com/ggml-org/llama.cpp/pull/32694
- S11 llama.cpp PR #32695 (draft, 20 Nov 2026) — https://github.com/ggml-org/llama.cpp/pull/32695
- S12 llama.cpp releases (b12600, 20 Nov 2026) — https://github.com/ggml-org/llama.cpp/releases
- S13 ggml-org Mini GGUF — https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF
- S14 ggml-org varloop GGUF — https://huggingface.co/ggml-org/TML-d24-varloop-research-GGUF
- S15 Unsloth fork PR #377 (19–20 Nov 2026) — https://github.com/unslothai/llama.cpp/pull/377
- S16 Unsloth guide — https://unsloth.ai/docs/models/inkling-2-mini ; card https://huggingface.co/unsloth/Inkling-2-Mini-GGUF
- S17 HF search, 21 Nov 2026 — https://huggingface.co/models?search=inkling-2-mini
- S18 AtomicChat GGUF — https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF
- S19 hermitdave MLX-RCO-3.5bpw — https://huggingface.co/hermitdave/Inkling-2-Mini-MLX-RCO-3.5bpw
- S20 llama-quant.cpp (master) — https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp
- S20b araysuter mixed-2bit — https://huggingface.co/araysuter/Inkling-2-Mini-MLX-mixed-2bit
- S21 imatrix.cpp (master) — https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/imatrix.cpp
- S22 perplexity.cpp (master) — https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/perplexity.cpp
- S23 transformers PR #51050 (open, 17–21 Nov 2026) — https://github.com/huggingface/transformers/pull/51050
- S24 LoopQ, v2 12 Oct 2026 — https://arxiv.org/abs/2605.16343
- S24b mlx-lm PR #2086 (draft, 20 Nov 2026) — https://github.com/ml-explore/mlx-lm/pull/2086
- S25 Loopy, v2 17 Oct 2026 — https://arxiv.org/abs/2610.05265
- S26 LoopQuant, 12 Oct 2026 — https://arxiv.org/abs/2610.16368 ; SGLang #50583 (15–19 Nov 2026) — https://github.com/sgl-project/sglang/issues/50583
- S26b Feedback Exposure, 25 Sep 2026 — https://arxiv.org/abs/2609.30820
- S27 Tilted Bowl, 30 Sep 2026 — https://arxiv.org/abs/2609.39277 (HTML tables)
- S27b GSQ, v3 10 Nov 2026 — https://arxiv.org/abs/2604.18556
- S28 Hyperloop, v3 2 Jul 2026 — https://arxiv.org/abs/2604.21254
- S29 MODE, v2 26 Aug 2026 — https://arxiv.org/abs/2606.17118
- S30 VSRAQ, 4 Jun 2026 — https://arxiv.org/abs/2606.05688
- S31 RCO, v3 27 Sep 2026 — https://arxiv.org/abs/2605.00649
- S32 RCO code — https://github.com/IST-DASLab/RCO
- S33 ResidualQuant, v2 29 Oct 2026 — https://arxiv.org/abs/2610.10381
- S34 LoopVQ, 3 Nov 2026 — https://arxiv.org/abs/2611.05304
- S35 ISTA-DASLab GSQ-RCO GGUFs — https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF ; https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF (updated 27 Oct 2026)
- S36 RECAP, 11 Oct 2026 — https://arxiv.org/abs/2610.15259
- S37 Matryoshka Quantization, 2025 — https://arxiv.org/abs/2502.06786
- S38 Any-Precision LLM, 2024 — https://arxiv.org/abs/2402.10517
- S39 MixFP4, 29 May 2026 — https://arxiv.org/abs/2605.31035
- S40 vLLM recipe (~20 Nov 2026) — https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini
- S41 g4-standard-48 prices, updated 18 Nov 2026 — https://gcloud-compute.com/g4-standard-48.html
- S42 MoESQ, 29 Sep 2026 — https://arxiv.org/abs/2610.02241
- S43 LittleBit-2, v2 3 May 2026 — https://arxiv.org/abs/2603.00042
- S44 TML Model AUP, updated 15 Jul 2026 — https://thinkingmachines.ai/model-acceptable-use-policy/
- Also checked: Lotfi et al., 29 May 2026 — https://arxiv.org/abs/2606.00206 ; Looping Beyond Twice, 1 Oct 2026 — https://arxiv.org/abs/2610.01153 ; Four Over Six, v5 9 May 2026 — https://arxiv.org/abs/2512.02010 ; Rectified Looped Transformer, 12 Nov 2026 — https://arxiv.org/abs/2611.16265
