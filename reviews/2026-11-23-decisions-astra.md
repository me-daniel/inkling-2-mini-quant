# Review of the release decisions (v2m)

- Date: 2026-11-23, ~07:30
- Model: GPT-6.2 Astra, effort high (Codex CLI, isolated)
- Repo HEAD: 18cd480

## Prompt

You are a skeptical reviewer of release decisions for a model-quantization project. You did not do the work. Answer in under 500 words. For each numbered decision give: your recommendation (yes / no / change it, and how), the main risk, and one thing the author must check first. Then add at most 3 points the decision list is missing. No praise.

## Setup (public facts)

- Model: Inkling 2 Mini, a looped MoE (layers 6-17 run three times per token on the same experts; 128 experts, top-8). Quantized to llama.cpp GGUF for Macs.
- v1 is public: 11.51 GB, every tensor runs on the Mac GPU (Metal) with stock llama.cpp, 1.75x faster generation than Unsloth's UD-IQ3_XXS (11.66 GB), which keeps some NVFP4 expert tensors on the CPU. v1 failed a pre-registered quality test against UD-IQ3_XXS and says so.
- Metric: KL divergence of next-token distributions against a Q8_0 reference, paired bootstrap over chunks. Sets: S (16 x 4096 WikiText-2 test tokens, used for selection), T1 (16 x 4096 WikiText-2 test, disjoint from S), T2 (5 x 2048 reasoning traces the model generated on GSM8K train prompts). All three have been looked at. T3 is a sealed WikiText-2 test span, registered but never opened.
- v2m (11.57 GB, same tensor types and byte sizes as a llama-quantize build "v2a" at UD's per-layer types): gate_up expert codes re-chosen by a codes-only search that minimises tr(E H E^T), E = quantized minus BF16 weights, H = per-expert input Gram + 0.1 x its diagonal, block scales fixed. Experts with fewer than 2304 Gram tokens are left unchanged; a per-row output-gain floor forbids shrinking any row's output gain below llama-quantize's. Grams are captured on a fixed reference build's forward pass (v1, not v2m itself) over 153,600 tokens of mixed calibration text: WikiText-2 train, chat/tool-use conversations rendered with the model's chat template, and general prose. Zero shared 13-grams with S, T1, T2, T3. Prior art for the search: MACaQ (arXiv 2611.10649, Appendix C), dense models only.
- History: v2r, the same search on WikiText-only Grams, closed 93% of the v2a-to-UD gap on S, 74% on T1 and 18% on T2 (domain overfit). v2m was then registered (rule written before data) with the mixed calibration: 95% (S), 90% (T1), 70% (T2).
- v2m vs v1: T1 KL 0.0199 vs 0.0269, diff -0.0069 [-0.0081, -0.0058]; T2 0.0153 vs 0.0194, diff -0.0041 [-0.0060, -0.0026]; top-1 agreement higher on both. Size +0.46%. Speed: not benched; same tensor types as a sibling build that benched equal to v1.
- v2m vs UD-IQ3_XXS: S and T1 no difference resolved; T2 worse, +0.0017 [+0.0004, +0.0030]. A diagnostic "mirror" (UD with its NVFP4 down projections replaced by our Metal-runnable IQ4_NL) loses 0.0008-0.0012 KL to UD on every set; v2m vs mirror: no difference resolved on any set.
- A second, smaller file family (8.5-8.9 GB, for 16 GB Macs) in a sibling project will reuse the same search on 7 of its 24 layers next; most of its layers use IQ2_XXS, a type the search does not support yet.

## The decisions

1. Open T3 (one-shot, sealed WikiText-2 test) for v2m. Proposed: only after decision 3 fixes the bar.
2. The author reviews the search code (551 lines, 12 tests) before any release use. Open judgement calls: is the output-gain floor (never below llama-quantize's per-row gain; motivated by the finding that output shrinkage compounds over the loop's repeated visits) the right rule or too strict; is it acceptable that Grams come from v1's forward and not the refined model's; shrink 0.1 and the 2304-token minimum were set in a pilot and never tuned.
3. Release bar for a "v1.1": v2m beats v1 on T3 (paired 95% CI below zero, top-1 not lower), the UD comparison reported alongside. Proposed additions: a fresh, never-touched text domain (e.g. code or non-English prose) scored by KL for v1 and v2m, and small task checks (e.g. GSM8K accuracy, possibly code, structured output, tool calling) for v1 vs v2m, all locally on one Mac (~19 tokens/s generation).
4. Optional: extend the search to IQ2_XXS (about half a day of work plus CPU hours) mainly for the smaller sibling file; proposed only after its 7-layer run shows an effect.

## Reply

1. **Yes. Open T3 once, after the code review and release rule are final.** Check first that the registration identifies the exact v2m file, T3 span, scoring script, and pass/fail rule, including what happens if v2m fails. The main risk is consuming the only sealed WikiText result on a build or rule the author later changes. Report the result either way.

2. **Yes. Review the code before release use.** Check first that an independent GGUF decoder reproduces the baseline and refined row gains and objective values on sampled emitted blocks, including low-count experts that should be unchanged. The main risk is that a lower local error on v1 activations hides a regression on later loop visits under v2m. Keep 0.1 and 2304 fixed for this candidate. On calibration text, compare activations by loop visit under v1 and v2m, and measure how often the floor binds and how much objective improvement it costs. Those diagnostics can inform a later candidate without spending T3.

3. **Change it. Keep the proposed T3 gate and give the additional checks explicit roles.** Check first that the paired bootstrap resamples chunks and that “top-1 not lower” specifies whether it refers to a point estimate or a confidence bound. The main risk is turning a WikiText win and a few inconclusive task scores into a general quality claim. Before scoring, choose one fresh domain and a maximum acceptable KL regression versus v1, with a rule for an inconclusive interval. At 19 tokens/s, run one bounded, paired task check, such as GSM8K on held-out prompts with fixed decoding, token cap, and scoring. Report paired uncertainty. Keep the UD comparison visible, including the T2 deficit. The mirror is diagnostic evidence, and an unresolved difference does not establish parity.

4. **Change it. Run the seven supported layers first, but use that result as one input to the IQ2_XXS decision.** Check first how much baseline error those seven layers account for on the smaller model. The main risk is treating a small whole-model change as evidence about a different tensor type. If IQ2_XXS layers dominate the error, a one-layer feasibility trial can be justified even after a flat seven-layer result.

**Missing from the list**

- Benchmark the shipped v2m file against v1 on the same Mac and stock llama.cpp revision. Verify full Metal offload, generation speed, and peak memory before carrying forward v1’s performance claims.
- Report T2 uncertainty at the trace level. Five generated traces provide limited evidence about reasoning broadly, regardless of how many tokens they contain.
