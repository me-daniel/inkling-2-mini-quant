# Review of the release decisions (v2m)

- Date: 2026-11-23, ~07:30
- Model: Claude Fable 5.2, effort medium (subagent, prompt file only, no tools beyond reading it)
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

1. Open T3 — change it: order must be 2, then 3, then 1.
- Recommendation: open T3 only after the code review is finished and the v2m file hash, scoring script, chunking and bootstrap seed are frozen. Any code fix after opening leaves the changed build with no sealed set.
- Main risk: T3 is WikiText-2, the same domain as S, T1 and part of the calibration. v2m already wins there by a wide margin, so T3 will almost certainly pass and says little about the weak spot (T2-like text).
- Check first: T3's chunk count gives the power you expect, and the registered rule states what gets published if it fails.

2. Code review — yes, and it blocks decision 1.
- Gain floor: keep it as shipped. Loosening it now means re-selecting on sets already looked at.
- Grams from v1: acceptable as a fixed, stated reference, but say why not Q8_0/BF16 activations, which are the actual target distribution.
- Shrink 0.1 and the 2304 minimum: do not tune them now. Report them as untuned pilot values.
- Main risk: the file does not match the description (floor binding almost everywhere, write-back errors, many experts skipped).
- Check first: decode v2m's GGUF with llama.cpp's own dequantiser and recompute the objective. It must be no higher than v2a's for every changed expert, and every other tensor must be byte-identical to v2a. Also report how many experts fell under 2304 tokens and how often the floor bound.

3. Release bar — change it.
- "Beats v1 on T3" is a low, in-domain bar. Add the fresh domain with a pre-written pass rule (for example, the CI upper bound of v2m minus v1 below zero), and include chat-template text, since that is real use.
- Task checks: keep them as non-regression smoke tests only. Two builds this close will have few discordant answers; at n=200 the interval is about ±7 points. Use paired McNemar, GSM8K test (train prompts fed T2), fixed seeds and greedy decoding.
- Main risk: "v1.1" reads as a general improvement while T2 remains resolved-worse than UD. That result belongs in the headline, not "alongside".
- Check first: a power calculation and wall-clock budget at 19 tokens/s before choosing n.

4. IQ2_XXS extension — yes to deferring, but fix the gate.
- Main risk: 7 of 24 layers may show no resolvable effect even if the IQ2_XXS layers would benefit, so a null would wrongly kill the extension. IQ2_XXS is a lattice codebook with coupled signs, so half a day looks optimistic.
- Check first: a per-layer swap ablation showing what share of the sibling file's KL comes from the IQ2_XXS layers, and "shows an effect" defined numerically beforehand.

Missing points
1. v2m's speed is not benched. Speed is the release's headline claim, and "same types as a sibling" is inference. Bench v2m itself (generation, prompt processing, peak memory, all tensors on Metal) at a pinned stock llama.cpp commit.
2. There is no sealed out-of-domain set. v2m is the second attempt after v2r, with T1 and T2 already seen, so the 70% on T2 is not confirmatory. Seal a new reasoning-trace or chat set before release.
3. Reproducibility and the failure plan are absent: publish the search code, calibration manifest, seeds and Gram reference build hash; state whether the search is deterministic; and decide now what ships if T3 or the fresh domain fails.
