# Pivot review: why UD encodes better

- Date: 2026-11-22, ~02:00
- Model: GPT-6.2 Astra, effort medium (Codex CLI, isolated)
- Response: CHANGELOG Addendum H amendment 5.

## Prompt

You are a skeptical quantization reviewer. Answer in under 700 words, numbered, each point with the cheapest discriminating experiment.

Setup: Inkling 2 Mini (looped MoE: layers 6-17 run 3 times per token on the same experts; 128 experts, top-8; source weights NVFP4 with a per-expert F32 scale in layers 0-19). llama.cpp GGUF. KL vs a Q8_0 reference on S (16 x 4096 WikiText-2 tokens), paired per-chunk bootstrap.

Facts (paired on S):
- Our file (cand): 0.02931. Unsloth UD-IQ3_XXS: 0.02050.
- Swapping in UD's gate_up expert tensors (their per-layer types AND their encoded values, plus their per-expert .scale) recovers 0.0081 of the gap.
- Our own llama-quantize output at UD's exact per-layer gate_up types (same imatrix we always use: ggml-org, 1,770 x 512 tokens of a chat-style calibration text, summed squared activations per input column) recovers only 0.0021. So UD's encoding at its own map is worth 0.0060 [0.0052, 0.0067] more than ours.
- Yet at the same type, UD's gate_up is further from our BF16 conversion than ours, in 17/17 layers, both plain relative L2 and weighted by OUR imatrix (gap about +0.006 to +0.009 relative error at IQ3_XXS/IQ2_S, 16 of 128 experts sampled).
- UD stores gate_up as quantize(W / s_e) with a per-expert F32 scale s_e applied at runtime (layers 0-19); our file stores quantize(W) directly with no extra scale. Layers 20-23 have no scale in UD, and the error gap is the same there.
- The down projections and non-expert tensors explain little (down format ~0.001; UD's non-expert tensors ~0).

Question: what most plausibly makes UD's tensors worse in weight space under our importance weights but much better in output KL? Rank the hypotheses (e.g. their importance weighting/calibration data, loop-visit accumulation of importance, expert/router-frequency weighting, a different rounding objective, a mismatch between our BF16 conversion and the release, scale handling, something about the gated product gate*up being fused). For each, give the cheapest experiment on this setup that would discriminate it, preferably one S run (about 10 minutes) or a CPU-only check. Say which single experiment you would run first.

## Reply (verbatim)

1. **Most likely: your importance metric misranks errors on the activations that matter. First experiment: frozen-activation replay.** Cache Q8 inputs, routes and router weights for one S chunk, covering all three visits to one looped layer and one unscaled layer. Replay both same-type encodings on identical routed tokens, with scales applied. Against Q8, compare squared error at four stages: separate gate/up preactivations predicted by the **S-derived diagonal** weights, actual preactivations, gated product, and router-weighted output through the same down projection. If the ranking flips at the first stage, calibration is the leading explanation. A flip only at actual preactivations implicates input correlations omitted by the diagonal. A later flip localizes the advantage to gating or downstream sensitivity. This is a diagnostic on S, not a calibration recipe. The paired interval establishes a 0.0060 KL advantage for the encoding **including its scale convention**; it cannot identify UD’s method.

2. **Different calibration data or importance handling is my leading encoder-side hypothesis.** At fixed source, scale convention, quantizer build and UD type map, re-encode only gate_up with an imatrix from disjoint WikiText-style text. Match the original calibration budget and context length to isolate corpus choice. Score once on S against the existing same-map baseline. Recovery would show that changing calibration alone suffices for that fraction of the gap. Also verify expert indexing and coverage in the actual imatrix; current llama.cpp stores expert-indexed sums and counts. [llama.cpp collector](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/imatrix.cpp)

3. **Different rounding, clipping or block-scale search is next.** On CPU, take identical source blocks from an unscaled layer and encode them with your build and UD’s build if identifiable, using identical importance columns. Compare decoded values, block scales and code choices to the released UD blocks. Reproducing UD where your build differs is strong evidence for the encoder. A different scale histogram alone is inconclusive because calibration can also change those choices.

4. **Source or conversion mismatch deserves an early CPU audit.** Verify checkpoint revisions for your conversion, UD and the Q8 reference wherever provenance exists. Independently decode sampled original NVFP4 blocks to FP32, apply the source scales once, and check your BF16 values after matching layout, expert order and gate/up halves. Compare both encodings to the independent decode and to dequantized Q8. A ranking reversal with the reference identifies a target mismatch; matching BF16 to source clears your decoder on those blocks but leaves undocumented UD provenance open.

5. **Loop visits matter through changed activation profiles or repeated error propagation.** Log per-expert counts and normalized column moments by visit on a short calibration batch. Check whether the actual collector pools all visits under the shared tensor. Tripling identical moments merely rescales the objective. If profiles differ materially and visits were omitted, compare first-visit versus pooled-visit imatrices at fixed map. Repeated propagation can amplify an encoding difference even when collection is correct.

6. **Gated-product sensitivity is plausible; fusion itself is weak evidence.** In the cached replay, cross UD gate with your up and vice versa. If one half carries the gain, that identifies the sensitive branch. If the UD pair gains substantially more than the two single substitutions predict, the errors interact at the product. Neither result establishes deliberate joint rounding.

7. **Expert frequency alone ranks low at a fixed type map.** Multiplying an independently encoded expert’s objective by one positive scalar preserves its minimizer. Test this directly by rescaling one expert’s imatrix and comparing encoded bytes. In replay, compare equal-expert and routed-token averages across all 128 experts to test whether the 16-expert sample hid a gain on busy experts.

8. **Scale handling alone ranks low but remains testable.** Encode your own \(W/s_e\), attach UD’s \(s_e\), hold everything else fixed and score once on S. The unscaled layers weaken a scale-only explanation for the **weight-error pattern**. They do not bound the scaled layers’ contribution to KL.
