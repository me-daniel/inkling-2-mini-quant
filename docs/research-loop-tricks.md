# Loop-specific and low-bit tricks: what exists, what is new for Inkling (22 Nov 2026)

Three GPT-6.2 Sol searches with web access (prompts: public facts only), then the key claims opened again by separate fetches. "Checked" means a page was opened and the quoted claim found there; everything else is the search's report only.

## Prior art that touches our own work

1. MACaQ, Appendix C (arXiv 2611.10649 v2, 11 Nov 2026), checked. Codes-only refinement of llama.cpp IQ3_S and IQ3_XXS: each 4-weight block takes the admissible codeword nearest to a Newton-step target in the metric of the layer's (regularised, transformed) Hessian, best improving move per block, at most 8 sweeps, scales fixed by default; a scale refit lowered the objective but not perplexity. Dense Qwen3-4B and 14B only, WikiText-2 train calibration (10^6 tokens). IQ3_XXS 4B: 13.09 to 12.64 WikiText-2 PPL over imatrix.
   - Consequence: the core of `scripts/refine-codes.py` (full-Gram codeword and sign search with fixed scales) was published 11 days before ours. We built it independently, but any write-up or card that describes the refiner must cite MACaQ as prior work.
   - Ours adds: per-expert Grams for a MoE, pooled over the three loop visits; IQ2_S with a sign search and polish; the output-gain guard (amendment 9c, from the attenuation finding); the minimum-token rule; the domain-transfer measurement (amendment 12 on T2).
2. "Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness" (arXiv 2609.30820), checked. GPTQ with the Hessian accumulated over all loop invocations beats one-step GPTQ and RTN on 9 of 9 looped checkpoints. Our Grams and llama-imatrix already pool all visits; per-pass weighting of the pool is H2's question (Daniel's function).

## Candidates not yet used on Inkling

1. Per-pass gain with frozen weights. LoopQuant (arXiv 2610.16368, checked) fits per-loop clipping, gain and bias on frozen quantized weights (gains start at 1); its ablation credits this part with a smaller gain than its other two components. Fits our attenuation finding. On Inkling it needs a llama.cpp patch so a cache slot shares the physical experts but carries its own `.scale`, so it would not run on stock llama.cpp.
2. Scale-only end-to-end KL distillation. Marco-Nano (llama.cpp discussion 31320, checked): IQ block scales trained against the bf16 teacher's logits on 24 to 50M tokens, codes kept, stock-loadable output; held-out KL 22 to 47% lower than llama.cpp's imatrix files (which are larger, not size-matched). Needs a differentiable forward pass; no working PyTorch path for Inkling yet.
3. Higher precision on late passes. "A Tilted Bowl Is Not a Slippery Slope" (arXiv 2609.39277, checked): precise final loops rescue collapsed low-bit looped models (Huginn w3a GSM8K 3% to 40%, bf16 42%; n = 100). Inkling has 3 passes, not 32; testable with our slot patch at a byte cost.
4. Forward-only selection through the full loop. Loopy (arXiv 2610.05265, search only): rank candidate encodings by final-output loss after all passes. Our analogue is a per-layer output-gain sweep scored by KL through all three passes (degain2.py already rescales block scales), stock format.
5. NVIDIA Dory1.0-4B (HF card and arXiv 2611.03998, checked): a looped model whose NVFP4 and Q5_K_M GGUF came from quantization-aware distillation with matched loop counts. Training-based, not post-training.
6. Loop-aware KV compression (LoopKV, LoopVQ, ResidualQuant; search only). Runtime memory, not file size.

## Other Inkling files found

- hridya423 "mixB" (HF, checked): 10.3 GiB, loop core IQ3_XXS, IQ2_S elsewhere, expert down rows zero-padded from 1152 to 1280 so IQ2/IQ3 fit the down projections. Needs a llama.cpp fork; upstream cannot load it. KL 0.048 against Q8_0 on WikiText-2 (its own setup). Not a stock-llama.cpp file, so outside our claim set.

## Order (proposal)

1. v2m (running): mixed-domain Grams, the calibration-composition lever.
2. Forward-only per-layer gain sweep on v2m, scored through all passes, stock format; the cheap test of the attenuation story.
3. If 2 shows per-pass structure: per-pass `.scale` via the slot patch (non-stock), then late-pass precision.
4. Scale-only KL distillation once a PyTorch forward exists (GPU spend, Daniel's go).
