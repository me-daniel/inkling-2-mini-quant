You are reviewing a master's thesis design you did not write. The brief is below. Today is 21 Nov 2026. The model family it targets was released on 20 Nov 2026, so your training data is out of date: research the live web (arXiv listings and new submissions, Hugging Face model and quant repos, GitHub PRs and issues for llama.cpp, transformers, mlx-lm, vLLM, lab blogs and announcements) and trust what you find over memory. Never invent a paper, link, number or arXiv id. Give each source as a URL with its date, and label every claim V (vendor or authors about their own work), I (independent), Est. (your estimate) or Verify (could not confirm).

Number every point. Be concrete and blunt. No praise. Cover, in this order:

A. Hypotheses H1–H5. For each: is it well posed and identifiable with this design? Confounds and alternative explanations (for example: varloop at P≠3 is off-distribution because RL ran at 3 passes only; the reference changes with P; block interactions; the native FP4 reference has no BF16 master; float vs NVFP4 activations; WikiText vs reasoning traces). The result that would falsify it. Missing controls or baselines. A cheaper design that answers the same question.

B. Prior art and scoop risk. What already exists on quantizing looped / recurrent-depth / weight-shared transformers, on quantizing Inkling 2 Mini or TML-d24-varloop specifically, and on visit-aware or recurrence-aware calibration (check the papers in section 8 and anything newer). Then forecast: what will other labs or community quantizers (Unsloth, ggml-org, bartowski, IST-DASLab, the Loopy/LoopQuant/LoopQ groups, Red Hat/Neural Magic, Intel AutoRound, NVIDIA ModelOpt, MLX community, EXL3) most likely release in the next 2 days, and in the next 2–6 weeks? Which of those would overlap with H1–H5, and which would only be baselines?

C. Novelty. What exactly is novel here after B, and what is not? What is the sharpest, defensible one-sentence contribution? What would make it novel enough for a thesis and a workshop paper (e.g. at a 2027 efficient-ML workshop)?

D. Getting it done. The most likely ways this project fails or stalls (tooling, the pass count not being settable, compute, time, statistical power, scope). For each: an early test, a kill or pivot criterion, and a fallback. What is the minimum viable thesis if most things go wrong? Are the gates G0–G3 and their dates realistic?

E. Integrations versus de-risking. Should it integrate a stronger method, and which: GSQ or other IST-DASLab work (e.g. their newest quantizers, MoE or expert-level methods, FP4 kernels), LittleBit 2 (research what it is and whether it applies to MoE experts and looped models), LoopQuant/Loopy/LoopQ ideas, Feedback-Exposure-style Hessian summing, QTIP/trellis or vector quantization, or something else? For each candidate: what it adds to the contribution, the cost in weeks and GPU-h on this hardware and budget, and whether it is blocked by the tooling in section 4. Or should the project stay narrow and cut risk instead? Give a clear recommendation.

F. What else to add or beware of: measurement (KLD reference choice, chunks, bootstrap, multiple comparisons, matched-size accounting), licensing and the acceptable-use policy, reproducibility, compute traps, anything the brief misses.

G. Your ranked top 7 changes to the plan, each with its cost and the hypothesis or gate it affects.

Keep it dense. Aim for 2,000–3,500 words.

=== BRIEF ===
# Design brief: "Counting Passes" (master's thesis, Cognitive Science, LLM quantization)

State as of 21 Nov 2026. Labels: V = vendor or paper authors about their own work, I = independent, Est. = estimate, Verify = not yet confirmed.

## 0. Context and constraints
- One student, master's thesis. Runs to 9 May 2027 (24 weeks). Wants a thesis that is simple, novel, independent of other people's code, and on a predictable workflow. Career interest: LLM quantization research.
- Hardware: MacBook Pro M5 Pro, 48 GB unified memory, 307 GB/s, ~690 GB free disk, llama.cpp and MLX installed. A laptop with an RTX 3080 Ti mobile (16 GB). GCP credits, budget 50–200 GPU-h.
- Direction: a pass-count study plus visit-aware sub-4-bit bit allocation for Inkling 2 Mini, with native-grid nested levels (2–3-bit subsets of E2M1 that reuse the E4M3 scales) as one extra level.
- A pre-registration of the design below was written on 20 Nov 2026 (private so far, to be mirrored publicly). Hypotheses, metrics and analysis may change only via dated changelog entries.

## 3. Model facts (V unless marked)
**Inkling 2 Mini** ([HF](https://huggingface.co/thinkingmachines/Inkling-2-Mini)):
- Thinking Machines Lab, released 20 Nov 2026, Apache 2.0 plus an acceptable-use policy (AUP).
- MoE looped transformer, 26.4B total, 3.4B active (5.4B counting loops).
- Arch `Inkling2LoopForConditionalGeneration`, `model_type` `inkling2_loop`.
- **Layers:**
  - 24 layers, hidden 2304, vocab 201,024, 24 Q / 8 KV heads, head_dim 128.
  - 20 of 24 layers use sliding-window attention (window 1024); each 6-layer block has 5 SWA and 1 full-attention layer.
  - Short convolution with kernel 4; final logit softcap 30.
- **Experts:** 128 SwiGLU experts per layer, top-8, **no shared expert**, intermediate 1152, router bias.
  - Experts were pruned from 384 during midtraining and healed on 1T distilled tokens.
- **Loop:**
  - Layers 6–17 run 3× (48 visits per token). Each pass restarts from the output of layer 5.
  - Each visit keeps its own KV and conv state.
  - From pass 2 on, a cross-pass "attention residual" uses four rank-64 gates into Q, K, V and the MoE input.
  - Conflict: config.json says the loop range is "6 to 18" (probably half-open, Verify).
- **Precision (from `hf_quant_config.json`):** only the routed-expert tensors `mlp.experts.w13_weight` and `w2_weight` are low-bit.
  - **Layers 0–19:** NVFP4 (E2M1 values, 1×16 blocks, E4M3 scales, round-to-nearest export, per-expert FP32 outer scale).
  - **Layers 20–23:** MXFP8 (32×32 tiles, UE8M0 scales).
  - **Everything else:** BF16/FP32.
  - Conflict: config says `modelopt`, `hf_quant_config.json` says `quant_method: inkling2` (Verify).
- **Training numerics:** trained with NVFP4 activations in the expert GEMMs (per-token "Four Over Six" scale selection). llama.cpp and MLX run float activations, so a small offset from training numerics is unavoidable and should be reported.
- **No BF16 master exists.** The FP4 checkpoint (~19.5 GB) is the reference model.
- **Reasoning effort:** none 0.0 / minimal 0.1 / low 0.2 / medium 0.7 / high 0.9 (default) / xhigh 0.99. Even the minimum does not fully disable reasoning, and the model can loop in reasoning, so always cap output tokens.
- **Sampling** (Unsloth): T 1.0, top-p 1.0, top-k 0, min-p 0.

**TML-d24-varloop-research** ([HF](https://huggingface.co/thinkingmachines/TML-d24-varloop-research)):
- Same layout and precision as Mini; Apache 2.0 + AUP; not on Tinker.
- Passes 1–5 were sampled 1:2:6:2:1 in late midtraining; RL was run at 3 passes only.
- Exposed as `loop_num_passes` in `text_config` (transformers).
- Active params: 3.4B / 5.4B / 7.4B at 1 / 3 / 5 passes.
- SWE-bench Verified by pass count 1–5: 56.3 / 76.5 / 78.7 / 79.3 / 79.6.

## 4. Tooling status on 20 Nov 2026
| Tool | State |
|---|---|
| llama.cpp ≥ b12406 (PR #32217) | **The only working requantization path.** The converter decodes packed NVFP4/MXFP8 experts. The PR said "real checkpoint conversion untested", but Unsloth and ggml-org shipped full imatrix ladders, so it works in practice. |
| transformers PR #51050 | **Open and unmerged.** Its FP8 quantizer wrongly dequantizes the BF16 lm_head (breaks a float32 forward pass); CI failures. So llm-compressor, GPTQModel, AutoRound, ModelOpt and GSQ's reference code are all blocked. |
| mlx-lm PR #2086 | **Draft**, no DSpark. Alternative: the nativ-community MLX repack plus a patched mlx-vlm 0.8.0+inkling2mini wheel. |
| GSQ (IST-DASLab, arXiv 2604.18556) | Strong 2–3-bit method that can refine GGUF K-quants, but its code expects standard HF architectures. Qwen3-30B-A3B is listed on 4×H200. Stretch goal only. |
| Speed | The only measured figure: M3 Ultra ~29–34 tok/s with mlx-lm (I). Nothing measured on the M5 Pro yet. |

**Existing quants, as baselines:**
- **Unsloth UD GGUFs (V):** KLD against their UD-Q8_K_XL reference (22.87 GB, which keeps the NVFP4 experts and upcasts MXFP8 to BF16).

| Quant | Size | KLD | Same top-1 |
|---|---|---|---|
| Q4_K_XL | 17.63 GB | 0.0107 | 98.0% |
| Q4_K_M | 16.61 GB | 0.0108 | — |
| IQ4_XS | 15.73 GB | 0.0109 | — |
| Q3_K_XL | 13.63 GB | 0.0174 | — |
| IQ3_XXS | 11.66 GB | 0.0269 | — |
| Q2_K_XL | 9.14 GB | 0.0595 | 91.6% |
| IQ1_S | 7.76 GB | 0.0911 | — |

  Their imatrix data and per-tensor recipe are undisclosed. MTP is dropped at ≤Q2_K_XL, so exclude MTP from size comparisons.
- **ggml-org:** IQ4_NL and Q8_0 for both models.
- The flat 4-bit floor (0.0107–0.0109 across 1.9 GB) suggests that **switching to a non-FP4 grid**, not the bit count, causes the 4-bit loss (Est.).

## 5. Pre-registered design (identical to the published page)
- **Core question:** when shared FP4 expert weights run in every pass, does their quantization error count **once**, **once per visit**, or **compound**? Does an allocation built on the measured answer beat a visit-blind one at matched size?
- **Instrument:** the varloop model at P = 1…5, then confirmation on Mini at P = 3.
- **Intervention:** quantize one block's experts at a time, everything else native.
  - **Blocks:** B1 = L0–5, B2 = L6–11, B3 = L12–17, B4 = L18–19, B5 = L20–23.
  - **Levels:** IQ2_XXS, IQ3_XXS, a nested E2M1-subset level, and Q4_K as a "same bits, new grid" control.
  - Non-expert tensors stay fixed at Q8_0.
- **Hypotheses:**
  - **H1:** ΔKLD(P) for B2/B3 grows faster than for B4/B5. A flat curve means the error counts once, a linear one means once per visit, a superlinear one means it compounds. B1 shows whether the loop contracts or amplifies upstream error, which tests Tilted Bowl's claim.
  - **H2:** an imatrix pooled over all visits beats one from the first visit only.
  - **H3:** at matched size, the measured end-to-end allocator beats "summed per visit", which beats "counted once". Budgets: 9.14 GB and 11.66 GB, against Unsloth UD-Q2_K_XL and UD-IQ3_XXS.
  - **H4:** router top-8 Jaccard against the reference falls with pass index in the looped layers.
  - **H5:** B5 (MXFP8) tolerates ~4 bits with negligible ΔKLD (tests the vendor's last-sixth rule).
- **Tier 0 (primary):**
  - KLD, top-1 agreement and perplexity against the native reference **at the same P**.
  - Data: 64 × 4,096-token WikiText-2 chunks (Unsloth-comparable) plus ~250K tokens of the reference model's own reasoning traces.
  - Bootstrap CIs over chunks.
  - If the eval callback exposes them, also per-visit hidden-state error and router flips.
- **Tier 1 (confirmation, six final builds):**
  - Tasks: GSM8K-500, MMLU-Pro-1K, IFEval (541), LiveCodeBench-150.
  - Settings: effort 0.2, 16K token cap (truncation counts as failure), T = 1.0, 3 seeds.
  - Stats: paired McNemar tests and Wilson intervals.
  - Report output length and truncation rate as outcomes, since quantization can inflate chain-of-thought length (Lotfi et al., arXiv 2606.00206).
- **Allocator:** per-layer expert levels from a greedy or ILP solver (MODE-style), with ~10 pairwise block configs to check additivity.
- **Gates:**

| Gate | Week | Passes if |
|---|---|---|
| G0 | 2, by 6 Dec | Both models convert; Unsloth's UD-Q2_K_XL KLD reproduced within ±15%; P settable |
| G1 | 6, by 3 Jan | Between-block ΔKLD exceeds the CI width; then publish the public map note |
| G2 | 10, by 31 Jan | Allocator beats Unsloth at matched size |
| G3 | 14, by 28 Feb | Tier-1 ordering agrees with tier-0 |

  Weeks 15–24 (to 9 May 2027): analysis, release, writing.
- **Fallbacks:**
  - If P can't be set: run the block decomposition on Mini at P = 3.
  - If no allocator wins: the sensitivity map is the contribution.
  - If KLD mispredicts tasks: report the mismatch.
- **Compute (Est.):** 40–135 GPU-h in total.
  - The tier-0 grid (75 quants plus 5 references) runs on the Mac.
  - Tier 1 runs on a GCP G4 (1× RTX PRO 6000, 96 GB) at about $1.58/h spot (V, region unstated).
  - H100/H200 are sold only as 8-GPU nodes at about $51–53/h spot.

## 6. First tasks (G0), in order
Ask the student before downloads above ~50 GB, before any cloud spend, and before any public push.

1. **Repo skeleton:** create `counting-passes/` with `PREREG.md` (the page text), `CHANGELOG.md`, `scripts/`, `configs/`, `results/` and `README.md`. MIT or Apache-2.0 for code (ask him which).
2. **llama.cpp:** fresh build at ≥ b12406 with Metal; record the commit hash in `results/env.md`.
3. **Downloads** (on the Mac; this cloud sandbox can't reach Hugging Face):
   - `thinkingmachines/Inkling-2-Mini` (~20 GB) and `thinkingmachines/TML-d24-varloop-research`.
   - `unsloth/Inkling-2-Mini-GGUF`: only UD-Q8_K_XL, UD-Q2_K_XL and UD-IQ3_XXS (~44 GB in total).
4. **Read the raw metadata:**
   - `config.json`, `hf_quant_config.json` and the safetensors headers of both repos.
   - Resolve the modelopt vs inkling2 conflict, the loop range, embed/unembed tying and whether varloop's layout matches Mini.
5. **Convert** both with `convert_hf_to_gguf.py` to a Q8_0/BF16 reference GGUF. Then dump per-tensor types (`gguf-dump` / `gguf_dump.py`):
   - which GGUF type holds the NVFP4 experts;
   - the GGUF tensor names for experts (likely `blk.N.ffn_*_exps`, Verify);
   - whether Unsloth's sub-4-bit files requantize the NVFP4 experts or pass them through.
6. **Pass count in llama.cpp:**
   - Find the GGUF metadata key for the loop count in the Inkling2 implementation (`src/models/…`).
   - Test `--override-kv <arch>.<key>=int:N` (Verify), or patch it (≤2 days allowed).
   - Sanity check: varloop at P = 1…5 should show perplexity improving with P.
7. **Reproduce the Unsloth protocol:**
   - Use `llama-perplexity` on 64 × 4,096 WikiText-2 tokens: `--kl-divergence-base` on the reference, then `--kl-divergence` on UD-Q2_K_XL.
   - Target: KLD 0.0595 ±15%.
   - Note that Unsloth's reference upcasts MXFP8 to BF16; match it.
8. **imatrix pooling:** read the llama.cpp `imatrix` code. Does it accumulate the 3 visits of a looped tensor, or keep only the last? This decides H2's implementation.
9. **Block-wise quants:** check that `llama-quantize --imatrix … --tensor-type <regex>=<type>` can target the experts of individual layers. Build B1–B5 × {IQ2_XXS, IQ3_XXS, Q4_K} with everything else at Q8_0.
10. Write the G0 result into `results/G0.md` and `CHANGELOG.md`. Report to the student in a few lines.

**Nested E2M1-subset level:** no kernel is needed for accuracy.
- Snap each NVFP4 value to a 2- or 3-bit subset of the E2M1 points while keeping the E4M3 block scales.
- Write the result as dequantized values into a Q8_0 or BF16 GGUF.
- Measure against the "Q8_0 of native" floor.
- This needs a small Python tool on the safetensors (decode NVFP4 → snap → re-encode). Design it after G0.

## 7. Risks to watch
- The three groups behind Loopy, LoopQuant and LoopQ (October 2026) are the most likely to quantize Inkling next. Publish the G1 map quickly.
- Community EXL3, AutoRound and MLX 2-bit builds will appear within weeks. They are baselines, not scoops.
- Engine support for the loop on non-SM103 GPUs (vLLM/SGLang) is unverified. Assume llama.cpp server for tier 1.
- **The perplexity trap:** perplexity hides collapse in looped models. Tilted Bowl reports that Ouro-1.4B INT4 RTN raised PPL from 11.51 to 14.38 while GSM8K fell from 73% to 7%. So don't stop at tier 0 for final claims.
- Effort settings change output length a lot; keep effort and token cap fixed and logged.

## 8. Key literature (all V)
- [LoopQ 2605.16343](https://arxiv.org/abs/2605.16343), [Loopy 2610.05265](https://arxiv.org/abs/2610.05265) (code: Shameless0817/Loopy-review), [LoopQuant 2610.16368](https://arxiv.org/abs/2610.16368) (code: Kishon-zzx/LoopQuant): accumulation-based methods on Ouro, Huginn and Nanbeige4.2-3B; W4A4/W4A8.
- [Feedback Exposure 2609.30820](https://arxiv.org/abs/2609.30820): one-step GPTQ is worse than RTN on 5 of 9 looped checkpoints; summing the Hessian over recurrence steps closes 17–102% of the gap.
- [Tilted Bowl 2609.39277](https://arxiv.org/abs/2609.39277): error only shifts the fixed point in loops that settle; running the final loops at high precision recovers collapsed models; a cheap failure predictor.
- [GSQ 2604.18556](https://arxiv.org/abs/2604.18556), [MODE 2606.17118](https://arxiv.org/abs/2606.17118) (expert ILP allocation), [VSRAQ 2606.05688](https://arxiv.org/abs/2606.05688) (routing consistency, 4-bit only).
- KV cache in looped models (not claimed): ResidualQuant 2610.10381, LoopVQ 2611.05304.
- Adjacent: Looping Beyond Twice (looped-MoE training, 2610.01153); Hyperloop Transformers (2604.21254).
