You are on a two-member council deciding the next 48 hours of a one-person research project. Today is 21 Nov 2026. Research the live web where it matters (new Inkling 2 Mini quant uploads on Hugging Face in the last hours, llama.cpp PR state) and trust it over memory. Never invent links or numbers. Label claims V (vendor/authors), I (independent), Est. (estimate) or Verify.

## The person and the goal
A master's student (Cognitive Science, career aim: LLM quantization research) starting a thesis on quantizing Inkling 2 Mini, released 20 Nov 2026. They want, fast, something that is special, capable and actually useful, and are willing to upload a model within a day. They also want the thesis to stay sound (a private pre-registration v1 exists; any change goes into a dated changelog before data). Thesis ends 9 May 2027.

Hardware: MacBook Pro M5 Pro, 48 GB unified memory (GPU wired limit 40 GB), ~640 GB free disk. RTX 3080 Ti mobile 16 GB laptop. GCP credits (50-200 GPU-h budget, each spend needs approval). Rule: one heavy run at a time, about 30 GB RAM max.

## Verified facts (checked 21 Nov 2026 against primary sources)
- Inkling 2 Mini: MoE looped transformer, 26.4B total, 3.4B active, 24 stored layers; layers 6-17 run 3x per token (48 layer visits); each pass restarts from layer 5's output and gets only gated rank-64 messages from the previous pass. 128 experts/layer, top-8. Experts ship NVFP4 (layers 0-19) and MXFP8 (20-23); everything else BF16; no BF16 master released. Companion TML-d24-varloop-research: same layout, 1-5 passes, RL only at 3; card warns behaviour may be unpredictable at other pass counts.
- Official llama.cpp: loop support merged as PR #32694 (b12600, commit 29b52d3). Pass count is a load-time key (`inkling2_loop.loop.count`, --override-kv). An open PR #32704 fixes passes 2-3 missing per-expert .scale tensors (affects only GGUFs with .scale tensors); applied locally with a test that fails without it. A local pinned build with that fix exists and passes upstream's random-model tests.
- A second, unmerged GGUF schema exists (draft PR #32695, Unsloth's fork). Unsloth and AtomicChat files need it and reportedly do not load on official llama.cpp.
- `ffn_down_exps` rows are 1,152 wide, so llama-quantize silently replaces every 256-block type (IQ2/IQ3, Q2_K-Q4_K) with IQ4_NL/Q4_0/Q5_0 on the down projections. Q2_0 (2.25 bpw, block 64) fits. llama.cpp's Q2_0 quantizer ignores the imatrix (AtomicChat patched this).
- llama.cpp's imatrix pools all 3 visits of a looped tensor under one name.
- KLD logits for llama-perplexity: about 52.7 GB per 64x4096-token reference (vocab 201,024); smaller chunk counts scale down linearly.
- ggml-org published BF16 (52.2 GB, does not fit 48 GB), Q8_0 (KLD 0.0030 vs BF16, 40x512 WikiText), IQ4_NL (0.0095) and an imatrix computed at 3 passes on bartowski v6 calibration data.

## Existing quants (one day after release)
| Repo | Runtime | What |
|---|---|---|
| AtomicChat/Inkling-2-Mini-GGUF | fork #32695 only | 12 per-tensor "AD" builds 6.9-23.8 GB with KLD/top-1 vs BF16 on an 87x4096 30-language corpus; correct block-32 down types; imatrix-aware Q2_0 patch. AD-IQ3_XXS 12.58 GB KLD 0.0271, AD-IQ2_S 11.56 GB KLD 0.047. Card guesses "rounding adds up over the 48 layer passes; we have not tested it". |
| unsloth/Inkling-2-Mini-GGUF | fork only | UD ladder; UD-IQ3_XXS 11.66 GB KLD 0.0269, UD-Q2_K_XL 9.14 GB 0.0595, vs their UD-Q8_K_XL on seven undisclosed domains |
| InterpreticAI/Inkling-2-Mini-IQ3_XXS-GGUF | official (29b52d3+) | 11.7 GB, imatrix, no quality numbers; also a Q4_K_M 17.5 GB |
| ggml-org | official | BF16, Q8_0, IQ4_NL |
| MLX community (hermitdave RCO 3.5 bpw KL 0.0669; araysuter mixed 2-bit; netprojects 2.5-bit) | MLX | visit-blind |
No published build or paper chooses bits from measured per-pass or per-block loop sensitivity (searched 21 Nov). The Loopy/LoopQuant/LoopQ groups and IST-DASLab (GSQ-RCO GGUFs) are the main 2-6-week research threats; community ladders (bartowski etc.) are expected within days.

## Plan A: thesis-first (from a four-reviewer design review)
Week 1-2 (G0 by 6 Dec): pin build; verify P changes 24/48/72 layer visits; reference that fits 48 GB with measured Q8_0, run-to-run and engine floors; fix level grid (per-projection types, block-32 down); write a llama.cpp patch that "unties" core weights per pass so the same quantized block can be switched on in pass 1 only, 2 only, 3 only, or all; message ablation (quantize pass k, feed pass k+1 the clean message) and route-lock mediation. Main H1 test: at P=3 compare e_all to sum of e_k in logit space (once / per-pass / compounding). Public note with the first per-pass result ~14 Dec. Then a pass-aware vs pass-blind allocation in one pipeline on held-out data (G2, ~31 Jan), small task pilot, then a reduced task panel (G3, ~28 Feb). No method integrations (no GSQ, LittleBit-2, trellis). Model release at the end.

## Plan B: fast release (about 4 hours of mostly unattended Mac runs, upload tonight after approval)
1. Download Mini (20.2 GB), convert to BF16 GGUF, build Q8_0 reference (~45 min).
2. Quantize 2-3 "pool" files at fixed levels (2-bit: IQ2_XXS gate_up + Q2_0 down; 3-bit: IQ3_XXS gate_up + IQ4_NL down; maybe IQ4_XS), with ggml-org's imatrix (~45 min).
3. A small gguf-py script splices tensors between pools, so each per-block test (blocks: L0-5, L6-11, L12-17, L18-19, L20-23) is a file copy instead of a requantize; evaluate KLD/top-1 vs the Q8_0 reference on 16x4096 held-out chunks (~30 min).
4. The student picks the allocation from the 5-block table (the allocator is theirs under a project rule).
5. Build a ~11.7 GB loop-aware GGUF for official llama.cpp; compare head-to-head with InterpreticAI's 11.7 GB IQ3_XXS and a same-size uniform build; stability checks: llama-server chat smoke test, 30 GSM8K prompts with a token cap (no reasoning loops/garbage) (~1 h).
6. Upload only if it wins, with the sensitivity map on the card; log the run as exploratory in the changelog first. If it loses, no "special" claim; the map is still the first result.

## Your task
Decide what is best for the next 48 hours and why: A, B, or a specific hybrid. Be blunt and concrete. Cover, numbered:
1. Verdict and the single reason that decides it.
2. Is Plan B's "special" claim real and defensible? Will a per-block (not per-pass) map at P=3 really be novel, or does it just restate what any sensitivity sweep shows? Is beating InterpreticAI's file at equal size likely, and is it a meaningful bar (vs AtomicChat AD-IQ3_XXS on the fork)?
3. What would make the release more useful or more special within 48 h without breaking the thesis (e.g. a per-pass result first, a different size, mainline-only niche, a 16 GB GPU target, an IQ2-level build, publishing the map without a model, contributing upstream)? Which one, and what does it cost?
4. Measurement traps specific to Plan B (reference choice, 16 chunks, calibration overlap with bartowski v6 imatrix data, splicing correctness, size matching incl. mmproj/MTP, stability checks too weak).
5. Kill criteria and exactly what may and may not be claimed on the model card.
6. A concrete hour-by-hour plan for the next 48 h with the chosen option, including where the student makes decisions.
Keep it to 700-1,300 words.
