ROLE_LINE_HERE

You are reviewing a quantization project at a decision point. Step back from the latest experiment and judge the work and the options as a whole. Everything below is measured unless marked otherwise; the owner's rules are at the end.

Answer as a numbered list of at most 10 points. Each point: a label [measured] / [inference] / [needs source], one claim, and a concrete action if any. Keep it under 900 words. Be blunt; disagreement is useful.

## 1. The project

- Model: Inkling 2 Mini, a looped MoE (24 layers, width 2304, 128 experts, top-8; one fused gate_up 2304→2×1152 and one down 1152→2304 per expert). Layers 6-17 run three times per token on the same weights (P = 3), so 36 of 48 layer runs use loop weights. Expert weights are 89% of an 11.6 GB file. Down rows are 1152 wide, so only 32-block types fit (IQ4_NL, NVFP4, Q2_0, Q4_0…).
- Two goals: (a) a master's thesis, "Counting Passes", on how quantization error behaves when weights are reused across loop passes; (b) public GGUF releases for Apple-silicon Macs, which have become popular. The owner wants the next big release to carry real advances of our own, and the card's speed claim gets fixed in that release.
- Thesis evidence so far: across visit splits, KL(all passes quantized)/sum(KL of one pass at a time) runs 0.87 to 1.23 (about 0.33 would mean "counts once", about 3 fully aligned); the third pass contributes most. A pass-split rule was met (ratio 0.748 [0.732, 0.761] against a layer split 1.108 [1.074, 1.145]); tied noise is about 14% super-additive.
- Metric: D_KL(Q8_0 ‖ file) per token at P = 3 against our Q8_0, on S (16 × 4096 WikiText tokens, used for selection), T1 (16 × 4096 other WikiText), T2 (the Q8_0's own GSM8K reasoning traces, 5 × 2048), and fresh five-domain sets (A3: python, javascript, german, english, math; 12 documents × 4 chunks each; now used up). Cross-file comparisons use G, the geometric mean over domains of the mean log KL ratio, with a domain-stratified document bootstrap. Plus a 300-item task battery (GSM8K 80, BFCL 100, MBPP+ 60, JSON 60).

## 2. Files and results

Fresh-set A3, token-weighted mean KL:

| file | GB | KL |
|---|---|---|
| ours c1 | 8.93 | 0.0683 |
| ISTA-DASLab GSQ-RCO IQ2_S | 8.98 | 0.0930 |
| ISTA Q2_K_XL | 9.79 | 0.0629 |
| ISTA IQ3_XXS | 10.61 | 0.0437 |
| ISTA IQ3_S | 11.43 | 0.0312 |
| ours v2c (released, refined codes) | 11.57 | 0.0220 |
| Unsloth UD-IQ3_XXS | 11.66 | 0.0195 |

- G v2c/ISTA IQ3_S 0.710 [0.692, 0.727]; c1/ISTA IQ2_S 0.757 [0.734, 0.781]; v2c/UD 1.116 [1.095, 1.136] (UD closer in all 5 domains, python's CI crosses 1). At 9 GB Unsloth's UD-IQ2_XXS (8.49 GB) matched c1 (0.0691 vs 0.0703, held-out set of a sibling project, not resolved).
- On T2 the ISTA gap narrows (c1 vs IQ2_S 2%, v2c vs IQ3_S 17%, against 27% and 32% on S). ISTA calibrated on model-generated text.
- Tasks: no resolved difference against ISTA at either size (+2.0 [-0.8, +4.5] and +1.7 [-0.4, +3.2] points, one-sided 95%).
- Same-type tensor swaps between v2c and ISTA IQ3_S change KL by at most about 2%: at fixed type the encoders are close. Joint scale-and-code refinement (GSQ-style) bought 1% of the local objective in a loop layer and nothing resolved in KL; stopped.
- ISTA's layout ideas tried singly on v2c (KL on S, v2c 0.02099): gate_up 0-2 to IQ1_S −212 MB +0.00249; downs 0, 1, 3 to Q2_0 −287 MB +0.00171; loop attn_v to Q8_0 +7 MB −0.00017; token_embd Q8_0 +232 MB (CPU-resident, 0 GPU bytes) −0.00063; gate_up IQ3_XXS in 6, 7, 8, 12, 18 +212 MB −0.00056; IQ3_S gate_up in 9, 10, 15 no effect. Every size-neutral subset was +0.0012 to +0.0013 worse than v2c, about 0.0004 worse than additive.
- v2c already copies Unsloth's gate_up type placement. Earlier we found most of Unsloth's gate_up edge lies in the encoded values, not the type map, with importance weighting (calibration data, or how importance is pooled over the three visits) as the suspected lever. Our encoder uses a public imatrix and refines codes under tr(E H E^T), H = G + 0.1 diag G, with a row output-gain floor.
- Speed: on llama.cpp before 23 Nov, Unsloth's NVFP4 down tensors ran on the CPU and our all-GPU file was 3.3× faster at prompts and 1.75× at generation (this is on the public card). llama.cpp master since 23 Nov runs NVFP4 on the Mac GPU: v2c 443 / 31.4 t/s vs UD 436 / 29.7 (pp512 / tg128, interleaved). The speed edge is gone on current builds.
- Newest (registered diagnostic, a donor build never to ship): v2c with Unsloth's NVFP4 downs in layers 0-19 instead of our IQ4_NL (same 4.5 bpw, same size): S 0.01990 vs v2c 0.02099, −0.00109 [−0.00146, −0.00068]; T1 −0.00099; T2 −0.00101. Against UD: S −0.00060 [−0.00145, +0.00029], T1 −0.00048, T2 +0.00019, none resolved, at 97 MB less. The registered rule opens a v3 track with our own NVFP4 encoding.

## 3. Options on the table

A. v3 with our own NVFP4 downs (needs the owner's NVFP4 tool; source weights: BF16 rebuild, 52 GB, or the original checkpoint). Ships as an extra file: builds before 23 Nov would run those tensors on the CPU.
B. Loop-aware gate_up exchange, size-neutral: v2c holds IQ2_S in loop layers 6, 7, 8, 12 and IQ3_XXS in once-run layers 19-23; swap them (42.5 MB per step).
C. A measured "price list": one step up and one step down per group (early/loop/late gate_up, downs, attention by role, embedding), about 12 KL runs, then an allocator picks a size-neutral set. Includes softer early layers (IQ3_XXS in 0-2, +170 MB) and all loop attention at Q8_0 (+78 MB). Proposed base: the donor NVFP4 file, for pricing only.
D. Visit-pooled importance (pool calibration inputs from all three passes) plus model-generated calibration text; re-refine all 24 layers (about 10-12 CPU hours). The owner writes the accumulation.
E. Q8_0 embedding: +232 MB disk, 0 GPU memory, −3% on S. Product call on whether disk size counts.
F. Smaller file (9 GB tier) improvements, the write-up, or upstream contributions.

Plan as proposed (inference): A first, C on the donor base, then D, combine, register, select on S, confirm on T1, one check on a new fresh set A4 against Unsloth and ISTA, then the release with the corrected speed claim.

## 4. Rules

Every number comes from logs via a script. Registrations precede data. The owner writes the core thesis functions (the NVFP4 snap tool, visit-pooled imatrix, bit allocator, the error-versus-pass analysis); the assistant runs, tests and reviews. One 48 GB Mac; one KL run on S is about 4 GPU minutes; refinement costs 25-31 CPU minutes per layer. Downloads above 50 GB, cloud spend and every publication need the owner's go. Own methods win; donor builds are diagnostics only.
