# Fact check: Inkling 2 Mini quantization study (2026-11-21)

Fetched files live next to this one (config.json, hf_quant_config.json, README, api.json under Inkling-2-Mini/, TML-d24-varloop-research/, gguf/; llama.cpp sources under m/ at master 29b52d3 and src/ at 5c008e0).
HF revisions read: Mini `ed40e7da625f594fc9c4990f5f66be24873c22d7` (2026-11-20T22:55Z), varloop `657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2`, Unsloth GGUF `29eac40c55957e3b76db6b7028e549cb244745de`.

## Q1 Loop range: PARTLY (claim "6-17, 3 passes, restarts from layer 5 output" is right; "config says 6 to 18" is just half-open notation)

Both configs, `text_config.loop_config` (https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/config.json, same path in TML-d24-varloop-research):
- Mini:    `{"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": false}`
- varloop: `{"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": true}`
Only `is_varloop` differs in loop_config.

Half-open: yes, [6,18) = layers 6..17 (12 layers). llama.cpp `src/models/inkling2-loop.cpp` @ 29b52d398b294509746ec0a3f28948a3107c0213:
- L4  `return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);`
- L19 `core_start >= core_end || core_end > (uint32_t) n_phys`
- L98 `for (int ip = start; ip < end; ++ip) {`
- L85-97 `// every pass starts from the prelude output` / `auto * prelude = cur;` ... `if (pass > 0) {... cur = prelude; blocks = prelude_blocks; }`
Prelude = layers 0..5, so each pass restarts from layer 5's output (the output of the last layer before core_start). Confirmed. The Mini model card agrees: "The middle 12 layers run three times with shared weights, giving 48 layer applications per token from 24 layers of weights."
Constraint in the code: core_start and core_end must be multiples of attn_res_block_size (6); fixed loop needs >=2 passes, varloop >=1 (L18-20; converter conversion/inkling2.py L296-298).

Pass count location: `text_config.loop_config.num_passes` (NOT `loop_num_passes`, and not top-level). 

Layout shared by Mini and varloop (both configs, identical except loop_config.is_varloop and Mini has top-level `"dtype": "bfloat16"`; hf_quant_config.json files are the same size, 67924 bytes, and the tensor groups match):
- `num_hidden_layers` 24, `hidden_size` 2304, `n_routed_experts` 128, `num_experts_per_tok` 8, `n_shared_experts` 0, `intermediate_size` 1152, `n_layers_per_attn_res_block` 6.
- `local_layer_ids` = [0-4, 6-10, 12-16, 18-22] (20 SWA layers); full-attention layers are 5, 11, 17, 23, i.e. 5 SWA + 1 full per 6-layer block. Confirmed.
- Tied embeddings: NO. config.json has no `tie_word_embeddings`; hf_quant_config.json lists `model.llm.embed.weight` (bf16 group) and `model.llm.unembed.weight` (own group) separately; the safetensors header (Mini, first 148408 header bytes via Range) has `model.llm.embed.weight BF16 [201024, 2304]` and `model.llm.unembed.weight BF16 [201024, 2304]` as distinct tensors. llama.cpp creates separate `tok_embd` and `output` (src/models/inkling2.cpp). Untied.

## Q2 Quantization metadata: CONFIRMED

- config.json (both models), top level: `"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}`.
- hf_quant_config.json (both): top-level keys `model`, `mtp`, `quant_method`; `"quant_method": "inkling2"`.
- Note the two are inconsistent as labels: config says modelopt/NVFP4 for the whole model, but the real recipe is mixed (below).
- parameter_groups in `model` (counts of tensors):
  - fp32 cast, 74 tensors (attn_res queries, `middle_cycle_cross_pass_queries`, ...)
  - bf16 cast, 993 tensors (embed, norms, attention, routers, vision, ...)
  - `weight_transform: nvfp4_nearest`, `compute_dtype: nvfp4`, `weight_block_shape [1,16]`, `weight_scale_format: amax_denominator`, scale e4m3, activation block 16: 40 tensors = `model.llm.layers.{0..19}.mlp.experts.w13_weight` and `.w2_weight` (first `layers.0...`, last listed `layers.9.mlp.experts.w2_weight`, sorted lexically; I verified the 40 are exactly layers 0-19 x 2 by count and the safetensors dtypes below)
  - `weight_transform: mxfp8_nearest`, `compute_dtype: mxfp8`, `weight_block_shape [32,32]`, scale ue8m0: 8 tensors = layers 20-23 `w13_weight`, `w2_weight`
  - bf16 with `fwd_output_dtype: float32`, 1 tensor: `model.llm.unembed.weight`
- Safetensors header (Mini): layer 0/6/19 `mlp.experts.w13_weight` U8 [128,2304,1152], `w2_weight` U8 [128,2304,576], `_weight_scale` F8_E4M3 [128,2304,144] / [128,2304,72], `_weight_amax` F32 [128], `_weight_outer_scale_denominator` F32 [1] (NVFP4). Layer 20/23 `w13_weight` F8_E4M3 [128,2304,2304], `w2_weight` F8_E4M3 [128,2304,1152], `_weight_scale` U8 [128,2304,72]/[.,.,36] (MXFP8, ue8m0 scales). Only expert tensors are 4-/8-bit; everything else bf16/fp32.
- Confirmed by the converter too (conversion/inkling2.py, `dequant_model`): only names ending `.w13_weight`/`.w2_weight` are treated as packed.

## Q3 llama.cpp: PARTLY / important correction

- PR #32217 "feat: add Inkling 2 support (text + vision)": MERGED 2026-11-14T23:51:32Z into master, merge commit `5c008e00085f4d3e394f9a1be649284999de1326`, which is exactly tag `b12406`. So "build >= b12406" is correct for #32217.
- BUT #32217 does NOT support Mini. At 5c008e0 the converter accepts only `model_type` in ("inkling2","inkling2_text") (conversion/inkling2.py L72 at that commit) and `src/models/inkling2.cpp` has a plain `for (int il = 0; il < n_layer; ++il)` with no loop/pass code (grep "loop|pass|core" finds nothing). Mini is `Inkling2LoopForConditionalGeneration` / `model_type: inkling2_loop`.
- Mini support is PR #32694 "model : Inkling 2 Mini" (body: "Supports https://huggingface.co/thinkingmachines/Inkling-2-Mini"): MERGED 2026-11-20T22:01:08Z, merge commit `29b52d398b294509746ec0a3f28948a3107c0213` = tag `b12600` (b12599 = e3c3f66 is the commit before it). Minimum build for Mini: b12600. (Latest GitHub "release" object is still v1.0.0; the bNNNN tags are plain git tags. master HEAD at check time was 29b52d3.) Anything between b12406 and b12599 cannot load Mini.
- GGUF arch string: `inkling2_loop` (gguf-py/gguf/constants.py L1803 `MODEL_ARCH.INKLING2_LOOP: "inkling2_loop"`).
- Keys (src/llama-arch.cpp @29b52d3 L289-293):
  `{ LLM_KV_LOOP_START, "%s.loop.start" }`, `LOOP_END "%s.loop.end"`, `LOOP_COUNT "%s.loop.count"`, `LOOP_GATE_RANK "%s.loop.gate_rank"`, `LOOP_VARYING "%s.loop.varying"`.
  So the pass-count key is `inkling2_loop.loop.count`, end is `inkling2_loop.loop.end` (stored as 18, half-open).
- Read at load time: src/models/inkling2-loop.cpp L10-14 `ml.get_key(LLM_KV_LOOP_START, core_start); ... ml.get_key(LLM_KV_LOOP_COUNT, n_passes); ...`. Cache slots are derived from it (L24 `const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);`, L30 `hparams.n_layer_all = n_slot;`, "expand to the cache slots before load_tensors() allocates the layers and the memory"). Loader override path: src/llama-model-loader.cpp L423-428 `get_key` does `kv_overrides.find(key)` and passes the override into `GKV<T>::set`; integer override applies via `try_override` with `LLAMA_KV_OVERRIDE_TYPE_INT`. Since everything is derived after get_key in load_arch_hparams, `--override-kv inkling2_loop.loop.count=int:N` should change the pass count at runtime without reconversion (INFERENCE from the code; I did not run it, no weights downloaded). Limits: N >= 2 for the fixed-loop Mini (error otherwise at L19), N >= 1 for varloop; n_slot <= LLAMA_MAX_LAYERS. Loop-control tensors are per layer, not per pass (`loop_layers` indexed by physical layer), so N>3 is structurally possible but untrained.
  Same for start/end, but they must be multiples of 6 (L20).
- Converter writes it: conversion/inkling2.py `Inkling2LoopModel.set_gguf_parameters` (L302-307): `self.gguf_writer.add_loop_start(self.loop["core_start"])`, `add_loop_end(self.loop["core_end"])`, `add_loop_count(self.loop["num_passes"])`, `add_loop_gate_rank(...)`, `add_loop_varying(self.loop["is_varloop"])`. It registers `Inkling2LoopForConditionalGeneration` (and `...ForCausalLM`). It reads `hparams["loop_config"]` and rejects any key set other than {core_start, core_end, num_passes, gate_rank, is_varloop}. Varloop converts the same way (is_varloop -> `loop.varying`).
- Expert tensor names: `MODEL_TENSOR.FFN_GATE_UP_EXP: "blk.{bid}.ffn_gate_up_exps"` (constants.py L1898); down = `blk.N.ffn_down_exps`. Converter maps `w13_weight` -> gate_up (gate/up interleaved halves handled in chunking) and `w2_weight` -> down. Loop tensors: `blk.N.loop_msg_query`, `loop_{q,k,v,ffn}_ctrl_{down,gain,forget}` for N in 6..17, always kept F32 (converter `tensor_force_quant` and llama-quant.cpp L344-351 `quantize &= name.find(".loop_") == npos`).
- NVFP4 in GGUF: the converter does NOT passthrough. `dequant_model` dequantizes NVFP4/MXFP8 experts to F32 (`gguf.LazyChunkedTensor(chunks, shape, np.float32)`), then the usual --outtype / llama-quantize decides the type. GGUF type enum has `NVFP4 = 40` and `MXFP4 = 39` (constants.py L7337-7338), and file types `MOSTLY_NVFP4 = 39`. So NVFP4 experts in a GGUF exist only if you quantize to NVFP4 (llama-quantize), not from the converter. I did not read GGUF headers of Unsloth's files to see their per-tensor types.
- Related open bug (relevant to any NVFP4 study): PR #32704 (OPEN, 2026-11-20T23:59Z) "model : inkling2_loop re-share the loop slots after the scale pass": "copies each physical layer into its loop slots before the scale tensors are attached. The extra slots therefore have no scales, so the 2nd and 3rd passes ... run the NVFP4 experts with neither per-expert scale. PPL for NVFP4 with scales (eg unsloth's) for inkling-2-mini are 130.43 (should be 7.6531)". Code cause visible at inkling2-loop.cpp L65-67 (`layers[il] = layers[layer_phys(il)];` in load_arch_tensors) with scales (`ffn_*_exps_s`, PRs #32657/#32669) attached afterwards. Pin to a build with #32704 merged (not merged at check time) or treat NVFP4 results from b12600 as suspect. Also open: #32669 "apply .scale tensors automatically".

## Q4 imatrix and looped tensors: CONFIRMED accumulate (sum), same names reused

tools/imatrix/imatrix.cpp @29b52d3, `IMatrixCollector::collect_imatrix` (L560):
- name matching L565: `std::string wname = filter_tensor_name(src0->name);` (strips backend prefix/suffix: `CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight`, L94-110). Key is the weight tensor's own name.
- Dense path L688: `auto & e = m_stats[wname];` then L724-731 `float * acc = e.values.data() + mat_id * ne0; ... acc[j] += x[j] * x[j]; ... act[j] += x[j];` and L742-743 `e.counts[i] += ggml_nrows(src1) / n_mat;`
- MoE path (MUL_MAT_ID, this is what the experts use) L621 `auto & e = m_stats[wname];`, L654-657 `float * acc = e.values.data() + ex * ne0; ... e.counts[ex]++; ... acc[j] += x[j] * x[j];`
No overwrite and no "first visit only"; every evaluation of the op adds into the same map entry, and counts are incremented per row, so the stored value (values/counts, a mean of squares) is the pooled mean over all 3 visits (tokens x passes). Per-pass statistics cannot be separated.
- Graph naming: the Inkling2 loop reuses the same weight tensor objects. inkling2-loop.cpp L65-67 `layers[il] = layers[layer_phys(il)];` copies the struct of tensor pointers, and `build_layer(il, ip, ...)` takes `layer = model.layers[il]` (inkling2.cpp L277) and passes `layer.ffn_gate_up_exps`, `layer.ffn_down_exps` to `build_moe_ffn`. Only one set of tensors named `blk.6..17.*` exists in the GGUF; there are no per-visit tensor names. Only the `cb(...)` activation names carry the slot index `il` (e.g. `l_out`), not the weights. So the imatrix has one entry per physical weight and layers 6-17 receive 3x the rows.
- Caveats: collector only takes MUL_MAT with src1 rows >= 16 and F32 src1 (L570-578). Also the 2nd/3rd passes only get correct NVFP4 scale behaviour once #32704 is in (see Q3), otherwise imatrix on an NVFP4 GGUF would be collected under broken activations; collect on a BF16/F16-or-Q8 GGUF.

## Unsloth card: CONFIRMED

https://huggingface.co/unsloth/Inkling-2-Mini-GGUF README, "Quantization Analysis" table (columns: `Quant | Size (GB) | Mean KLD vs UD-Q8_K_XL | Same top token`):
- `| UD-Q2_K_XL | 9.14 | 0.0595 | 91.6% |`
- `| UD-IQ3_XXS | 11.66 | 0.0269 | 94.7% |`
(others: IQ1_S 7.76/0.0911, IQ1_M 8.02/0.0867, IQ2_XXS 8.49/0.0726, Q3_K_XL 13.63/0.0174, IQ4_XS 15.73/0.0109, Q4_K_M 16.61/0.0108, Q4_K_XL 17.63/0.0107, Q8_K_XL 22.87/reference.)
HF API byte sizes: `Inkling-2-Mini-UD-Q2_K_XL.gguf` 9138715328 B (=9.14 GB decimal), `Inkling-2-Mini-UD-IQ3_XXS.gguf` 11663309504 B (=11.66 GB decimal). Files are single (unsharded) .gguf.
Caveat: the KLD reference is UD-Q8_K_XL (22.87 GB), NOT the BF16/original checkpoint, so these KLDs understate distance from the true model.
The card does not say which llama.cpp build to use (it says only "Run the model via Unsloth Desktop or llama.cpp").

## Unsloth GGUF header (read via HTTP Range, first 30 MB of Inkling-2-Mini-UD-Q2_K_XL.gguf; header only, no weights): SCHEMA DIFFERS FROM MERGED MASTER

Parsed KV + tensor-info section of https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-UD-Q2_K_XL.gguf:
- `general.architecture = inkling2` (NOT `inkling2_loop`), `inkling2.block_count = 48`, but tensors exist only for blk.0..blk.23 (734 tensors, 62 KV). `general.file_type = 10` (MOSTLY_Q2_K).
- Loop metadata: `inkling2.num_loops = 3`, `inkling2.loop.core_start = 6`, `inkling2.loop.core_end = 18`, `inkling2.loop.gate_rank = 64`. No `loop.count`, `loop.start`, `loop.end`, `loop.varying`.
- Tensor names: `blk.6.loop_attn_down.weight`, `loop_attn_gain`, `loop_attn_forget`, `loop_{k,v,mlp}_{down,gain,forget}`, global `loop_res_query.weight`, `loop_kv_res_query.weight`; experts `blk.N.ffn_gate_up_exps.weight` + `blk.N.ffn_gate_up_exps.scale` [128] (F32) and `blk.N.ffn_down_exps.weight/.scale`.
- Expert tensor GGML types in this file (ffn_gate_up_exps + ffn_down_exps, 76 tensors, id: count): 0 (the `.scale` F32 tensors) 40; 42 (Q2_0 per gguf-py constants.py) 12; 40 (NVFP4) 11; 22 (IQ2_S) 11; 19 (IQ1_S) 6; 16 (IQ2_XXS) 5; 17 (IQ2_XS) 2; 39 (MXFP4) 1 (ids as in ggml's standard enum; 39/40/42 confirmed in gguf-py constants.py, others from ggml convention, not re-verified). For the .weight tensors only, e.g. blk.0 gate_up = type 19, down = type 42; blk.20 gate_up = 22, down = 39; blk.6 gate_up 19. So NVFP4 (type 40) IS used for some expert tensors in this Q2 file, mixed with Q2_0/IQ types; it is not uniform.
- This matches the schema of PR #32695 "model: Inkling-2-Mini - looped arch" (OPEN, author danielhanchen, 2026-11-20T20:17Z): its src/llama-arch.cpp patch adds `"%s.loop.core_start"`, `"%s.loop.core_end"`, `"%s.loop.gate_rank"` and tensors `blk.%d.loop_attn_down` etc., and keeps the existing `"%s.num_loops"`. It is NOT the merged #32694 schema (`inkling2_loop`, `loop.start/end/count/varying`, `loop_q_ctrl_*`).
- Consequence (INFERENCE, not tested): these Unsloth GGUFs likely do not load on master b12600 (different arch string, key set and tensor names), and the `--override-kv` key for them would be `inkling2.num_loops=int:N`, not `inkling2_loop.loop.count`. The correct key depends on which llama.cpp build/branch (PR #32695 vs master #32694) you run; check the GGUF's own keys. If you convert yourself with master's convert_hf_to_gguf.py you get the `inkling2_loop.*` schema. Whether #32695's code reads `num_loops` at load time I did not verify (PR diff not read beyond arch.cpp keys).
