# Unsloth prebuild b12588-mix: where UD-IQ2_XXS (root GGUF) loads

Excerpt of `logs/load-UD-IQ2_XXS-root-unsloth-b12588.log` (local only; paths removed). Binary: `llama-b12588-mix-c0c3de8-bin-macos-arm64.tar.gz` sha256 `229def18e27ed740ec4f3a6508f084b25630e08d8a87ada40b00622fc496ceb8`, `llama-perplexity -m <UD-IQ2_XXS root GGUF> -c 4096 -ngl 99 -lv 5` on a 2,000-byte text (loads, then stops before evaluating).

```
0.03.757.015 I load_tensors:   CPU_Mapped model buffer size =   234.65 MiB
0.03.757.016 I load_tensors:  MTL0_Mapped model buffer size =   235.09 MiB
0.03.757.016 I load_tensors:  MTL0_Mapped model buffer size =  7611.93 MiB
0.00.181.057 I llama_model_loader: - kv   0:                       general.architecture str              = inkling2
0.00.181.094 I llama_model_loader: - type nvfp4:    7 tensors
0.03.926.760 D ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_mul_mv_id_nvfp4_f32', name = 'kernel_mul_mv_id_nvfp4_f32_nsg=2_split=0'
0.03.992.348 D ggml_metal_library_compile_pipeline: loaded kernel_mul_mv_id_nvfp4_f32_nsg=2_split=0      0x810325880 | th_max = 1024 | th_width =   32
0.04.042.993 E perplexity: you need at least 8192 tokens to evaluate perplexity with a context of 4096
```
