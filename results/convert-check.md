# BF16 conversion check

Written by hand from `scripts/compare-remote.py` output, 21 Nov 2026.

- Converter: `convert_hf_to_gguf.py` at llama.cpp `29b52d3` with patch 0007 (each packed expert tensor decoded once instead of once per 128-row chunk). Conversion time 9 min 47 s, against an estimated 2.4 h without the patch.
- Reference: ggml-org/Inkling-2-Mini-GGUF `Inkling-2-Mini-BF16.gguf` at revision `db61c5b`, header fetched by range request.
- Compared the first, a middle and the last MiB of: `blk.0.ffn_down_exps.weight` (NVFP4 source), `blk.6.ffn_gate_up_exps.weight` (NVFP4, interleaved gate/up rows), `blk.20.ffn_gate_up_exps.weight` and `blk.23.ffn_down_exps.weight` (MXFP8 source), `token_embd.weight`. All identical, with identical types and shapes.
- Shape of `ffn_down_exps.weight`: [1152, 2304, 128] (ne0 = 1152), so 256-block quant types fall back on the down projections.
