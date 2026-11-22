# NVFP4 on Metal: end-to-end parity (Addendum H amendment 8)

Written by `scripts/nvfp4-parity.py` from the logs named below. Nothing in the release or the thesis depends on this build; it informs the upstream decision.

## Builds

Worktree `vendor/llama.cpp-nvfp4`, branch `nvfp4-metal` (`git log --oneline -3`):

```
436898eed tests : NVFP4 cases with every scale byte and k = 1152
ed4cafa05 metal : add NVFP4 get_rows, mul_mv, mul_mv_id, mul_mm and mul_mm_id
9f5b9d6b7 convert : inkling2 decode each packed expert tensor once
```

Build command (existing build dir, Release, `GGML_METAL=ON`):

```
cd vendor/llama.cpp-nvfp4 && nice -n 10 cmake --build build --target llama-perplexity -j 4   # cmake 4.5.1 from PATH; build dir Release, GGML_METAL=ON (CMakeCache)
```

Measurement build: `vendor/llama.cpp` (pinned, patches/llama.cpp), run of `scripts/map3.sh`.

Both KL runs: `llama-perplexity -m models/UD-IQ3_XXS.gguf -f data/S.txt -c 4096 --chunks 16 -ngl 99 --override-kv inkling2_loop.loop.count=int:3 --kl-divergence-base models/logits/ref-S-P3.kld --kl-divergence`.

## Where the expert tensors live

From the load-only runs at `-lv 5` (the KL logs are at default verbosity and print no buffer lines): `<build>/bin/llama-perplexity -m models/UD-IQ3_XXS.gguf -f <first 2,000 bytes of data/S.txt> -c 4096 -ngl 99 --override-kv inkling2_loop.loop.count=int:3 -lv 5`, which loads, warms up and stops at 'you need at least 8192 tokens'. Logs: `logs/load-UD-IQ3_XXS-measurement.log`, `logs/load-UD-IQ3_XXS-nvfp4metal.log`.

| | measurement build | NVFP4-Metal build |
|---|---|---|
| tensors moved off their preferred buffer (loader debug line) | 21 tensors (q4_K, first `token_embd.weight`) moved from CPU_REPACK to CPU | 1 tensor (q4_K, first `token_embd.weight`) moved from CPU_REPACK to CPU |
| CPU model buffers (non-empty) | 21 buffers, 3893.46 MiB | 1 buffer, 248.46 MiB |
| MTL0 model buffers (non-empty) | 22 buffers, 7218.13 MiB | 2 buffers, 10863.12 MiB |
| CPU buffers of exactly one NVFP4 tensor (182.25 MiB) | 20 | 0 |
| layers assigned to device CPU | 0 | 0 |
| layer/device mismatch warnings in the KL log | 0 | 0 |

From the GGUF: 20 NVFP4 tensors (`blk.0.ffn_down_exps.weight` to `blk.19.ffn_down_exps.weight`), 182.25 MiB each; `token_embd.weight` (Q4_K, 248.46 MiB) sits on the CPU in every build (input embedding). The loader's debug line names only the first moved tensor and counts the rest. MTL0 total, NVFP4-Metal minus measurement: 3644.99 MiB (= 20.000 NVFP4 tensors).

NVFP4 tensor still on the CPU in the NVFP4-Metal build: no.

## Per-chunk KL on S, P = 3

Per-chunk values telescoped from the running means (`analysis/maptable.parse`); difference = NVFP4-Metal minus measurement. Rounding bound: (2k - 1) * 5e-6 per run, summed over both runs.

| chunk | measurement | NVFP4-Metal | difference | rounding bound | within 2e-4 |
|---:|---:|---:|---:|---:|---|
| 1 | 0.018570 | 0.018520 | -0.000050 | 0.000010 | yes |
| 2 | 0.028110 | 0.027220 | -0.000890 | 0.000030 | NO |
| 3 | 0.028800 | 0.030610 | +0.001810 | 0.000050 | NO |
| 4 | 0.022640 | 0.023090 | +0.000450 | 0.000070 | NO |
| 5 | 0.022130 | 0.022410 | +0.000280 | 0.000090 | NO |
| 6 | 0.020450 | 0.020410 | -0.000040 | 0.000110 | yes |
| 7 | 0.016590 | 0.016640 | +0.000050 | 0.000130 | yes |
| 8 | 0.018790 | 0.018300 | -0.000490 | 0.000150 | NO |
| 9 | 0.017330 | 0.018190 | +0.000860 | 0.000170 | NO |
| 10 | 0.018890 | 0.017810 | -0.001080 | 0.000190 | NO |
| 11 | 0.024090 | 0.023960 | -0.000130 | 0.000210 | yes |
| 12 | 0.018370 | 0.018320 | -0.000050 | 0.000230 | yes |
| 13 | 0.019670 | 0.019730 | +0.000060 | 0.000250 | yes |
| 14 | 0.018310 | 0.018510 | +0.000200 | 0.000270 | yes |
| 15 | 0.015060 | 0.015280 | +0.000220 | 0.000290 | NO |
| 16 | 0.020200 | 0.020280 | +0.000080 | 0.000310 | yes |

| | measurement | NVFP4-Metal |
|---|---:|---:|
| Mean KLD (log) | 0.020503 ± 0.000286 | 0.020582 ± 0.000310 |
| Same top p (%) | 94.263 | 94.181 |

- Max |per-chunk difference|: 0.001810 (chunk 3); tolerance 0.0002.
- Mean per-chunk difference: +0.000080, paired 95% bootstrap CI [-0.000217, +0.000403] (`analysis/compare.paired`, 10,000 resamples, seed 0).
- Mean KLD difference from the logs' 6-decimal means: +0.000079; tolerance 0.0001.

## Verdict

FAIL under the amendment 8 rule: every chunk within 2e-4: no (8 of 16 chunks outside); mean within 1e-4: yes.

Not a conclusion about the kernel, only context: the two runs use different numerics by design. The CPU path multiplies NVFP4 against Q8_0-quantized activations (`ggml-cpu.c`, `vec_dot_type = GGML_TYPE_Q8_0`), Metal against f32 (`mul_mv`) or f16 tiles (`mul_mm`). The rule was set without a measured run-to-run floor for such a change, and the signed per-chunk differences scatter around zero (paired CI above includes 0).

## Timings (NOT a speed measurement)

Printed by llama-perplexity. The machine was shared with other jobs during both runs, so these numbers say nothing about kernel speed. Speed waits for an idle machine (amendment 8).

- measurement (`ppl-UD-IQ3_XXS-S-P3.log`): 57.21 seconds per pass - ETA 15.25 minutes
- NVFP4-Metal (`ppl-UD-IQ3_XXS-nvfp4metal-S-P3.log`): 18.94 seconds per pass - ETA 5.05 minutes
