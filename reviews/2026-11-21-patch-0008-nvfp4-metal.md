# Patch 0008: NVFP4 on Metal (spike, 2026-11-21)

This is a spike for review, and nothing depends on it. It lives in the worktree `vendor/llama.cpp-nvfp4` (branch `nvfp4-metal`, two commits on top of the pinned patched HEAD `9f5b9d6b7`) and is exported to `patches/llama.cpp-proposed/`. The measurement build (`vendor/llama.cpp`, `patches/llama.cpp/`) is untouched, and the GPU chain kept running on it the whole time.

Every count below comes from `reviews/2026-11-21-patch-0008-nvfp4-metal/counts.py`, which reads the `test-backend-ops` logs and the patch files in that folder and writes `counts.txt`. The machine state behind the skipped smoke test is in `smoke-skip.txt`.

## Files

- `patches/llama.cpp-proposed/0008-metal-add-NVFP4-get_rows-mul_mv-mul_mv_id-mul_mm-and.patch` holds the Metal backend change.
- `patches/llama.cpp-proposed/0009-tests-NVFP4-cases-with-every-scale-byte-and-k-1152.patch` holds the new `test-backend-ops` cases.
- Both are by the repository's git identity (me-daniel), with subject lines only and no attribution lines. They apply with `git apply` to `9f5b9d6b7` in a clean detached worktree. 0008 also passes `git apply --check` on upstream `c661033`. 0009 does not, because `tests/test-backend-ops.cpp` has drifted there.
- Build: `vendor/llama.cpp-nvfp4/build` (Release, `GGML_METAL=ON`, cmake 4.4.5 via uvx, target `test-backend-ops` only, `nice -n 10`).

## What changed

Patch 0008 adds 168 lines and removes 3, over 9 files.

| file | added | removed | where (new-file line of the hunk) | what |
|---|---:|---:|---|---|
| `kernels/common.h` | 10 | 0 | 68 | `ue4m3_to_fp32`, a bit decode of the UE4M3 scale |
| `kernels/dequantize.h` | 38 | 0 | 378 | `dequantize_nvfp4` (nl = 4, one 16-value sub-block per call) and `dequantize_nvfp4_t4` |
| `kernels/mul_mv.metal` | 100 | 0 | 894, 3151, 3511 | `kernel_mul_mv_nvfp4_f32_impl` and its host kernel, 4 `mul_mv_ext` lines, 1 `mul_mv_id` line |
| `kernels/mul_mm.metal` | 4 | 0 | 906, 933, 973, 1004 | `mul_mm` and `mul_mm_id`, f32 and f16 src1, half tiles |
| `kernels/quantize.metal` | 1 | 0 | 341 | `get_rows` |
| `ggml-metal-device.cpp` | 10 | 0 | 1104, 1404 | NVFP4 case in the `mul_mv` and `mul_mv_id` pipeline selectors (nsg 2, nr0 2, no smem) |
| `ggml-metal-device.m` | 1 | 3 | 1529, 1988 | NVFP4 dropped from the exclusions in `ggml_metal_supports_mul_mat_op` and `GET_ROWS` |
| `ggml-metal-impl.h` | 3 | 0 | 50 | `N_R0_NVFP4`, `N_SG_NVFP4` |
| `ggml-metal-ops.cpp` | 1 | 0 | 2775 | NVFP4 in the small-batch `mul_mv_ext` list |

Patch 0009 adds 71 lines to `tests/test-backend-ops.cpp`.
- `test_get_rows_nvfp4_raw`, `test_mul_mat_nvfp4_raw` and `test_mul_mat_id_nvfp4_raw` fill the NVFP4 tensor with raw bytes instead of quantizing floats. Scale byte j is `(167 j + 13) mod 256` over the sub-block index j, so any 256 consecutive sub-blocks carry every byte value 0x00 to 0xFF once. The quant bytes are random. The parameter string gets `raw=1`.
- The GET_ROWS raw cases use rows of 4096 values (256 sub-blocks), so every fetched row holds all 256 scale bytes. GET_ROWS allows an NMSE of only 1e-7, so these two cases check the decode against the CPU for every byte.
- New shapes at k = 1152, the width of our `ffn_down_exps` rows. MUL_MAT at m = 64 with n = 1, 3, 8, 64 covers `mul_mv`, `mul_mv_ext` and `mul_mm`. MUL_MAT_ID with 8 experts, 4 used, m = 128 and n = 1, 4, 64 covers `mul_mv_id` and `mul_mm_id`. Each shape runs once with quantized data and once raw.

### Design choices

- Scale decode: no table. `ue4m3_to_fp32` builds the float from the exponent and mantissa bits and copies the CPU's special cases (0x00 and 0x7F give 0, bit 7 is ignored). A byte of 0x80 or more is therefore decoded the way the CPU decodes it. This replaces PR #20456's 128-entry table, which reads past its end for such bytes.
- The CPU's `ggml_ue4m3_to_fp32` returns half the true scale because the CPU E2M1 table holds doubled int8 values. Metal reuses `kvalues_mxfp4_f`, which holds the true E2M1 values, so the Metal decode returns twice the CPU value. My first build got this wrong, and every NVFP4 case failed until I fixed it. A small debug program (`debug-get-rows.cpp`, output in `debug-get-rows.txt`) compared one hand-made block on Metal and the CPU. After the fix, all 64 values match.
- `mul_mv`: one thread per 16-value sub-block, so 8 blocks per simdgroup step. A 36-byte block starts its quant bytes at offset 4, so each thread loads two aligned `uint32` words and reuses MXFP4's table-free E2M1 decode (`mxfp4_lo4`, `mxfp4_hi4`). For k = 1152 this layout uses 0.7500 of the lanes, against 0.5625 for the lane-pair-per-block layout of the MXFP4 kernel and #20456 (arithmetic, not a measurement).
- `mul_mm_id` uses half tiles only, with no bfloat variant. The largest decoded scale over all 256 bytes is 480, so the largest weight is 2880, far below the fp16 maximum of 65504. The smallest non-zero weight is 0.0009765625, far above the smallest fp16 subnormal. NVFP4 weights are exact in half.
- The few-row MMA path (`mul_mv_mma.metal`) is left out. NVFP4 is not in `ggml_metal_mul_mv_mma_type_supported`, so the old `mul_mv` and `mul_mv_ext` paths serve it.

## test-backend-ops, before and after

All runs are `test-backend-ops test -b MTL0 -o <op> -p nvfp4` (Metal compared against the CPU). "Before" means the Metal sources of `9f5b9d6b7` and "after" means patch 0008. The middle column uses the new test binary (0009) on the old Metal sources, to show that the new cases also skip without the patch.

| op | old tests, old Metal | new tests, old Metal | new tests, patched Metal |
|---|---|---|---|
| GET_ROWS | 8 not supported | 10 not supported, 0/0 run | 10 OK, 10/10 passed |
| MUL_MAT | 52 not supported (plus 36 not supported on the CPU either) | 60 not supported (plus 36), 0/0 run | 60 OK, 60/60 passed |
| MUL_MAT_ID | 73 not supported | 79 not supported, 0/0 run | 79 OK, 79/79 passed |

The 36 MUL_MAT cases that both backends reject before the patch use f16 src1, which the CPU does not support with NVFP4. After the patch Metal takes 13 of them (they show as "not supported [CPU]" and are not compared), and 23 stay unsupported on both.

All 16 new cases pass on patched Metal (2 raw GET_ROWS, 4 quantized and 4 raw MUL_MAT, 3 quantized and 3 raw MUL_MAT_ID).

The patched runs compiled `kernel_get_rows_nvfp4`, `kernel_mul_mv_nvfp4_f32`, `kernel_mul_mv_ext_nvfp4_f32_r1_2` to `_r1_5`, `kernel_mul_mv_id_nvfp4_f32`, `kernel_mul_mm_nvfp4_f32` and `kernel_mul_mm_id_nvfp4_f32`. The two f16-src1 `mul_mm` kernels compile but never run against a reference.

### Do the raw cases catch the 128-table class of bug?

I changed the decode for one build so that every byte of 0x80 or more returns 0 (in the spirit of a table that stops at 128), ran the same three commands, then restored the file and rebuilt. Logs are in `mutant-*.txt`.

| op | raw cases FAIL | raw cases OK | other cases FAIL | other cases OK |
|---|---:|---:|---:|---:|
| GET_ROWS | 2 | 0 | 0 | 8 |
| MUL_MAT | 4 | 0 | 0 | 56 |
| MUL_MAT_ID | 3 | 0 | 0 | 76 |

Every raw case fails and every quantized case passes. The upstream cases fill tensors through quantization, which never writes a scale byte of 0x80 or more, so they cannot see this bug.

### Wider runs on the patched build

- `-p nvfp4` over all ops (`after-all-ops.txt`), and every case that runs passes. That includes 232 MUL_MAT_FUSION cases (gate, up and GLU, some with per-expert weight scales), 6 MUL_MAT_FUSION_W4A8, 3 MUL_MAT_W4A8, 2 MUL_MAT_W4A4, 2 MUL_MAT_ID_W4A8 and 2 MUL_MAT_ID_W4A4. Before the patch all of these were unsupported on Metal (`before-all-ops.txt`).
- Still unsupported on Metal after the patch: 17 CPY, 25 SET_ROWS and 16 OUT_PROD cases (out of scope here).
- Full regression without the filter, every type: GET_ROWS 234/234, MUL_MAT 2672/2672, MUL_MAT_ID 1198/1198 passed (`after-full-*.txt`).

## Smoke test

Not run. The task allowed a 2-chunk `llama-perplexity` of `models/UD-IQ3_XXS.gguf` if it finished in under 2 minutes. When I checked (`smoke-skip.txt`), chain7 was running `llama-perplexity` on `models/tmp/mirror.gguf` with 13.2 GB resident, the machine had 16% of its 48 GB free, and the UD file is 11.7 GB. A second model load would have been a second heavy run next to the chain, which the workspace rule (one heavy run at a time) forbids, and it could have pushed the chain into swap. It needs a quiet machine. To run it later, build `llama-perplexity` in the worktree first, as below.

```sh
cd vendor/llama.cpp-nvfp4
nice -n 10 uvx --from cmake==4.4.5 cmake --build build --target llama-perplexity -j 8
build/bin/llama-perplexity -m ../../models/UD-IQ3_XXS.gguf -f ../../data/S.txt --chunks 2 -c 512
```

Two things to check there. The scheduler log should show no CPU fallback for NVFP4 (the 20 `ffn_down_exps` tensors should land on MTL0), and the PPL should be close to the CPU-fallback value at the same settings. The two will not be bit-identical, because the CPU quantizes activations to Q8_0 while Metal keeps f32 in `mul_mv` and f16 tiles in `mul_mm`.

## Known limits

- Only GET_ROWS, MUL_MAT and MUL_MAT_ID. CPY (NVFP4 to and from float), SET_ROWS and OUT_PROD stay on the CPU. Inference on our files only needs the first three.
- The f16-src1 kernels (`kernel_mul_mm_nvfp4_f16`, `kernel_mul_mm_id_nvfp4_f16`) have no reference test, because the CPU backend has no NVFP4 x f16 path. They are the same template as the tested f32 versions with a different output type.
- `mul_mv` assumes the NVFP4 data starts on a 4-byte boundary. Whole tensors and row or batch views do, since 36 is a multiple of 4. A view at an odd byte offset into NVFP4 data would read misaligned words. No current test or model makes such a view.
- No speed measurement. The machine was busy, and the task excluded llama-bench. Whether `mul_mv` at k = 1152 needs more work is open.
- The register-spill check that ggerganov asked for on #20456 (#20399) was not done. It needs a Mac Studio and Xcode GPU counters.
- Upstream PR #28898 (weight scales moved into ggml) would turn off Metal for any op with a src scale. If it lands first, this patch needs a scale epilogue to keep scaled NVFP4 on Metal.
- Not covered end to end: `test-llama-archs` with an NVFP4 random model, and the KLD of `UD-IQ3_XXS` on Metal against its CPU-fallback number. Both wait for a free GPU.

## What Daniel must review

1. The diff of 0008, mainly `ue4m3_to_fp32` (the factor of 2 against the CPU function and the special cases), the index math in `dequantize_nvfp4` and `_t4` (low nibbles are values 0 to 7 of a sub-block and high nibbles 8 to 15, as in `dequantize_row_nvfp4`), and the thread layout of `kernel_mul_mv_nvfp4_f32_impl`.
2. The test design in 0009: whether raw scale bytes of 0x80 or more belong in an upstream test, given that valid files never contain them.
3. Adoption. Moving 0008 and 0009 into `patches/llama.cpp/` changes the pinned measurement build. Every speed number and every result that offloads NVFP4 would then run on different code, so it needs a dated CHANGELOG addendum before any data, and it has to wait until chain7 and anything else running on `vendor/llama.cpp` are done. `scripts/build-llama.sh` would rebuild with the new patches, and `results/env.md` would record their hashes.
4. What it means for the release track. The feasibility review already notes that `UD-IQ3_XXS` would then run fully on Metal, which removes the candidate's measured speed edge (Addendum G), and that the Addendum C native-down build becomes Metal-runnable. Both need Daniel's call.
5. Upstream. llama.cpp's `AGENTS.md` requires that the contributor understands and can defend every line, that commit messages and the PR description are written by the human, and that AI help is disclosed in the PR template. The commit messages here are placeholders. A PR would need Daniel's own text, a rebase of 0009 onto current `test-backend-ops.cpp`, and credit to #20456. Whether to open one is Daniel's decision, and nothing has been pushed or posted.

## Reproduce

```sh
cd vendor/llama.cpp-nvfp4
nice -n 10 uvx --from cmake==4.4.5 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF
nice -n 10 uvx --from cmake==4.4.5 cmake --build build --target test-backend-ops -j 8
for op in GET_ROWS MUL_MAT MUL_MAT_ID; do nice -n 10 build/bin/test-backend-ops test -b MTL0 -o $op -p nvfp4; done
cd ../.. && .venv/bin/python -I reviews/2026-11-21-patch-0008-nvfp4-metal/counts.py
```
