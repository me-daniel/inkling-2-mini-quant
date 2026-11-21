# Metal NVFP4 matmul in llama.cpp: feasibility (2026-11-21)

Reading only. Nothing was built or run on the GPU. Sources: upstream master `c661033` in `vendor/llama.cpp-upstream`, PR #20456 and PR #20060 (GitHub API and the raw diff), and this repo's bench files. Every number below is printed by `reviews/2026-11-21-metal-nvfp4-feasibility/facts.py` (output in `facts.txt` beside it; the PR JSON and diff it reads are in `pr/`). Sentences marked **inference** are judgement, not measurement.

## Verdict

Feasible, and small. The port is about 200 to 250 lines (**inference**), almost all of it copied from MXFP4. PR #20456 does not rebase. Its shader file no longer exists, so the kernels have to be moved by hand into the split `kernels/` files. One real bug has to be fixed on the way: a 128-entry scale table indexed with the raw byte. The per-expert F32 `.scale` needs no kernel work because it is a separate graph op that Metal already runs. Expected speed is close to IQ4_NL, since both are 4.5 bpw (**inference**).

The strategic catch: once NVFP4 runs on Metal, `UD-IQ3_XXS` no longer has 20 expert tensors on the CPU. The Mac speed edge from Addendum G then disappears (see the last section).

## What is missing today

- `ggml_metal_supports_mul_mat_op` (`ggml-metal-device.m`, around line 1532) returns false for `GGML_TYPE_NVFP4`, so MUL_MAT and MUL_MAT_ID fall back to the CPU. GET_ROWS (around line 1992) rejects NVFP4 too.
- There is no NVFP4 Metal kernel: no `dequantize_nvfp4`, no `kernel_mul_mv_nvfp4_f32`, no `mul_mm` or `get_rows` template.
- The pinned tree (`vendor/llama.cpp`, 9f5b9d6) and upstream `c661033` have byte-identical `ggml/src/ggml-metal/` directories (`diff -rq`). A patch written for one applies to the other.

## Format, next to MXFP4 and IQ4_NL

From `ggml-common.h`:

| type | block | bytes | bpw | block 4-byte aligned |
|---|---:|---:|---:|---|
| NVFP4 | 64 | 36 | 4.5000 | yes |
| MXFP4 | 32 | 17 | 4.2500 | no |
| IQ4_NL | 32 | 18 | 4.5000 | no |

In ggml, NVFP4 is a 64-element block: four UE4M3 bytes, one per 16-element sub-block, and then 32 bytes of E2M1 nibbles. In each sub-block the low nibbles hold elements 0..7 and the high nibbles hold 8..15. MXFP4 uses one E8M0 byte per 32 elements, and its nibble layout is split the same way.

How close the existing kernels are:
- The E2M1 values match. Metal's `kvalues_mxfp4_f` (`kernels/common.h`) can be reused unchanged. The CPU uses doubled int8 values and halves the scale in `ggml_ue4m3_to_fp32`, so the product comes out the same.
- Only the scale decode is new: UE4M3 instead of E8M0, four per block instead of one.
- `mul_mm`, `mul_mm_id`, `get_rows` and `mul_mv_ext` are templates over `(block type, nl, dequantize fn)`. NVFP4 needs one `dequantize_nvfp4` (nl = 4, 16 elements per call) and one `_t4` variant, and then template lines only.
- `mul_mv` needs its own kernel. MXFP4's current kernel (`kernel_mul_mv_mxfp4_f32_impl`) decodes E2M1 without a lookup table (`mxfp4_dq1`). It has to funnel-shift because its blocks are 17 bytes. NVFP4 blocks are 36 bytes, so every sub-block starts 4-byte aligned and plain `uint32` loads work. The MXFP4 kernel is a good template with the alignment trick removed (**inference**).
- The per-expert F32 scale of Unsloth-style files is not part of the type. `build_lora_mm_id` (`src/llama-graph.cpp`) applies it after the matmul as `ggml_get_rows` on F32 plus `ggml_mul`, and Metal already supports both. Patch 0001 still governs whether passes 2 and 3 see it.

## PR #20456 and #20060

- #20060: closed unmerged, draft, +707 -2 over 25 files. It was the combined proof of concept stacked on the CPU type PR #19769. ggerganov's comment there: "For now this is OK - we can improve this a bit after the base implementation is merged."
- #20456 "metal : add NVFP4 quantization support": open, not draft, +220 -2 over 5 files, `mergeable_state` dirty. Base `d8c331c0a`. Created 2026-03-12, last updated 2026-06-23. It has been open 253 days, and nothing has happened for the last 150.
- Reviews on #20456: ggerganov APPROVED on 2026-03-12. He also left an inline note on `ggml-metal-impl.h`: "Want to double check this on my Mac Studio to confirm no register spill (#20399)". The author asked about it on 2026-05-03 and got no reply. forforever73 (a Metal code owner) suggested an nsg-independent LUT load on 2026-06-23, and the author applied it the same day.
- Diff per file: `ggml-metal.metal` +202 in 11 hunks, `ggml-metal-device.cpp` +12, `ggml-metal-device.m` +2 -2, `ggml-metal-impl.h` +3, `ggml-metal-ops.cpp` +1.
- The PR contains: a constexpr UE4M3 table with 128 entries, `ue4m3_to_fp32`, `dequantize_nvfp4` and `_t4`, `kernel_mul_mv_nvfp4_f32` (E2M1 table plus UE4M3 table in threadgroup memory), `mul_mv_ext` r1_2..5, `mul_mv_id`, `mul_mm` and `mul_mm_id` in f32/f16 variants, and `get_rows`. It removes the two NVFP4 exclusions.
- It reports pp512 489 and tg128 25.3 t/s on Metal against 44.75 and 1.03 on the CPU, for an unnamed model. It gives no comparison with IQ4_NL.

### Does it rebase cleanly? No.

`git apply --check` against `c661033` (a dry run, so the tree is untouched):
- `ggml-metal.metal`: the file is gone. Upstream split it into `kernels/*.metal`, `dequantize.h` and `common.h`.
- `ggml-metal-device.m`: fails. The exclusion moved into the new `ggml_metal_supports_mul_mat_op` helper.
- `ggml-metal-device.cpp`: fails. The MXFP4 case no longer sets `smem`.
- `ggml-metal-impl.h` and `ggml-metal-ops.cpp` apply with offsets.

There are also semantic drifts a mechanical port would miss:
1. `mul_mv` kernels now read `FC_mul_mv_ne12/r2/r3` function constants. The PR uses `args.ne12/args.r2`, which no longer exist in `ggml_metal_kargs_mul_mv`.
2. `mul_mm_id` now has bfloat tile variants under `GGML_METAL_HAS_BF16` next to the half variants (MXFP4 has both). NVFP4 needs both names, or the BF16 build fails to find the pipeline. Half tiles are enough for the range: E2M1 times UE4M3 is at most 6 x 448 (**inference**).
3. There is a new few-row MMA path (`mul_mv_mma.metal`, and `ggml_metal_mul_mv_mma_type_supported` in `ggml-metal-common.cpp`). It is optional. Leaving NVFP4 out keeps the old path.
4. Precision: `build_lora_mm_id` sets `GGML_PREC_BF16` on the accumulator for NVFP4 weights. F32 accumulation satisfies that.

### Bug to fix in the port

The PR's UE4M3 table has 128 entries, and both the `constant` table and the threadgroup copy are indexed with the raw scale byte `xb.d[s]`. A byte of 0x80 or higher reads past the table. The CPU decoder masks bit 7 when it extracts the exponent and mantissa, but it tests the special values 0 and 0x7F on the full byte, so 0xFF decodes to a non-zero value. Vulkan had exactly this bug, fixed in #31079 (merged 2026-11-03) with a 256-entry table. Metal should use either a 256-entry table or a bit decode that copies the CPU's special cases. Valid files should never contain such bytes (**inference**). But `ggml_validate_row_data` accepts them, and `test-backend-ops` fills tensors with random data.

## Size of the port

| file | change |
|---|---|
| `kernels/common.h` | UE4M3 decode (256 table or bit decode) |
| `kernels/dequantize.h` | `dequantize_nvfp4`, `dequantize_nvfp4_t4` |
| `kernels/mul_mv.metal` | `kernel_mul_mv_nvfp4_f32_impl` + host kernel, 4 `mul_mv_ext` lines, 1 `mul_mv_id` line |
| `kernels/mul_mm.metal` | 2 `mul_mm` + 4 `mul_mm_id` lines (half and bfloat) |
| `kernels/quantize.metal` | 1 `get_rows` line |
| `ggml-metal-device.cpp` | NVFP4 case in the `mul_mv` and `mul_mv_id` pipeline selectors (nsg, nr0, smem) |
| `ggml-metal-device.m` | drop NVFP4 from the two exclusions |
| `ggml-metal-impl.h` | `N_R0_NVFP4`, `N_SG_NVFP4` |
| `ggml-metal-ops.cpp` | add NVFP4 to the small-batch `mul_mv_ext` list |

Most of the 202 shader lines in #20456 carry over. My estimate is 200 to 250 lines and a day of work plus testing (**inference**). It would be patch 0008, with a test, in the project's usual patch workflow.

## Tests that would prove it

- `test-backend-ops -b MTL0 -o MUL_MAT`, then `-o MUL_MAT_ID`, then `-o GET_ROWS`. NVFP4 is already in `all_types` and `base_types`, and the NVFP4 W4A8 and W4A4 cases exist (`test_mul_mat_w4a8`, `test_mul_mat_id_w4a8`, `_w4a4`). Once `supports_op` returns true they run against the CPU. The NMSE limit is 5e-4 for `mul_mat`. The Blackwell 2e-2 allowance does not apply to Metal.
- Add one shape case that matches our tensors: MUL_MAT_ID with NVFP4, k = 1152, several experts, n = 1, n = 2..8 (the `mul_mv_ext` gate, `ne00 % 128 == 0` is True for 1152) and n = 512 (`mul_mm_id`). The random cases use k = 256 or 512, which never hits a partial lane step.
- A scale-byte case with bytes 0x80..0xFF, to cover the bug above.
- In the project: `test-llama-archs -a 'inkling2.*'` (CPU vs Metal), and `--loop-scales -s 42` on an NVFP4 random model if the generator can emit one. Then a KLD of `UD-IQ3_XXS` on Metal against its CPU-fallback number. They should match within noise (**inference**). The CPU path quantizes activations to Q8_0 (`vec_dot_type` for NVFP4), while Metal keeps f32 in `mul_mv` and f16 tiles in `mul_mm`, so they will not be bit-identical. That is a GPU job, so it waits for the chain.

## Expected speed relative to IQ4_NL (inference)

Measured on this Mac (M5 Pro, build 9f5b9d6, `results/pilot/bench-*.txt`):

| build | pp512 t/s | tg128 t/s | expert tensors off Metal |
|---|---:|---:|---:|
| cand (IQ4_NL down x24) | 224.41 | 18.97 | 0 |
| uniform IQ3_XXS (IQ4_NL down) | 234.34 | 19.43 | 0 |
| UD-IQ3_XXS (NVFP4 down x20) | 68.06 | 10.87 | 20 |

UD-IQ3_XXS runs at 0.303 of the candidate's pp512 and 0.573 of its tg128.

The rest of this section is **inference**:
- NVFP4 and IQ4_NL both use 4.5 bpw, so token generation, which is bandwidth-bound, should land within about 10% of IQ4_NL.
- NVFP4 decodes four scale bytes per 64 elements, where IQ4_NL reads two half scales. That is extra ALU work, but small next to the memory traffic.
- One shape cost is specific to us. On a 1152-wide row, the `mul_mv` layout (16 lane pairs, one block per pair per step) uses 0.562 of its lanes for NVFP4 (18 blocks in 2 steps) against 0.750 for IQ4_NL (36 blocks in 3 steps). Mapping one lane pair to a 16-element sub-block would fix that if tg is short.
- Prompt processing goes through `mul_mm_id` dequantize, which costs about the same as IQ4_NL.
- With all 24 down tensors on Metal, UD-IQ3_XXS should move close to the candidate's speed, in the 200+ pp512 and around 19 tg128 range, because its sizes are similar. This is a guess until it is measured.

## Upstream path

- Code owners: `CODEOWNERS` maps `/ggml/src/ggml-metal/` to `@ggml-org/ggml-metal` (forforever73, ggerganov, jhen0409, masterFoad). ggerganov writes most Metal changes himself.
- Merge times for merged "Apple Metal" PRs created since 2026-03-01 (190 PRs): ggerganov's own, n 68, median 0.2 days, p90 2.8. All other authors, n 122, median 3.7 days, q3 12.1, p90 30.7. This counts merged PRs only, so it is biased low. #20456 is a counterexample: approved and still open after 253 days.
- Risk, PR #28898 "ggml: support quant scales for fp8 and nvfp4" (open, +2885 -2541, updated 2026-11-19). It moves the weight scales into ggml. Its Metal change makes `ggml_backend_metal_device_supports_op` return false for any op with a src scale. If it lands, Metal needs a scale epilogue too, or scaled NVFP4 falls back to the CPU again. A Metal NVFP4 PR should either handle that epilogue or land first.
- Likely route (**inference**): a fresh PR rebased on the `kernels/` split, with the 256-entry fix and a k = 1152 MUL_MAT_ID case, credited to #20456. It could also go as a rebase offered to the #20456 author. The register-spill check ggerganov asked for (#20399) needs a Mac Studio run, and only he can give it. Realistic merge time is one to four weeks if a code owner engages, and open-ended if not, as #20456 shows. For the thesis, only the local patch matters, and it does not have to wait for upstream.

## What this means for the project

Any wording here is **inference** and needs Daniel's call, along with a CHANGELOG addendum before any data.

1. Addendum G's Mac track rests on the fact that "the pinned and current Metal backends cannot multiply" NVFP4. A local patch 0008 would end that on our own build. Upstream merging would end it for everyone. Either way, the candidate's only measured edge over UD-IQ3_XXS (speed) is gone, and the KL FAIL remains.
2. A Metal NVFP4 patch also makes Addendum C's "native down" build (the release's NVFP4 down projections copied byte for byte) a Metal-runnable file. Addendum C allows it as a quality reference only, so making it a release candidate needs a new dated addendum.
3. The card wording "all expert tensors in Metal-supported types" stays true for the candidate. It just stops being a difference once NVFP4 is supported.
