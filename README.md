# Counting passes

![Inkling 2 Mini has 24 layers. Layers 6 to 17 run three times on the same expert weights. Does quantization error count once per pass?](figures/hero.svg)

Inkling 2 Mini is a mixture-of-experts model with a loop. Layers 6 to 17 run three times on the same expert weights. In this exploratory pilot, we measured what that loop does to quantization error. The repository holds the llama.cpp patches, run scripts, analysis code and result files. The GGUF is hosted separately on Hugging Face.

- Full write-up: [docs/report.md](docs/report.md)
- Model on Hugging Face: [Me-Daniel/Inkling-2-Mini-Metal-GGUF](https://huggingface.co/Me-Daniel/Inkling-2-Mini-Metal-GGUF)

## Recommended file: Inkling-2-Mini-Metal-refined2.gguf (11.57 GB)

Since 24 Nov 2026 this is the file we recommend. It is the refined file described below with one change: the refiner's calibration text now also holds Python and Rust source from 16 other repositories. Tensor types, size and speed are the same as the other two files.

| KL on new text | Main file | Refined file | Refined2 | UD-IQ3_XXS |
|---|---|---|---|---|
| Python source | 0.01944 | 0.01659 | 0.01489 | 0.01409 |
| Rust source | 0.01144 | 0.01013 | 0.00896 | 0.00840 |
| Go source (in no calibration text) | 0.01799 | 0.01653 | 0.01417 | 0.01324 |
| TypeScript source (in no calibration text) | 0.01980 | 0.01697 | 0.01511 | 0.01335 |
| German parliament speeches | 0.03626 | 0.03073 | 0.03149 | 0.02571 |
| Tasks right of 300 (maths, tool calls, code and JSON) | 253 | 262 | 264 | not run |

- Against the main file: KL 23% lower on new code in four languages (ratio 0.773 [0.745, 0.803]) and 13% lower on German (0.868 [0.842, 0.895]).
- Against the refined file: 9% lower on Python and Rust (0.905 [0.862, 0.958]) and 12% lower on Go and TypeScript (0.879 [0.852, 0.904]), but 3% higher on German (1.028 [0.999, 1.055]). Before testing we allowed German to get at most 5% worse, judged by that upper bound, which this narrowly misses. By our registered rule the refined file would have stayed; we decided the German difference does not outweigh the gain on code and say so here.
- The same new texts chose this file over the refined one and then measured it, so its numbers are a little optimistic. The refined file was not chosen on them: against the main file it is 13% lower on the same code (0.867 [0.834, 0.902]).
- Unsloth's UD-IQ3_XXS is still 5% to 12% closer to the original on new code, and it does not run fully on the Mac GPU.
- Tasks: of the 200 maths, code and JSON tasks, this file and the main file disagree on 5, and this file is right on 5 of them.
- Speed and memory match the main file: in two back-to-back runs, whichever file ran earlier was faster.
- Not scored on T3, which was opened once, for the refined file.

Run it with the same commands as the main file, with `-m Inkling-2-Mini-Metal-refined2.gguf`.

## Results

Every number comes from `results/pilot/`. KL is measured against a Q8_0 reference at P = 3 passes. T1 and T2 are held-out sets.

### 1. Closest to the original among the all-Metal files we tested, behind Unsloth's NVFP4 file

Unsloth's UD-IQ3_XXS is a 149 MB larger file and stays closer to the original model (KL +0.00747 on T1 and +0.00581 on T2, our file minus theirs), so the quality test we registered against it beforehand did not pass.

With stock llama.cpp, this GGUF keeps the whole model on the Mac GPU instead. There it currently reads prompts 3.3× faster and generates 1.75× faster than UD-IQ3_XXS. Among the all-Metal GGUFs under 11.7 GB we tested (AtomicChat AD-IQ2_S, bartowski IQ2_S, mudler APEX-I-Nano), it stays closest to the original model on both held-out sets.

- Pre-registered gate: FAIL. The candidate (11,514,054,944 bytes) has higher KL than Unsloth UD-IQ3_XXS (11,663,310,784 bytes) on both held-out sets. The difference is +0.00747 [+0.00654, +0.00845] on T1 and +0.00581 [+0.00513, +0.00673] on T2. The FAIL stands.
- The candidate passes a narrower, post hoc comparison that we registered after the FAIL. We scored 3 comparison files under the size limit that keep all expert tensors in Metal-supported types (bartowski IQ2_S, AtomicChat AD-IQ2_S, mudler APEX-I-Nano). The candidate has lower KL than each on T1 and T2, with the 95% CI for each difference below zero. We registered two more files under the limit (IsValorum APEX-I-NanoPlus-V2.1, prithivMLmods Q2_K) and later scored them with three more (amendment 11): the candidate has lower KL than each of the 8 on both sets (`results/pilot/under116-quality.txt`).
- With stock llama.cpp, UD-IQ3_XXS is slower on our Mac. It keeps 20 expert tensors in NVFP4, which stock llama.cpp's Metal backend cannot multiply yet. On an Apple M5 Pro with llama-bench (llama.cpp `29b52d3` plus our patches), UD-IQ3_XXS generated at 10.87 t/s and the candidate at 18.97 t/s. The candidate was 1.75 times as fast. The all-Metal files we tested at this size ran at about the same speed. Builds that run NVFP4 on Metal don't have the CPU fallback, so the gap applies to stock llama.cpp.

![KL on T1 and T2 for the seven files](figures/mac-comparison.png)

![Speed as the context fills up](figures/depth-speed.png)

With 16K tokens already in context, the speed gap to UD-IQ3_XXS narrows to 2.1× on prompts and 1.46× on generation (`results/pilot/depth-*.txt`).

### 2. Single-pass changes do not add up to the all-pass change (2-bit block, layers 6-11)

- The test: we put a 2-bit block (layers 6-11) on pass 1 only, pass 2 only, pass 3 only and on all three. Then we compared the all-pass logit change with the sum of the three single-pass changes. The residual share r is 0.675 [0.657, 0.693], where 0 would mean the parts add up exactly.
- The control: splitting the same block by layer pair, where the parts share no weights, gives r = 0.838 [0.822, 0.855]. By both r and the norm ratio, the pass split is closer to additive than the layer split. The intervals for the two splits do not overlap on either measure.
- What it does not show: the additivity measures disagree. KL all / sum is 1.135, above 1, and the Fisher norm ratio is 0.841, below 1. We do not classify the error. The result covers one block at one amplitude on 16 chunks of text.

![Residual share and norm ratio for the pass split and the layer-split control](figures/pass-vs-layer-split.png)

How the test is built:

![Physical layers, cache slots and the five conditions of the per-pass test](figures/loop-diagram.svg)

## Second file: Inkling-2-Mini-Metal-refined.gguf (11.57 GB)

Same tensor types, size class and speed as the main file, with the expert weights of 22 layers re-encoded. Our code refiner keeps llama-quantize's block scales and picks, for each small group of weights, the grid code that keeps the layer's output closest to the original on calibration text (WikiText, chat conversations and prose). [MACaQ](https://arxiv.org/abs/2611.10649) (Appendix C) published the same idea for dense models.

| Test | Main file | Refined file | UD-IQ3_XXS |
|---|---|---|---|
| KL on T3, a sealed WikiText set opened once | 0.02847 | 0.02081 | 0.01987 |
| KL on new Python source | 0.01668 | 0.01529 | 0.01261 |
| KL on new Rust source | 0.01703 | 0.01641 | 0.01269 |
| KL on new German parliament speeches | 0.03626 | 0.03073 | 0.02571 |
| Tasks right (maths, tool calls, code and JSON) | 253 of 300 | 262 of 300 | not run |
| Prompt / generation t/s (same run) | 265 / 20.7 | 272 / 20.9 | |

- Closer to the original on every text we tried, though on Rust the difference is within noise (ratio 0.955 [0.910, 1.009]). On T3 its KL is 26% lower than the main file's (-0.00766 [-0.00898, -0.00631]). On new code the gain is smaller: 8% on Python and 5% on Rust (16% on German). The refiner partly fits its calibration text, and that text holds almost no code.
- Unsloth's UD-IQ3_XXS stays closer to the original: by 6% on T3 (+0.00094 [+0.00029, +0.00155]) and by 20% to 30% on the new texts.
- No task type got worse: of the 200 maths, code and JSON tasks, the two files disagree on 6, and the refined file is right on 6 of them. 200 tasks can rule out a large drop but can't measure a small gain.
- Why it was not made the default on 23 Nov: before testing we required KL at least 10% lower than the main file's on new Python, Rust and German together, with 95% confidence. It came out 9% lower (95% CI 7% to 12%), so the rule we registered kept the main file as the default.

Run it with the same commands as the main file, with `-m Inkling-2-Mini-Metal-refined.gguf`.

## Run the model

The model needs a llama.cpp build with Inkling 2 support (upstream after 20 Nov 2026). Homebrew's llama.cpp releases as of that date could not load it.

```sh
llama-cli -m Inkling-2-Mini-Metal-refined2.gguf -ngl 99 --jinja -c 32768 --temp 1.0 --reasoning-budget 16384
llama-server -m Inkling-2-Mini-Metal.gguf -ngl 99 --jinja -c 32768 --temp 1.0 --reasoning-budget 16384 --port 8080
```

The release sets the temperature to 1.0 in `generation_config.json`. The base model card warns about long reasoning loops, so the commands cap thinking at 16K tokens. The loop count stays at its default of 3.

Stock upstream llama.cpp (`c661033`, no patches) gives this file the same scores as the build used for the tables (T1: KL 0.026856, top-1 93.139%).

### Which Mac

- 24 GB or more: the model fits with room for a 16K context and other apps.
- 16 GB: the model can run after you raise the GPU memory limit, with a short context and little RAM left for other apps. By default, macOS lets the GPU use only about two thirds of RAM, which is less than the model needs. To raise the limit:

```sh
sudo sysctl iogpu.wired_limit_mb=13000   # resets on reboot
```

On our Mac, memory use for the whole process peaked at 13.9 GB with the context filled to 16K tokens (`results/pilot/memory-cand.txt`).

## Reproduce

Everything ran on one Apple M5 Pro with 48 GB, one heavy process at a time. Model files, data and logs are not in git. The scripts rebuild them and resume where they stopped.

```sh
sh scripts/build-llama.sh                     # llama.cpp 29b52d3 (b12600) plus patches/llama.cpp, Release with Metal
.venv/bin/python -I scripts/prepare-data.py   # S, T1 and the T2 prompts, with hashes
sh scripts/g0.sh                              # BF16 conversion, Q8_0 reference, IQ4_NL build, 2-bit and 3-bit pools
sh scripts/map.sh                             # block map and pass sweep
sh scripts/phase2b.sh                         # map entry with non-expert tensors at Q6_K
sh scripts/perpass.sh                         # per-pass test and noise conditions
sh scripts/map2.sh                            # map levels with a 2-bit gate_up only
sh scripts/q2opt-all.sh                       # our weighted Q2_0 encoding of every layer
sh scripts/map3.sh                            # map level with the weighted Q2_0 down projection
sh scripts/perpass2.sh                        # per-pass controls
sh scripts/phase4.sh                          # T2 traces, allocation, candidate, gate
sh scripts/fetch-ext.sh                       # outside files for the Mac comparison
sh scripts/macgate.sh                         # Mac comparison, stability, bench
.venv/bin/python -I scripts/figures.py        # figures from results/pilot/
```

We ran the steps in the order shown. The chain scripts `scripts/pilot-chain.sh` to `scripts/pilot-chain6.sh` ran them back to back. Each one waits for the step before it and expects the earlier chain's output in `logs/`.

Checks:

```sh
.venv/bin/python -I -m pytest -q tests                                  # analysis tests on synthetic data
vendor/llama.cpp/build/bin/test-llama-archs --loop-scales -s 42         # passes share the expert scales
vendor/llama.cpp/build/bin/test-llama-archs --loop-slots -s 42          # a slot uses its own experts
vendor/llama.cpp/build/bin/test-llama-archs -a 'inkling2.*'             # upstream consistency tests
```

We created the Python environment with `uv venv .venv` and installed the packages in llama.cpp's `requirements-convert_hf_to_gguf.txt`, plus gguf-py, pyarrow, tokenizers, matplotlib and pytest. The engine is pinned to llama.cpp `29b52d398b294509746ec0a3f28948a3107c0213` plus 7 patches. `results/env.md` lists the patch hashes and the build.

## Repository layout

| Path | What it holds |
|---|---|
| `docs/report.md` | The pilot report |
| `figures/` | Figures as PNG and SVG, written by `scripts/figures.py`, plus the two diagrams |
| `results/` | Build environment, data hashes and the pilot's result files in `results/pilot/` |
| `analysis/` | Per-pass residual, allocator, paired bootstrap, report writers and the figure style |
| `scripts/` | Build, data, splicing, run chains, reports and figures |
| `patches/llama.cpp/` | 7 patches on llama.cpp `29b52d3`, three of them tests |
| `tests/` | Tests for the analysis code |

The scripts create `models/`, `data/`, `logs/` and `vendor/`. These directories stay out of git.

## Status

We ran this exploratory pilot on 21 November 2026 for a master's thesis, "Counting Passes". None of the results are confirmatory evidence. We used one reference, one importance matrix, one machine and one engine commit. KL measures divergence from the reference and does not measure task quality.

We recorded the hypotheses, data splits and gates before collecting the data each one governs. The pre-registration and a dated addendum for every later change appear in [CHANGELOG.md](CHANGELOG.md).
