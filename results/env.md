# Build environment

Written by `scripts/build-llama.sh` on 2026-11-21 00:46 UTC. Do not edit by hand.

- llama.cpp base: `29b52d398b294509746ec0a3f28948a3107c0213` (b12600)
- Patched HEAD: `f3023915438694a4df7aa66bd6dcba882f2d0cb3`
- Patches (sha256, first 16):
  - `0001-model-inkling2_loop-re-share-the-loop-slots-after-th.patch` `1eee2e92c54de6cd`
  - `0002-tests-check-that-inkling2_loop-passes-share-the-expe.patch` `b94187ed8d142d7f`
- Build: Release, GGML_METAL=ON, cmake version 4.4.5, Apple clang version 21.0.0 (clang-2100.1.1.101)
- Machine: Apple M5 Pro, 48 GB, macOS 26.5, iogpu.wired_limit_mb=40960
- Loop-scale test (`test-llama-archs --loop-scales -s 42`):
  - loop-scales: fold weights/2 scale*2 on blk.1: nmse 0.000e+00 (OK)
  - loop-scales: scale*2 only on blk.1:          nmse 7.000e-09 (OK)
  - loop-scales: scale*2 only on slot blk.3:     nmse 0.000e+00 (OK)
