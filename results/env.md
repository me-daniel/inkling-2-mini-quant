# Build environment

Written by `scripts/build-llama.sh` on 2026-11-21 01:53 UTC. Do not edit by hand.

- llama.cpp base: `29b52d398b294509746ec0a3f28948a3107c0213` (b12600)
- Patched HEAD: `365e9a0cca4103c90702322bf9b7d2ca1afc8390`
- Patches (sha256, first 16):
  - `0001-model-inkling2_loop-re-share-the-loop-slots-after-th.patch` `1a74d86a8f51e83c`
  - `0002-tests-check-that-inkling2_loop-passes-share-the-expe.patch` `6c374aaee9b86dd5`
  - `0003-model-inkling2_loop-cache-slots-can-carry-their-own-.patch` `0384b584ba9dac21`
  - `0004-tests-check-that-inkling2_loop-slots-use-their-own-r.patch` `8df95dba137fab3a`
- Build: Release, GGML_METAL=ON, cmake version 4.4.5, Apple clang version 21.0.0 (clang-2100.1.1.101)
- Machine: Apple M5 Pro, 48 GB, macOS 26.5, iogpu.wired_limit_mb=40960
- Loop tests (`test-llama-archs --loop-scales -s 42`, `--loop-slots -s 42`):
  - loop-scales: fold weights/2 scale*2 on blk.1: nmse 0.000e+00 (OK)
  - loop-scales: scale*2 only on blk.1:          nmse 7.000e-09 (OK)
  - loop-scales: scale*2 only on slot blk.3:     nmse 0.000e+00 (OK)
  - loop-slots: slot blk.3 = physical values:   nmse 0.000e+00 (OK)
  - loop-slots: slot blk.3 = own values:        nmse 2.074e-10 (OK)
  - loop-slots: slot blk.3 scale*2:             nmse 2.108e-09 (OK)
