# Decision brief, evening of 2026-11-21

Written by `reviews/2026-11-21-decision-brief-evening/build.py` at 22:44. The script reads every number from a file. "Inference" and "estimate" mark what is not measured. No GPU job was started.

## 1. Answer

There is an edge, and it is narrow: among GGUFs under 11,663,310,784 bytes that run fully on Metal, the candidate has the lowest held-out KL, ahead of AtomicChat AD-IQ2_S by 0.00811 on T1 and 0.00608 on T2 at the same speed. Against UD-IQ3_XXS it loses by 0.00747 and 0.00581, and leads only on Mac speed (3.30x pp512, 1.75x tg128), because Metal has no NVFP4 matmul. Upstream PR #20456 (220 added lines, open, last updated 2026-06-23) would end that lead (inference). Do not redo the candidate: it is frozen, T1 and T2 are seen, and our 12.66 GB all-3-bit build scores 0.02213 on S against UD's 0.02050, so no reallocation inside Metal types passes the gate (inference). The model is a stopgap for llama.cpp on a Mac at 11.51 GB, and the pilot's per-pass method is worth more than the file.

**P = 4 speed adds up.** 507 s against 400 s at P = 3 is a ratio of 1.27. Layer visits (60 against 48) give 1.25. Timing only, no KL at P = 4.

## 2. Your decisions

1. **The upload.** `logs/hf-upload-v1.log` (written 22:21) holds a Hugging Face commit URL. I did not open it. 0 of 4 Amendment 2 comparator runs have a KL, so the Mac-track rule is not complete. Options: leave it (private, or approved by you), or set it private until the rule is complete. Recommendation: private until then.
2. **Addendum G.2 deviations.** Approve or reject. Recommendation: approve, and keep the full-vocabulary form as the sensitivity check.
3. **Diagnostic on S.** (a) Hybrid: the candidate with UD's native NVFP4 down. (b) Hybrid plus mirror: UD's gate_up and non-experts with our IQ4_NL down. (c) Neither. Recommendation: (b). Paste the Amendment 3 draft and write a dated mirror entry before any score. Mirror is the obvious rival to the Mac claim.
4. **Release.** (a) One model upload, no upkeep. (b) Measurements only. (c) Nothing. Recommendation: (a) only if both pending comparators are beaten and mirror does not beat the candidate, otherwise (b). `drafts/release/README.md` first mentions the failure on line 20 of 63 and states 0 of 2 margins. Addendum G wants the FAIL first, with both.
5. **Metal NVFP4 kernel (Claude) and the NVFP4 to IQ4_NL re-encode (yours).** (a) Start now (kernel spike estimate: half a day). (b) Wait for decision 3: cand minus hybrid on S is the prize for the kernel and the ceiling for the re-encode. (c) Drop both from the pilot. Recommendation: (b).
6. **Gate-path tests.** Claude adds tests for `maptable.py`, `allocate.py`, `compare.py` and a chunk-count check now (CPU only), or later. Recommendation: now, before anything is published.

## 3. GPU queue, in order

1. **Running:** `scripts/pilot-chain6.sh`, last log line "[22:43:37] depth bench atomic-AD-IQ2_S". It ends with `macgate.sh`: IsValorum APEX-I-NanoPlus-V2.1 and prithivMLmods Q2_K on T1 and T2. 0 of 2 files are in place under `models/ext`.
2. **`scripts/hybrid.sh`.** Ready, not run. Needs Amendment 3 in CHANGELOG first. About 15.25 min, the ETA of UD's own S run (inference).
3. **Mirror.** No script yet. `scripts/splice.py --take` and `--add` cover it, with `cand.gguf` as base. About 5.30 min, the ETA of an all-Metal S run (inference).
4. **Per-pass controls,** each registered first: 3-bit layer split, Gaussian single-pass, more noise draws, then B3. At P = 4 a per-pass file leaves pass 4 (slots 48 to 59) on the physical experts (inference from the patch review).

## 4. Code-review findings to look at

No finding changes a reported number. A second agent rechecked the first four and rated each minor.

- `analysis/maptable.py:13` and `:24`. Chunk lines the regex cannot parse are dropped without a warning, and the next chunk gets a wrong value. Fired on 1 of 54 KL logs: `ppl-floor-q8-S-P3.log` keeps 14 of 16 chunks (largest kept value 0.000000).
- `tests/`. 0 test files touch `maptable.py`, `allocate.py` or `compare.py` (24 tests pass in total). The gate and Mac verdicts run through all three.
- `tests/test_per_pass.py:65`: `test_shift_invariance` cannot fail. `:108`: the head norm, which scores the registration, is checked only as `full > head`. Removing the centring at `analysis/per_pass.py:49` or `:52` passes every test.
- `patches/llama.cpp/0003-model-inkling2_loop-cache-slots-can-carry-their-own-.patch:28`. On upstream 690b87d the series applies without a conflict, but this hunk lands in the wrong function. Keep the pin at `scripts/build-llama.sh:7`.
- `analysis/compare.py:44` prints one label for two rules: `mac-quality.txt` ends with "quality gate: PASS", `gate-quality.txt` with "quality gate: FAIL". From the review only.
