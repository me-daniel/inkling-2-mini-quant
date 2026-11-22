# Concept v3: a lean order

Written 22 Nov 2026, morning, after the night's runs (CHANGELOG, Addendum H amendments 1 to 9d). It replaces the run order of `concept-v2.md`; the workflow lessons there still hold.

## What we have (measured)

- v1 is public: 11.5 GB, every expert tensor in a type stock llama.cpp runs on Metal; 1.75x generation and 3.3x prompt speed against UD-IQ3_XXS on stock llama.cpp; closest to the original among the all-Metal files under 12 GB we scored. It fails the registered gate against UD-IQ3_XXS, and says so.
- llama.cpp patches 0001 to 0007: the loop-scale fix (without 0001, passes 2 and 3 of Unsloth-style files run unscaled), per-pass cache slots that let one pass carry its own experts, and the converter fix. The slots are the instrument nobody else has.
- The per-pass test: loop error is not additive across passes (residual about 0.75 at two amplitudes), with layer-split, noise and B3 controls running.
- The night's diagnosis of UD's lead: it lives in the gate_up encoding, not the type map, the imatrix (domain, source, 4x data all near zero), the down format, our BF16 conversion or the quantizer version. Under the full input Gram, UD's gate_up beats llama-quantize's in 128 of 128 experts while losing under the diagonal, mostly along the experts' mean input.
- From the sibling repo: lowering per-layer error can shrink each expert's output (gain below 1), and over 48 layer visits that compounds into a worse KL. That explains the mean-aware refit's failure (gain 0.79, +0.072 KL). It is a necessary guard, not the whole story: rescaling its Q2_0 down to gain 1 (X5) still lost to UD by +0.0040, and UD's own Q2_0 runs at gain 0.91. At 8.49 GB the gap to UD splits into gate_up 61%, non-expert 23% and down 15%.

## Where we cannot compete, and where we can

- Not ours to win: day-0 releases of every size, an encoder we cannot see, a fork with its own kernels (Unsloth's prebuild already runs NVFP4 on Metal), GPU-heavy methods that need a PyTorch forward (GSQ-style). A leaderboard of quant files is their game.
- Ours: the loop. Everyone quantizes Inkling as a plain transformer. We can measure per pass, untie passes, and test whether a gain error compounds over visits (the refit failure suggests it does). The thesis question compares allocations within one encoder, so UD's encoder edge does not touch it. The win is a finding, not a file, unless the pilot in point 2 says otherwise.

## Lean order

1. Thesis core, H1 (GPU, in progress): finish the per-pass controls (H1 to H5). Add one clean instrument from the night: pure attenuation. Scaling a block's fp16 d by a constant (a byte edit, no other error) in a looped versus a non-looped layer, at P = 2, 3 and 4, shows whether a gain error compounds per visit while noise does not. Cheap, and it is the loop-specific result.
2. Encoder pilot (amendment 9 to 9d): code refinement under the full Gram with no attenuation allowed, on layers 9 to 11. The pilot decides: at least half of UD's 9-11 effect means all 24 layers next, then T3; below a quarter stops the route.
3. H2 (Daniel): llama-imatrix already sums the three visits, so "visit-pooled" needs a definition that adds something, for example weighting each visit by its effect on the output, or a gain-aware importance. Daniel's decision and his function.
4. v2 release only from our own encodings: the refiner (point 2) or cand-u (amendment 10, v1 rescaled per row to gain 1), and only if it beats v1 on T3. The focus is our own encodings: a file that stands without crediting anyone is itself the win (Daniel, 22 Nov). A build that carries UD's tensors (the mirror) may still be improved and released if it clearly beats UD itself (paired CI below 0 on T3), with credit; otherwise it stays a diagnostic. mirror-u is dropped: the sibling's X6d (UD's own Q2_0 downs rescaled to gain 1) was worse, +0.00185 [+0.00119, +0.00248]. The release path is v2r (amendment 12), our own encoding.
5. Parked: our Metal NVFP4 PR (Unsloth's fork ships one, and upstream has #330), more imatrix variants, more allocation search. Comparators (H6) finish in the background.

## Decisions for Daniel

- H2's definition (point 3).
- Whether the attenuation instrument (point 1) goes into the thesis as its own registered test.
- Approve Addenda F, G section 2, H and amendment 1 (T3).
