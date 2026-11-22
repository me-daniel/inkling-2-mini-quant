# Refiner code review

- Date: 2026-11-22, ~09:10
- Model: GPT-6.2 Astra, effort high (Codex CLI, isolated)
- Reviewed: scripts/refine-codes.py and tests/test_refine_codes.py against amendments 9, 9b, 9c.
- Response: point 1 limits the IQ2_S search (signs fixed from the continuous target), not correctness, and does not touch the 9-11 pilot (IQ3_XXS); fix before any IQ2_S layer is used. Points 2 and 3 are edge cases that did not occur in the 9-11 run (it completed; gains stayed at or above their floors in the log); both get fixed before a full run.

## Reply (verbatim)

1. **Blocking — IQ2_S can miss a gain-feasible improvement.** `scripts/refine-codes.py:198` fixes all eight signs from the continuous minimizer and searches only grid magnitudes. Take `db=1`, grid zero, `q[0:2]=(8,8)`, `w[0:2]=(1,1)`, and `q[8]=w[8]=-8`. Let all other weights match. Set `G` to identity with the submatrix on columns `{0,1,8}` filled with ones. The proposed signs stay positive. Since 8 is the smallest grid magnitude, no candidate searched improves the row. Yet flipping just the first sign lowers the regularized row objective from **205.8 to 17** and **raises** the gain numerator by **94.4**. The winner test misses this because its IQ2_S oracle reuses the picker’s signs. [llama.cpp IQ2_S grid](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h#L718-L720)

2. **A valid 2304-token expert can abort the entire run during reporting.** `scripts/refine-codes.py:368` divides by `den = tr(W G Wᵀ)` unconditionally. Take 2304 identical tokens `(1,1,0,…)` and BF16 rows `(1,-1,0,…)`. The count passes, `G` has positive trace, and the regularized denominator is positive. The unregularized denominator is exactly zero. Computing `rel_before` raises `ZeroDivisionError` before the GGUF is written. The undefined reporting ratio should be recorded as `null`.

3. **The strict gain floor can admit a small loss.** `scripts/refine-codes.py:293` adds `dnum` to the full numerator before comparing it to the floor. For IQ2_S with `db=1`, take `w` all 8s and `q` identical except `q[0]=43`. Use diagonal `G` with `G[0,0]=2⁻⁶⁴` and all other diagonal entries 10. Switching to grid zero changes 43 to 8 and eliminates the error, but lowers the gain numerator by `308 × 2⁻⁶⁴`. The group dot products round equal in the candidate mask, and the row numerators round equal in the acceptance check. Tracking gain *change from baseline*, initialized to zero, catches this first move.

By inspection, the accepted-row `E` and `R` updates, weight units, and `dJ` expansion are correct. The default cutoff includes exactly 2304 tokens. The IQ2_S sign and high-bit layout and IQ3_XXS parity packing agree with the upstream decoder. The workspace lacked the repository and model artifacts, so I could not run the model-level tests or inspect `ge.symmetrise`. [llama.cpp reference decoders](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c#L2397-L2454)
