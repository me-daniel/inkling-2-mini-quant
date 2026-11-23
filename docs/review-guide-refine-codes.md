# Reading guide: `scripts/refine-codes.py` (for Daniel's review, 23 Nov 2026)

What it does: for each expert's gate_up matrix, it keeps llama-quantize's block scales and changes only which grid point (and sign) each group of 4 (IQ3_XXS) or 8 (IQ2_S) weights uses. A change is kept if it lowers the output error that the layer's real inputs would see. v2r and v2m are built with it; the sibling's c1r will be too. Prior work: MACaQ, Appendix C (arXiv 2611.10649), the same idea on dense models.

## Read in this order (about an hour)

1. Docstring, lines 1-55. The whole method in prose. The objective is J = tr(E H E^T): E = quantized minus BF16 weights, H = the input Gram G (how often input dimensions fire together) plus 0.1 x its diagonal.
2. `refine`, lines 314-395. The core loop. Per sweep: for each group of columns, pick the best candidate per row (`_pick_iq2s` / `_pick_iq3xxs`), score it exactly (`dj`), and accept it only if dj < 0 and the gain guard holds. Then update E, R = E H, and the raw bytes.
3. `_one`, lines 421-457. Per expert: the two guards (skip experts with fewer than 2304 Gram tokens; H = G + shrink x Diag(G)), then the hard checks after refining. Decode equals gguf-py bit for bit, the gain did not fall, and the scale bits are unchanged.
4. `validate`, lines 176-196. Every output value is a legal code of the format (IQ3_XXS sign parity included).
5. Skim only: `_pick_iq2s` / `_polish_iq2s` (203-283, the IQ2_S sign search from Astra's review) and `_pick_iq3xxs` (284-312, exhaustive over 256 grids x 8 sign patterns).

## What to check (the judgement calls, not the arithmetic)

- The gain guard (amendment 9c): a row may not end with a lower output gain than llama-quantize gave it. This comes from the sibling's finding that shrinking outputs compounds over the loop's 48 layer visits. Is "never below the starting value" the right rule, or too strict?
- shrink = 0.1 and min tokens = 2304: guards against a Gram estimated from too few tokens (2304 = the input width). Both were set in the 9b pilot and never tuned.
- The Grams come from v1's forward pass (cand.gguf), not from the model being refined. It is cheaper and the same for every build, but the inputs differ slightly. For c1 (1-bit early layers) the difference is larger.
- Layers 0 and 4 (IQ2_XXS, IQ2_XS) are not refined. Amendment 14 found no resolved effect from UD's tensors there on T1 or T2.

## Evidence it does what it says

- `tests/test_refine_codes.py`: 12 tests. Decode equals gguf-py, re-encoding unchanged codes is the identity, a group write touches only its group, J falls and the output stays valid, the ranked winner is the exact best candidate, and one test each for Astra's three review points.
- Every run checks each expert (the `OK` lines in `logs/v2m-check-*.txt`) and the final splice (`logs/v2m-check-final.txt`: size, types, layers 0 and 4 byte-identical).
- Offline error tables per group: `results/pilot/v2m-offline-*.md`.
