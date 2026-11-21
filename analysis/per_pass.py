"""Per-pass residual for H1 (CHANGELOG.md, run 4). Core function: Daniel writes the body; tests in tests/test_per_pass.py.

Inputs are llama-perplexity logits files (scripts/kldfile.py reads them) scored on the same tokens:
- base:     the IQ4_NL build (no perturbation)
- singles:  the pass-1-only, pass-2-only and pass-3-only conditions, in that order
- combined: all three passes perturbed (the tied, ordinary build)

For each scored position, Delta_S = logp_S - logp_base (log-probabilities over the vocabulary). The residual is
R = Delta_all - sum_k Delta_k and the ratio r = ||R|| / ||Delta_all||. Two choices are yours and must be stated:
1. The norm. Plain l2 over the vocabulary is dominated by the clipped tail (values below max - 16 are floored by
   the writer). The Fisher norm under the base distribution, ||x||_p^2 = sum_i p_i (x_i - sum_j p_j x_j)^2, is the
   second-order KL geometry and ignores the tail; it also removes the per-token constant of log-softmax.
2. The aggregate: per-chunk ratio of sums of squares sqrt(sum_t ||R_t||^2 / sum_t ||Delta_all,t||^2) is robust to
   near-zero tokens; the mean of per-token r is not. Bootstrap over chunks, not tokens.

Return a dict with at least:
    "r_chunks": np.ndarray [n_chunk]   aggregate r per chunk
    "r":        float                  aggregate over all chunks
    "share_last": float                ||Delta_3|| / ||Delta_all||, same aggregation (1 means "counts once")
"""

from __future__ import annotations

from pathlib import Path


def residual_ratio(base: str | Path, singles: list[str | Path], combined: str | Path) -> dict:
    raise NotImplementedError("Daniel: core function for H1, see the docstring and tests/test_per_pass.py")
