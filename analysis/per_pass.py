"""Per-pass residual for H1 (CHANGELOG.md, run 4). Written by Claude at Daniel's request (21 Nov 2026); Daniel reviews.

Inputs are llama-perplexity logits files (scripts/kldfile.py) scored on the same tokens:
- base:     the IQ4_NL build (no perturbation)
- singles:  the pass-1-only, pass-2-only and pass-3-only conditions, in that order
- combined: all three passes perturbed

Per scored position, Delta_S = logp_S - logp_base. R = Delta_all - sum_k Delta_k.

Primary norm: the Fisher norm under the base distribution, ||x||_p^2 = sum_i p_i (x_i - sum_j p_j x_j)^2. It is the
second-order geometry of KL (KL(base || S) ~ 1/2 ||Delta_S||_p^2) and removes the per-token log-softmax constant. It
gives little, not zero, weight to the clipped tail: the writer floors values below max - 16, and the floored entries
keep their small base mass.
Secondary norms. The registered prediction (reviews/2026-11-21-prediction-perpass-gpt-6.2-astra.md, points 5 and 6)
asks for pooled Euclidean RMS with "the same vocabulary centring everywhere", i.e. centred Euclidean over the full
vocabulary. That form cannot be computed faithfully from these files, because the floored entries hold the writer's
floor, not the model's log-probs. Two forms are reported:
- head_euclid, the substitute used to score the registration: centred Euclidean over the "head", the entries within
  10 nats of the base's top log-prob, which stay clear of the floor;
- full_euclid, a sensitivity check: centred Euclidean over the full vocabulary, floored entries included as written.

Aggregation: ratios of pooled sums of squares, per chunk and overall (robust to tokens with tiny effects); the
interval is a percentile bootstrap over chunks (chunks are the sampling units; tokens within a chunk are correlated).
CHANGELOG run 4 names "a per-token residual ratio"; the pooled ratio replaces it. The Fisher primary norm, the
10-nat head and the pooled ratio were fixed in this file before the first per-pass logits existed, but they are not
recorded in CHANGELOG.md (audit reviews/2026-11-21-audit-perpass-opus.md, finding 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import kldfile  # noqa: E402

HEAD_NATS = 10.0
NORMS = ("fisher", "head_euclid", "full_euclid")  # order of the last axis of accumulate()
# accumulated per chunk: squared norms of d1, d2, d3, d_all, sum_k d_k, R and the pairwise sums (for the 3x3 Gram)
TERMS = ("d1", "d2", "d3", "all", "sum", "res", "d12", "d23", "d13")


def _sq_norms(p: np.ndarray, head: np.ndarray, deltas: dict[str, np.ndarray]) -> dict[str, tuple[float, float, float]]:
    out = {}
    for k, x in deltas.items():
        m = np.sum(p * x, axis=1, keepdims=True)
        fisher = float(np.sum(p * (x - m) ** 2))
        n_head = head.sum(axis=1, keepdims=True)
        mh = np.sum(np.where(head, x, 0.0), axis=1, keepdims=True) / n_head
        euclid = float(np.sum(np.where(head, (x - mh) ** 2, 0.0)))
        full = float(np.sum((x - x.mean(axis=1, keepdims=True)) ** 2))
        out[k] = (fisher, euclid, full)
    return out


def accumulate(base: str | Path, singles: list[str | Path], combined: str | Path, rows: int = 256) -> np.ndarray:
    """Per-chunk sums of squared norms, float64 [n_chunk, len(TERMS), len(NORMS)] (last axis in NORMS order)."""
    paths = [base, *singles, combined]
    if len(singles) != 3:
        raise ValueError("need exactly three single-pass conditions")
    if not kldfile.same_tokens(*paths):
        raise ValueError("files were scored on different tokens")
    n_chunk = kldfile.read_header(base).n_chunk
    acc = np.zeros((n_chunk, len(TERMS), len(NORMS)))
    for blocks in zip(*(kldfile.iter_blocks(p, rows) for p in paths)):
        c = blocks[0][0]
        lb, l1, l2, l3, la = (b[2].astype(np.float64) for b in blocks)
        p = np.exp(lb)
        p /= p.sum(axis=1, keepdims=True)
        head = lb >= lb.max(axis=1, keepdims=True) - HEAD_NATS
        d = {"d1": l1 - lb, "d2": l2 - lb, "d3": l3 - lb, "all": la - lb}
        d["sum"] = d["d1"] + d["d2"] + d["d3"]
        d["res"] = d["all"] - d["sum"]
        d["d12"], d["d23"], d["d13"] = d["d1"] + d["d2"], d["d2"] + d["d3"], d["d1"] + d["d3"]
        for i, (k, v) in enumerate(_sq_norms(p, head, {k: d[k] for k in TERMS}).items()):
            acc[c, i] += v
    return acc


def _stats(acc: np.ndarray, norm: int) -> dict[str, float]:
    s = acc[:, :, norm].sum(axis=0)
    get = dict(zip(TERMS, s))
    return {
        "r": float(np.sqrt(get["res"] / get["all"])),
        "share_last": float(np.sqrt(get["d3"] / get["all"])),
        "norm_ratio": float(np.sqrt(get["all"] / get["sum"])),  # >1: the combined effect is larger than the sum
        "pass1_over_pass3": float(np.sqrt(get["d1"] / get["d3"])),
        "pass2_over_pass3": float(np.sqrt(get["d2"] / get["d3"])),
        # squared-norm ratio ~ KL ratio: what perturbing all visits costs relative to one average visit
        "visit_multiplier": float(get["all"] / ((get["d1"] + get["d2"] + get["d3"]) / 3)),
        **{f"cos_{a}{b}": float((get[f"d{a}{b}"] - get[f"d{a}"] - get[f"d{b}"]) / (2 * np.sqrt(get[f"d{a}"] * get[f"d{b}"])))
           for a, b in (("1", "2"), ("2", "3"), ("1", "3"))},
    }


def summarize(acc: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    out = {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, acc.shape[0], size=(n_boot, acc.shape[0]))
    for norm, name in enumerate(NORMS):
        point = _stats(acc, norm)
        boots = [_stats(acc[i], norm) for i in idx]
        ci = {k: (float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))) for k in point}
        per_chunk = acc[:, :, norm]
        out[name] = {
            **point,
            "ci95": ci,
            "r_chunks": np.sqrt(per_chunk[:, TERMS.index("res")] / per_chunk[:, TERMS.index("all")]),
        }
    return out


def residual_ratio(base: str | Path, singles: list[str | Path], combined: str | Path) -> dict:
    """The primary (Fisher) result with r, r_chunks and share_last, plus the full summary under "detail"."""
    s = summarize(accumulate(base, singles, combined))
    f = s["fisher"]
    return {"r": f["r"], "r_chunks": f["r_chunks"], "share_last": f["share_last"], "detail": s}


if __name__ == "__main__":
    import json

    if len(sys.argv) != 6:
        sys.exit("usage: per_pass.py BASE.kld P1.kld P2.kld P3.kld ALL.kld")
    res = residual_ratio(sys.argv[1], sys.argv[2:5], sys.argv[5])
    det = res["detail"]
    for k in det:
        det[k]["r_chunks"] = [round(float(x), 4) for x in det[k]["r_chunks"]]
    print(json.dumps(det, indent=1))
