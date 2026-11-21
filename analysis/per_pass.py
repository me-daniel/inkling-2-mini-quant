"""Per-pass residual for H1 (CHANGELOG.md, run 4). Written by Claude at Daniel's request (21 Nov 2026); Daniel reviews.

Inputs are llama-perplexity logits files (scripts/kldfile.py) scored on the same tokens:
- base:     the IQ4_NL build (no perturbation)
- singles:  the pass-1-only, pass-2-only and pass-3-only conditions, in that order
- combined: all three passes perturbed

Per scored position, Delta_S = logp_S - logp_base. R = Delta_all - sum_k Delta_k.

Primary norm: the Fisher norm under the base distribution, ||x||_p^2 = sum_i p_i (x_i - sum_j p_j x_j)^2. It is the
second-order geometry of KL (KL(base || S) ~ 1/2 ||Delta_S||_p^2), removes the per-token log-softmax constant, and
gives no weight to the clipped tail of the files (values below max - 16 are floored by the writer).
Secondary norm, matching the form of the registered prediction (reviews/2026-11-21-prediction-perpass-gpt-6.2-astra.md):
centred Euclidean over the "head" of the vocabulary, the entries within 10 nats of the base's top log-prob, which
stay clear of the clipping floor.

Aggregation: ratios of pooled sums of squares, per chunk and overall (robust to tokens with tiny effects); the
interval is a percentile bootstrap over chunks (chunks are the sampling units; tokens within a chunk are correlated).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import kldfile  # noqa: E402

HEAD_NATS = 10.0
# accumulated per chunk: squared norms of d1, d2, d3, d_all, sum_k d_k, R, under each norm
TERMS = ("d1", "d2", "d3", "all", "sum", "res")


def _sq_norms(p: np.ndarray, head: np.ndarray, deltas: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    out = {}
    for k, x in deltas.items():
        m = np.sum(p * x, axis=1, keepdims=True)
        fisher = float(np.sum(p * (x - m) ** 2))
        n_head = head.sum(axis=1, keepdims=True)
        mh = np.sum(np.where(head, x, 0.0), axis=1, keepdims=True) / n_head
        euclid = float(np.sum(np.where(head, (x - mh) ** 2, 0.0)))
        out[k] = (fisher, euclid)
    return out


def accumulate(base: str | Path, singles: list[str | Path], combined: str | Path, rows: int = 256) -> np.ndarray:
    """Per-chunk sums of squared norms, float64 [n_chunk, len(TERMS), 2] (last axis: Fisher, head Euclidean)."""
    paths = [base, *singles, combined]
    if len(singles) != 3:
        raise ValueError("need exactly three single-pass conditions")
    if not kldfile.same_tokens(*paths):
        raise ValueError("files were scored on different tokens")
    n_chunk = kldfile.read_header(base).n_chunk
    acc = np.zeros((n_chunk, len(TERMS), 2))
    for blocks in zip(*(kldfile.iter_blocks(p, rows) for p in paths)):
        c = blocks[0][0]
        lb, l1, l2, l3, la = (b[2].astype(np.float64) for b in blocks)
        p = np.exp(lb)
        p /= p.sum(axis=1, keepdims=True)
        head = lb >= lb.max(axis=1, keepdims=True) - HEAD_NATS
        d = {"d1": l1 - lb, "d2": l2 - lb, "d3": l3 - lb, "all": la - lb}
        d["sum"] = d["d1"] + d["d2"] + d["d3"]
        d["res"] = d["all"] - d["sum"]
        for i, (k, (f, e)) in enumerate(_sq_norms(p, head, {k: d[k] for k in TERMS}).items()):
            acc[c, i] += (f, e)
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
    }


def summarize(acc: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    out = {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, acc.shape[0], size=(n_boot, acc.shape[0]))
    for norm, name in ((0, "fisher"), (1, "head_euclid")):
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
