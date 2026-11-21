"""Addendum A: tied against independent noise. KL ratio with a paired bootstrap over chunks (results/pilot/perpass-kl.tsv)."""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def chunks(table: Path, cond: str, against: str = "pp-base") -> np.ndarray:
    for r in csv.DictReader(open(table), delimiter="\t"):
        if r["cond"] == cond and r["against"] == against:
            return np.array([float(x) for x in r["per_chunk"].split(",")])
    raise KeyError(cond)


def paired_ratio(num: list[np.ndarray], den: list[np.ndarray], scale: float = 1.0, n_boot: int = 10000, seed: int = 0) -> dict:
    """scale * sum of mean chunk KLs in num / the same for den, e.g. all / (p1 + p2 + p3).

    Paired percentile bootstrap over chunks: every replicate draws one set of chunk indices for all conditions and
    recomputes the ratio. Returns the point estimate, the 95% CI and the per-chunk ratios (min, max)."""
    a, b = np.sum(num, axis=0), np.sum(den, axis=0)  # per-chunk sums over the conditions
    if a.shape != b.shape:
        raise ValueError("conditions differ in chunk count")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = scale * a[idx].mean(axis=1) / b[idx].mean(axis=1)
    per_chunk = scale * a / b
    return {"ratio": float(scale * a.mean() / b.mean()),
            "ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
            "chunk_range": (float(per_chunk.min()), float(per_chunk.max()))}


def ratio(tied: np.ndarray, indep: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    out = paired_ratio([tied], [indep], n_boot=n_boot, seed=seed)
    return {"ratio": out["ratio"], "ci95": out["ci95"], "kl_tied": float(tied.mean()), "kl_indep": float(indep.mean())}

if __name__ == "__main__":
    t = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/pilot/perpass-kl.tsv"
    print(ratio(chunks(t, "noise-tied"), chunks(t, "noise-indep")))
