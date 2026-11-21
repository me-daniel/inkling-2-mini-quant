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


def ratio(tied: np.ndarray, indep: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, tied.size, size=(n_boot, tied.size))
    boots = tied[idx].mean(axis=1) / indep[idx].mean(axis=1)
    return {"ratio": float(tied.mean() / indep.mean()), "ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
            "kl_tied": float(tied.mean()), "kl_indep": float(indep.mean())}


if __name__ == "__main__":
    t = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/pilot/perpass-kl.tsv"
    print(ratio(chunks(t, "noise-tied"), chunks(t, "noise-indep")))
