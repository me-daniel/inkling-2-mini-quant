"""Amendment 15, component A: fresh-domain KL of v2m against v1 (H1) and, descriptive, against UD-IQ3_XXS.

    python -I analysis/battery_A.py   (reads results/pilot/kl.tsv rows batA-<domain>-<model>-P3; prints Markdown)

H1: R = mean over the H1 domains of exp(mean over chunks of log(KL_v2m / KL_v1)); 95% CI by cluster bootstrap,
documents resampled with replacement within each domain (chunk i belongs to document i // 4), 10,000 resamples,
seed 0. Math is near-calibration by the freshness rule (results/battery/data.md: 4.76% of calib-mix lines hold
inline LaTeX, above the 2% bar), so it is reported but left out of H1.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load  # noqa: E402

H1_DOMAINS = ("python", "rust", "german")
ALL = H1_DOMAINS + ("math",)
CHUNKS_PER_DOC = 4


def ratio_ci(pairs: dict[str, tuple[np.ndarray, np.ndarray]], n_boot: int = 10000, seed: int = 0, per_doc: int = CHUNKS_PER_DOC):
    """pairs: domain -> (KL of the candidate, KL of the base) per chunk. Returns (R, lo, hi): the mean over domains
    of the per-domain geometric-mean ratio, with a percentile CI from resampling documents within each domain."""
    rng = np.random.default_rng(seed)
    logs = {d: np.log(a / b).reshape(-1, per_doc) for d, (a, b) in pairs.items()}
    point = float(np.mean([np.exp(x.mean()) for x in logs.values()]))
    boots = np.zeros(n_boot)
    for x in logs.values():
        idx = rng.integers(0, x.shape[0], size=(n_boot, x.shape[0]))
        boots += np.exp(x[idx].mean(axis=(1, 2)))
    boots /= len(logs)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def verdict(lo: float, hi: float) -> str:
    if hi <= 0.90:
        return "PASS (transfer)"
    if lo > 1.0:
        return "REGRESSION"
    if hi <= 1.0:
        return "PARTIAL transfer"
    return "no evidence of transfer"


def main() -> None:
    runs = load()
    get = lambda d, n: runs[f"batA-{d}-{n}-P3"]  # noqa: E731
    print("| domain | v1 KL | v2m KL | UD KL | v2m / v1 [95% CI] | v2m / UD [95% CI] | top-1 v1 / v2m / UD |")
    print("|---|---|---|---|---|---|---|")
    for d in ALL:
        c, m, u = get(d, "cand"), get(d, "v2m"), get(d, "UD-IQ3_XXS")
        r1 = ratio_ci({d: (m["chunks"], c["chunks"])})
        r2 = ratio_ci({d: (m["chunks"], u["chunks"])})
        tag = "" if d in H1_DOMAINS else " (near-calibration)"
        print(f"| {d}{tag} | {c['chunks'].mean():.5f} | {m['chunks'].mean():.5f} | {u['chunks'].mean():.5f} | "
              f"{r1[0]:.3f} [{r1[1]:.3f}, {r1[2]:.3f}] | {r2[0]:.3f} [{r2[1]:.3f}, {r2[2]:.3f}] | "
              f"{c['top']:.2f} / {m['top']:.2f} / {u['top']:.2f} |")
    R, lo, hi = ratio_ci({d: (get(d, "v2m")["chunks"], get(d, "cand")["chunks"]) for d in H1_DOMAINS})
    print(f"\nH1 ({', '.join(H1_DOMAINS)}): R = {R:.3f} [{lo:.3f}, {hi:.3f}] -> {verdict(lo, hi)}")


if __name__ == "__main__":
    main()
