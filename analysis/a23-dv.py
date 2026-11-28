"""Amendment 23: baselines on the development set DV (selection set; A4 stays unbuilt).

    python -I analysis/a23-dv.py   (reads logs/ppl-dv-<domain>-<name>-P3.log; prints Markdown)

Per domain the mean KL against Q8_0 (P = 3, -c 1024, 48 chunks, second half scored). G = exp(mean over domains of the
per-domain mean log chunk ratio), 95% CI from analysis/a18.py's document bootstrap (4 chunks per document, 10,000
resamples, seed 0). Chunk values are telescoped from llama-perplexity's 5-decimal running means (as for A3).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from a18 import DOMAINS, g_stat  # noqa: E402
from maptable import parse  # noqa: E402

NAMES = ["v2c", "v2c-A", "UD-IQ3_XXS", "c1", "c1-A", "UD-IQ2_XXS"]
PAIRS = [("v2c-A", "UD-IQ3_XXS"), ("v2c-A", "v2c"), ("c1", "UD-IQ2_XXS"), ("c1-A", "c1"), ("c1-A", "UD-IQ2_XXS")]


def chunks(name: str, d: str) -> np.ndarray:
    r = parse(ROOT / f"logs/ppl-dv-{d}-{name}-P3.log")
    if r is None or len(r["per_chunk"]) != 48:
        raise SystemExit(f"{d} {name}: no complete DV log")
    return np.array(r["per_chunk"])


def main() -> None:
    kl = {(n, d): chunks(n, d) for n in NAMES for d in DOMAINS}
    lines = ["# Amendment 23: baselines on DV", "", "Built by `analysis/a23-dv.py`. Mean KL against Q8_0, P = 3.", "",
             "| file | " + " | ".join(DOMAINS) + " |", "|---|" + "---|" * len(DOMAINS)]
    lines += [f"| {n} | " + " | ".join(f"{kl[n, d].mean():.5f}" for d in DOMAINS) + " |" for n in NAMES]
    lines += ["", "| a / b | G [95% CI] | per domain |", "|---|---|---|"]
    for a, b in PAIRS:
        g, lo, hi = g_stat({d: (kl[a, d], kl[b, d]) for d in DOMAINS})
        per = ", ".join(f"{d} {np.exp(np.mean(np.log(kl[a, d] / kl[b, d]))):.3f}" for d in DOMAINS)
        lines.append(f"| {a} / {b} | {g:.3f} [{lo:.3f}, {hi:.3f}] | {per} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
