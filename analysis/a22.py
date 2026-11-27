"""Addendum H amendment 22: v2c with UD-IQ3_XXS's NVFP4 downs in layers 0-19 (diagnostic, never a release).

    python -I analysis/a22.py   (logs/ppl-a22-nv-S-P3.log, logs/ppl-final-a22-nv-{T1,T2}-P3.log)

Paired chunk bootstrap (analysis/a19.py paired_delta) against v2c and against UD-IQ3_XXS on S, T1 and T2.
Registered rule on S only: KL(a22-nv) - KL(v2c) with CI upper below 0 opens a v3 track. The rest is descriptive.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from a19 import paired_delta  # noqa: E402
from maptable import parse  # noqa: E402


def log(name: str, s: str) -> Path:
    return ROOT / ("logs/ppl-%s-S-P3.log" % name if s == "S" else "logs/ppl-final-%s-%s-P3.log" % (name, s))


def main() -> None:
    lines = ["# Amendment 22: NVFP4-down diagnostic (v2c + UD's NVFP4 downs, layers 0-19)", "",
             "Built by `analysis/a22.py`. A donor build: diagnostic only. Mean KL (top-1 %); paired chunk CI.", "",
             "| set | a22-nv | v2c | UD-IQ3_XXS | a22-nv - v2c | a22-nv - UD |", "|---|---|---|---|---|---|"]
    rule = None
    for s in ("S", "T1", "T2"):
        r = {n: parse(log(n, s)) for n in ("a22-nv", "v2c", "UD-IQ3_XXS")}
        a = np.array(r["a22-nv"]["per_chunk"])
        cells = []
        for b in ("v2c", "UD-IQ3_XXS"):
            d, lo, hi = paired_delta(a, np.array(r[b]["per_chunk"]))
            cells.append(f"{d:+.5f} [{lo:+.5f}, {hi:+.5f}]")
            if s == "S" and b == "v2c":
                rule = hi < 0
        fmt = lambda n: f"{r[n]['mean_kl']:.5f} ({r[n]['same_top_p']:.1f})"
        lines.append(f"| {s} ({a.size} chunks) | {fmt('a22-nv')} | {fmt('v2c')} | {fmt('UD-IQ3_XXS')} | {cells[0]} | {cells[1]} |")
    lines += ["", "Registered rule (S, against v2c): " + ("CI upper below 0, so a v3 track with our own NVFP4 downs opens."
              if rule else "CI not below 0, so the track stops.")]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
