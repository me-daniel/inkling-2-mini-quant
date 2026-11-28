"""Addendum H amendment 23, part A: the native-NVFP4 base files.

    python -I analysis/a23.py

v2c-A smoke check (S, against amendment 22's a22-nv: registered within 1e-5) and c1-A against c1 on S, T1, T2
(paired chunk bootstrap, analysis/a19.py). Registered rule on S: CI upper below 0 -> c1-A is the 9 GB base.
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
    a, b = parse(log("v2c-A", "S")), parse(log("a22-nv", "S"))
    gap = abs(a["mean_kl"] - b["mean_kl"])
    lines = ["# Amendment 23, part A: native-NVFP4 base files", "", "Built by `analysis/a23.py`.", "",
             f"v2c-A on S: {a['mean_kl']:.6f} (a22-nv {b['mean_kl']:.6f}; gap {gap:.1e}, registered bound 1e-5: "
             f"{'pass' if gap <= 1e-5 else 'FAIL'}).", "",
             "| set | c1-A | c1 | c1-A - c1 |", "|---|---|---|---|"]
    rule = None
    for s in ("S", "T1", "T2"):
        x, y = parse(log("c1-A", s)), parse(log("c1", s))
        d, lo, hi = paired_delta(np.array(x["per_chunk"]), np.array(y["per_chunk"]))
        if s == "S":
            rule = hi < 0
        lines.append(f"| {s} | {x['mean_kl']:.5f} ({x['same_top_p']:.1f}) | {y['mean_kl']:.5f} ({y['same_top_p']:.1f}) | "
                     f"{d:+.5f} [{lo:+.5f}, {hi:+.5f}] ({d / y['mean_kl']:+.1%}) |")
    lines += ["", "Registered rule (S): " + ("CI upper below 0, so c1-A is the 9 GB base." if rule else "CI not below 0; c1 stays the 9 GB base.")]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
