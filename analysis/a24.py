"""Addendum H amendment 24 (B): the loop-aware gate_up exchange on v2c-A, scored on DV.

    python -I analysis/a24.py select   prints D4, the four of d19..d23 with the lowest G against v2c-A (for scripts/a24.sh)
    python -I analysis/a24.py          report (Markdown)

G as in analysis/a18.py (document bootstrap, 4 chunks per document, 10,000 resamples, seed 0). S, T1, T2: paired chunk
bootstrap (analysis/a19.py).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from a18 import DOMAINS, g_stat, outcome  # noqa: E402
from a19 import paired_delta  # noqa: E402
from maptable import parse  # noqa: E402

BASE = "v2c-A"
DEMOTE = [19, 20, 21, 22, 23]


def dv(name: str) -> dict[str, np.ndarray]:
    out = {}
    for d in DOMAINS:
        r = parse(ROOT / f"logs/ppl-dv-{d}-{name}-P3.log")
        if r is None or len(r["per_chunk"]) != 48:
            raise SystemExit(f"{d} {name}: no complete DV log")
        out[d] = np.array(r["per_chunk"])
    return out


def g(a: str, b: str = BASE) -> tuple[float, float, float]:
    x, y = dv(a), dv(b)
    return g_stat({d: (x[d], y[d]) for d in DOMAINS})


def d4() -> list[int]:
    return sorted(sorted(DEMOTE, key=lambda L: g(f"b-d{L}")[0])[:4])


def tset(name: str, s: str) -> np.ndarray | None:
    p = ROOT / ("logs/ppl-%s-S-P3.log" % name if s == "S" else "logs/ppl-final-%s-%s-P3.log" % (name, s))
    r = parse(p) if p.exists() else None
    return None if r is None else np.array(r["per_chunk"])


def main() -> None:
    if sys.argv[1:] == ["select"]:
        print(" ".join(map(str, d4())))
        return
    sel = d4()
    fmt = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"  # noqa: E731
    lines = ["# Amendment 24 (B): loop-aware gate_up exchange on v2c-A", "", "Built by `analysis/a24.py`. DV, P = 3.", "",
             "| file | change | G vs v2c-A [95% CI] |", "|---|---|---|",
             f"| b-U | gate_up 6, 7, 8, 12 IQ2_S → IQ3_XXS, +169.9 MB | {fmt(g('b-U'))} |",
             f"| b-Uc | gate_up 1, 2, 3, 5 IQ2_S → IQ3_XXS, +169.9 MB | {fmt(g('b-Uc'))} |"]
    lines += [f"| b-d{L} | gate_up {L} IQ3_XXS → IQ2_S, −42.5 MB | {fmt(g(f'b-d{L}'))} |" for L in DEMOTE]
    x = g("b-X")
    pred = np.exp(np.log(g("b-U")[0]) + sum(np.log(g(f"b-d{L}")[0]) for L in sel))
    adopt = x[2] < 1
    lines += [f"| b-X | U + demotions {', '.join(map(str, sel))}, ±0 bytes | {fmt(x)} |", "",
              f"D4 by rule (lowest G of the five demotions): {', '.join(map(str, sel))}.", "",
              f"Primary: G(X / v2c-A) = {fmt(x)}. "
              + ("Upper bound below 1: X becomes the 11.5 GB base." if adopt else "Upper bound not below 1: v2c-A stays the base."),
              f"Release margin (amendment 18's rule, G ≤ 0.97 and upper bound below 1): {outcome(*x, 'b-X', 'v2c-A')}.", "",
              "Descriptive:", "",
              f"- Thesis contrast G(U / Uc), paired: {fmt(g('b-U', 'b-Uc'))} (below 1: the loop promotion bought more).",
              f"- Additivity: predicted G(X) from the singles {pred:.3f}, measured {x[0]:.3f}.", "",
              "| set | b-X | v2c-A | b-X − v2c-A |", "|---|---|---|---|"]
    for s in ("S", "T1", "T2"):
        a, b = tset("b-X", s), tset(BASE, s)
        if a is None or b is None:
            lines.append(f"| {s} | missing | | |")
            continue
        dlt, lo, hi = paired_delta(a, b)
        lines.append(f"| {s} | {a.mean():.5f} | {b.mean():.5f} | {dlt:+.5f} [{lo:+.5f}, {hi:+.5f}] ({dlt / b.mean():+.1%}) |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
