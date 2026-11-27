"""Amendment 18 parts B and C (descriptive).

    python -I analysis/a18-bc.py > results/a18/bc.md

B: speed from scripts/bench-v2m.sh, runs T = 6..9 (each pair back to back in both orders). Per file and test, the mean
of its two runs (one per order) and the spread between them.
C: reciprocal same-type swaps between v2c and ISTA IQ3_S on S and T1, each against its recipient; paired chunk
bootstrap from analysis/a19.py. Swapped payloads carry their encoder, calibration and pipeline together.
"""

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from a19 import paired_delta  # noqa: E402
from maptable import parse  # noqa: E402

R = ROOT / "results/pilot"
PAIRS = [((6, 7), "v2c", "ista-IQ3_S"), ((8, 9), "c1", "ista-IQ2_S")]
SWAPS = [("v2c-gu6i", "v2c", "ISTA's gate_up in 6 layers (6, 11, 13, 17, 20, 22) into v2c"),
         ("ista-gu6v", "ista-IQ3_S", "v2c's gate_up in those 6 layers into ISTA"),
         ("v2c-dni", "v2c", "ISTA's IQ4_NL downs in 21 layers into v2c"),
         ("ista-dnv", "ista-IQ3_S", "v2c's IQ4_NL downs in those 21 layers into ISTA"),
         ("v2c-embi", "v2c", "ISTA's BF16 token_embd into v2c (added before data, labelled)")]


def speeds(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\|.*\|\s*([\w @]+?)\s*\|\s*([\d.]+) ± ([\d.]+) \|$", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def peak(path: Path) -> float:
    m = re.search(r"(\d+)\s+peak memory footprint", path.read_text())
    return int(m.group(1)) / 1e9 if m else float("nan")


def log(name: str, s: str) -> Path:
    return ROOT / "logs" / (f"ppl-{name}-S-P3.log" if s == "S" else f"ppl-final-{name}-{s}-P3.log")


def main() -> None:
    out = ["# Amendment 18 parts B and C (built by `analysis/a18-bc.py`)", "",
           "## B. Speed (llama-bench, t/s; mean of the two orders, spread in brackets)", ""]
    for (t1, t2), a, b in PAIRS:
        rows = {n: [speeds(R / f"depth{t}-{n}.txt") for t in (t1, t2)] for n in (a, b)}
        tests = list(rows[a][0])
        out += [f"Runs {t1} ({a} first) and {t2} ({b} first).", "",
                "| test | " + " | ".join(rows) + f" | {a} / {b} |", "|---|---|---|---|"]
        for t in tests:
            m = {n: np.mean([r[t] for r in rows[n]]) for n in rows}
            sp = {n: abs(rows[n][0][t] - rows[n][1][t]) for n in rows}
            out.append(f"| {t} | " + " | ".join(f"{m[n]:.1f} ({sp[n]:.1f})" for n in rows) + f" | {m[a] / m[b]:.3f} |")
        pk = {n: [peak(ROOT / f"logs/depth{t}-{n}.log") for t in (t1, t2)] for n in (a, b)}
        out += ["", "Process peak memory footprint of the depth runs (GB, per order; mapped Metal buffers are not counted, see `results/a18/load.md` for GPU bytes): " +
                "; ".join(f"{n} {pk[n][0]:.2f} / {pk[n][1]:.2f}" for n in pk), ""]
    out += ["## C. Reciprocal swaps (KL at P = 3; swap minus recipient, paired chunk bootstrap 95% CI)", "",
            "| swap | what | S | S Δ [CI] | T1 | T1 Δ [CI] |", "|---|---|---|---|---|---|"]
    for n, base, what in SWAPS:
        cells = []
        for s in ("S", "T1"):
            x, y = parse(log(n, s)), parse(log(base, s))
            d, lo, hi = paired_delta(np.array(x["per_chunk"]), np.array(y["per_chunk"]))
            cells += [f"{x['mean_kl']:.5f}", f"{d:+.5f} [{lo:+.5f}, {hi:+.5f}]"]
        out.append(f"| {n} | {what} | " + " | ".join(cells) + " |")
    out += ["", "Recipients: " + "; ".join(
        f"{b} S {parse(log(b, 'S'))['mean_kl']:.5f}, T1 {parse(log(b, 'T1'))['mean_kl']:.5f}" for b in ("v2c", "ista-IQ3_S"))]
    print("\n".join(out))


if __name__ == "__main__":
    main()
