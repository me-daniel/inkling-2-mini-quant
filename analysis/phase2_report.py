"""Writes results/pilot/phase2.md from results/pilot/kl.tsv and the step timestamps in logs/map.out."""

import csv
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAYERS = {"B1": (6, "0-5, before the loop"), "B2": (6, "6-11, looped"), "B3": (6, "12-17, looped"), "B4": (2, "18-19, after the loop"), "B5": (4, "20-23, after the loop, MXFP8 source")}


def main() -> None:
    kl = {r["run"]: r for r in csv.DictReader(open(ROOT / "results/pilot/kl.tsv"), delimiter="\t")}
    g = lambda run: float(kl[run]["mean_kl"])  # noqa: E731
    lines = ["# Pilot phase 2: block map and pass sweep", "", "Written by `analysis/phase2_report.py` from `results/pilot/kl.tsv`. KL against the Q8_0 reference on S (16 x 4,096 tokens).", "",
             "## Block map at P = 3 (one block changed, everything else Q8_0)", "",
             "| Block | Layers | 2-bit KL | per layer | 3-bit KL | per layer |", "|---|---|---|---|---|---|"]
    s2 = s3 = 0.0
    for b, (n, desc) in LAYERS.items():
        k2, k3 = g(f"map-{b}-2bit-S-P3"), g(f"map-{b}-3bit-S-P3")
        s2, s3 = s2 + k2, s3 + k3
        lines.append(f"| {b} | {desc} | {k2:.4f} | {k2 / n:.4f} | {k3:.4f} | {k3 / n:.5f} |")
    u2, u3 = g("sweep-2bit-S-P3"), g("sweep-3bit-S-P3")
    lines += ["", f"Additivity: sum of the five blocks against the uniform build. 2-bit {s2:.4f} vs {u2:.4f} (ratio {u2 / s2:.2f}); 3-bit {s3:.4f} vs {u3:.4f} (ratio {u3 / s3:.2f}).", "",
              "## Pass sweep (uniform builds, each against the reference at the same pass count)", "",
              "| Level | P = 2 | P = 3 | P = 4 |", "|---|---|---|---|"]
    for lvl in ("2bit", "3bit"):
        lines.append(f"| {lvl} | " + " | ".join(f"{g(f'sweep-{lvl}-S-P{p}'):.4f}" for p in (2, 3, 4)) + " |")
    stamps = [(datetime.strptime(m.group(1), "%H:%M:%S"), m.group(2)) for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\] (.*)", (ROOT / "logs/map.out").read_text())]
    dur = {}
    for (t0, what), (t1, _) in zip(stamps, stamps[1:]):
        m = re.match(r"perplexity (sweep-\w+-S-P(\d))", what)
        if m:
            dur.setdefault(int(m.group(2)), []).append((t1 - t0).total_seconds())
    mean = {p: sum(v) / len(v) for p, v in dur.items()}
    lines += ["", "Run time per sweep evaluation (mean of both levels), against the layer visits 12 + 12 P:", ""]
    for p in sorted(mean):
        lines.append(f"- P = {p}: {mean[p]:.0f} s, ratio to P = 3 {mean[p] / mean[3]:.2f} (visits ratio {(12 + 12 * p) / 48:.2f})")
    (ROOT / "results/pilot/phase2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
