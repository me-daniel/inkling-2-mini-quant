"""Collects every llama-perplexity KL log in logs/ into results/pilot/kl.tsv (one row per run, per-chunk values).

The chunk lines of a KL log hold running means; every chunk scores the same number of positions, so the mean of
chunk k is k * m_k - (k - 1) * m_(k-1). Also records the wall time of each run (for the pass-count check).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNK = re.compile(r"^\s*(\d+)\s+[\d.]+\s+±\s+[\d.]+\s+-?[\d.]+\s+±\s+[\d.]+\s+([\d.]+)\s+±\s+[\d.]+\s+[\d.]+\s+±\s+[\d.]+\s+%\s+([\d.]+)\s+±")


def parse(log: Path) -> dict | None:
    text = log.read_text(errors="replace")
    mean = re.search(r"Mean\s+KLD:\s+([\d.]+)\s+±\s+([\d.]+)", text)
    if not mean:
        return None
    running = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in map(CHUNK.match, text.splitlines()) if m]
    per_chunk, prev = [], 0.0
    for k, m, _ in running:
        per_chunk.append(k * m - (k - 1) * prev)
        prev = m
    top = re.search(r"Same top p:\s+([\d.]+)\s+±\s+([\d.]+)", text)
    total = re.search(r"total time\s*=\s*([\d.]+) ms", text)
    return {
        "run": log.stem.removeprefix("ppl-"),
        "mean_kl": float(mean.group(1)),
        "se": float(mean.group(2)),
        "same_top_p": float(top.group(1)) if top else float("nan"),
        "seconds": float(total.group(1)) / 1000 if total else float("nan"),
        "per_chunk": per_chunk,
    }


def collect() -> list[dict]:
    rows = [r for log in sorted((ROOT / "logs").glob("ppl-*.log")) if (r := parse(log))]
    out = ROOT / "results/pilot/kl.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("run\tmean_kl\tse\tsame_top_p\tseconds\tper_chunk\n")
        for r in rows:
            f.write(f"{r['run']}\t{r['mean_kl']:.6f}\t{r['se']:.6f}\t{r['same_top_p']:.3f}\t{r['seconds']:.1f}\t"
                    + ",".join(f"{x:.6f}" for x in r["per_chunk"]) + "\n")
    return rows


if __name__ == "__main__":
    for r in collect():
        print(f"{r['run']:28s} KL {r['mean_kl']:.5f} ± {r['se']:.5f}  top-1 {r['same_top_p']:.2f}%  {r['seconds']:.0f} s  chunks {len(r['per_chunk'])}")
