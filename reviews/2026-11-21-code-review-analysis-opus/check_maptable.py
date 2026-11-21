"""analysis/maptable.py: chunk lines the regex drops, on every real KL log and on synthetic logs; the seconds column.
CPU only, reads logs/ and results/pilot/kl.tsv, writes only to a temp dir.

    .venv/bin/python -I reviews/2026-11-21-code-review-analysis-opus/check_maptable.py
"""

import csv
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
import maptable  # noqa: E402

print("== real logs: chunks declared by llama-perplexity vs chunk lines kept ==")
n_logs, bad, gaps = 0, [], []
for log in sorted((ROOT / "logs").glob("ppl-*.log")):
    r = maptable.parse(log)
    if not r:
        continue
    n_logs += 1
    text = log.read_text(errors="replace")
    decl = int(re.search(r"computing over (\d+) chunks", text).group(1))
    idx = [int(m.group(1)) for m in map(maptable.CHUNK.match, text.splitlines()) if m]
    if idx != list(range(1, decl + 1)):
        bad.append(f"{log.name}: declared {decl}, kept {len(idx)}, first kept chunk {idx[0]}")
    else:
        gaps.append(abs(np.mean(r["per_chunk"]) - r["mean_kl"]))
print(f"{n_logs} KL logs; {len(bad)} with dropped chunk lines: {bad}")
print(f"on the other {len(gaps)}: max |mean(per_chunk) - Mean KLD| = {max(gaps):.1e}")
lines = (ROOT / "logs/ppl-floor-q8-S-P3.log").read_text().splitlines()
print("dropped lines in the floor log:", [l.strip()[:110] for l in lines if re.match(r"^\s*[12]\s+\d", l)])

print("\n== synthetic log, 6 chunks, line 4 unparseable ==")
kl = np.array([0.020, 0.030, 0.025, 0.015, 0.040, 0.010])
run = np.cumsum(kl) / np.arange(1, 7)


def line(k, m, se="0.00100", kl_s=None):
    kl_s = kl_s if kl_s is not None else f"{m:.5f}"
    return (f"  {k:2d}       6.1392 ±    0.0794       0.00030 ±    {se}       {kl_s} ±    0.00013     0.001 ±  0.000 %"
            f"    100.000 ±  0.000 %")


for label, kw in (("an SE printed as nan", {"se": "nan"}), ("a running KL printed as -0.00000", {"kl_s": "-0.00000"})):
    body = [line(k, m, **kw) if k == 4 else line(k, m) for k, m in zip(range(1, 7), run)]
    body += [f"Mean    KLD:   {run[-1]:.6f} ±   0.000100", "Same top p: 99.0 ± 0.1 %"]
    p = Path(tempfile.mkdtemp()) / "ppl-x.log"
    p.write_text("\n".join(body) + "\n")
    got = maptable.parse(p)["per_chunk"]
    print(f"{label}: true {kl.tolist()}, parsed {np.round(got, 4).tolist()}, n {len(got)}, "
          f"mean {np.mean(got):.5f} vs true {kl.mean():.5f}; chunk 5 should be {kl[4]:.3f} (or {kl[3] + kl[4]:.3f} for 4+5)")

print("\n== seconds column ==")
rows = list(csv.DictReader(open(ROOT / "results/pilot/kl.tsv"), delimiter="\t"))
print(f"kl.tsv rows {len(rows)}, seconds = nan in {sum(math.isnan(float(r['seconds'])) for r in rows)}; "
      f"logs with a 'total time' line: {sum('total time' in l.read_text(errors='replace') for l in (ROOT / 'logs').glob('ppl-*.log'))}")
