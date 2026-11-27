"""Addendum H amendment 21 (exploratory): task battery B for c1 against ISTA IQ2_S and v2c against ISTA IQ3_S.

    python -I analysis/battery_B.py > /dev/null   (scores and caches every B-*.jsonl)
    python -I analysis/a21.py > results/a18/a21-tasks.md

Per suite: right counts and discordant pairs. Pooled B1+B3+B4 (H3's pool) and all four suites: paired difference
with one-sided 95% bounds from battery_B.h3 (exact mid-p on the discordant pairs). The bounds' width is the
precision statement: 300 items screen only for large differences.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import battery_B as bb  # noqa: E402

PAIRS = [("P2", "c1", "ista-IQ2_S"), ("P1", "v2c", "ista-IQ3_S")]


def main() -> None:
    cache = json.loads((ROOT / "results/battery/B-scores.json").read_text())
    its = {it["id"]: it for it in bb.bB.items()}
    out = ["# Amendment 21: task battery B, exploratory (`analysis/a21.py`)", ""]
    for tag, a, b in PAIRS:
        out += [f"## {tag}: {a} against {b}", "", f"| suite | n | {a} right | {b} right | {a} only | {b} only |",
                "|---|---|---|---|---|---|"]
        for t in ("B1", "B2", "B3", "B4"):
            ids = [i for i in its if its[i]["task"] == t]
            ra, rb = (sum(cache.get(x, {}).get(i, False) for i in ids) for x in (a, b))
            w = sum(cache.get(a, {}).get(i, False) and not cache.get(b, {}).get(i, False) for i in ids)
            l = sum(cache.get(b, {}).get(i, False) and not cache.get(a, {}).get(i, False) for i in ids)
            out.append(f"| {t} | {len(ids)} | {ra} | {rb} | {w} | {l} |")
        for label, suites in (("B1+B3+B4", ("B1", "B3", "B4")), ("all four", ("B1", "B2", "B3", "B4"))):
            ids = [i for i in its if its[i]["task"] in suites]
            missing = [i for i in ids if i not in cache.get(a, {}) or i not in cache.get(b, {})]
            if missing:
                out.append(f"\n{label}: {len(missing)} items unscored, no bound")
                continue
            d, lo, hi, w, l = bb.h3([cache[a][i] for i in ids], [cache[b][i] for i in ids])
            out.append(f"\n{label} (n={len(ids)}): {a} - {b} = {d:+.1f} points, one-sided 95% bounds [{lo:+.1f}, {hi:+.1f}], "
                       f"discordant {w} / {l}; lower bound above -5: {'yes' if lo > -5 else 'no'}")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
