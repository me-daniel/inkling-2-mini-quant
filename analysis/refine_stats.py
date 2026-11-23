"""What the refiner did in a build, from its per-group JSON logs (review points of 23 Nov, Astra and Fable): per layer,
experts refined and skipped (and why), accepted moves, moves blocked by the output-gain floor and their share, the
in-sample error before and after (J under G relative to tr(W G W^T)), the aggregate output gain, and the largest gap
between the tracked J (sum of accepted dJ) and J recomputed from E (an exactness check of the incremental update).

    python -I analysis/refine_stats.py [PREFIX]   (default models/refined/gu-refined-mix-; prints Markdown)
"""

import glob
import json
import sys

import numpy as np


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "models/refined/gu-refined-mix-"
    rows, tot = [], np.zeros(5)
    for f in sorted(glob.glob(prefix + "*.gguf.json")):
        for layer, x in json.load(open(f))["layers"].items():
            ex = x["experts"]
            done = [e for e in ex if e.get("sweeps")]
            skipped = [e for e in ex if not e.get("sweeps")]
            acc = sum(s["accepted"] for e in done for s in e["sweeps"])
            blk = sum(s["gain_blocked"] for e in done for s in e["sweeps"])
            r0 = np.median([e["sweeps"][0]["rel_before"] for e in done if e["sweeps"][0]["rel_before"] is not None])
            r1 = np.median([e["sweeps"][-1]["rel_after"] for e in done if e["sweeps"][-1]["rel_after"] is not None])
            g0 = np.median([e["sweeps"][0]["ogain_before"] for e in done])
            g1 = np.median([e["sweeps"][-1]["ogain_after"] for e in done])
            gap = max(abs(s["J_tracked"] - s["J_after"]) / max(s["J_after"], 1e-30) for e in done for s in e["sweeps"])
            why = sorted({e.get("skipped", "") for e in skipped} - {""})
            rows.append((int(layer), x["type"], len(done), len(skipped), acc, blk, r0, r1, g0, g1, gap,
                         "; ".join(why)[:40]))
            tot += [len(done), len(skipped), acc, blk, 1]
    rows.sort()
    print("| layer | type | refined | skipped | accepted moves | gain-blocked | blocked share | median rel. error before / after | median gain before / after | max tracked-J gap |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:,} | {r[5]:,} | {r[5] / max(r[4] + r[5], 1):.2%} | "
              f"{r[6]:.4f} / {r[7]:.4f} | {r[8]:.4f} / {r[9]:.4f} | {r[10]:.1e} |")
    print(f"\n{int(tot[4])} layers: {int(tot[0])} experts refined, {int(tot[1])} skipped (under the token minimum or "
          f"empty Gram); {int(tot[2]):,} moves accepted, {int(tot[3]):,} blocked by the gain floor "
          f"({tot[3] / max(tot[2] + tot[3], 1):.2%} of candidate moves that lowered the error).")


if __name__ == "__main__":
    main()
