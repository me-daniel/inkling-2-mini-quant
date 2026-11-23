"""Amendment 13 reading (CHANGELOG): v2m against v2r, v2a, UD-IQ3_XXS and v1 (cand) on S, T1 and T2, paired over
chunks, plus the share of the v2a to UD gap closed, and the registered support rule.

    python -I analysis/v2m_report.py   (reads results/pilot/kl.tsv; run analysis/maptable.py first)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load, paired  # noqa: E402

RUN = {"S": "{}-S-P3", "T1": "final-{}-T1-P3", "T2": "final-{}-T2-P3"}


def main() -> None:
    runs = load()
    get = lambda name, t: runs.get(RUN[t].format(name))  # noqa: E731
    share = {}
    print("| set | v2m | v2r | v2a | UD | v2m - v2r | v2m - UD | v2m - v1 | share v2m / v2r | top-1 v2m / v2r / UD |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for t in RUN:
        m, r, a, u, c = (get(n, t) for n in ("v2m", "v2r", "v2a", "UD-IQ3_XXS", "cand"))
        mean = lambda x: x["chunks"].mean()  # noqa: E731
        ci = lambda x, y: "{:+.5f} [{:+.5f}, {:+.5f}]".format(*paired(x["chunks"], y["chunks"]))  # noqa: E731
        sm = (mean(a) - mean(m)) / (mean(a) - mean(u))
        sr = (mean(a) - mean(r)) / (mean(a) - mean(u))
        share[t] = (sm, sr, paired(m["chunks"], r["chunks"]))
        print(f"| {t} | {mean(m):.5f} | {mean(r):.5f} | {mean(a):.5f} | {mean(u):.5f} | {ci(m, r)} | {ci(m, u)} | "
              f"{ci(m, c) if c else 'n/a'} | {sm:.2f} / {sr:.2f} | {m['top']:.2f} / {r['top']:.2f} / {u['top']:.2f} |")
    t2 = share["T2"]
    ok = t2[0] >= 2 * t2[1] and t2[2][0] < 0 and share["S"][0] >= 0.6 and share["T1"][0] >= 0.6
    print(f"\nrule: T2 share {t2[0]:.2f} >= 2 x {t2[1]:.2f} = {2 * t2[1]:.2f}, v2m - v2r on T2 {t2[2][0]:+.5f} < 0, "
          f"S share {share['S'][0]:.2f} and T1 share {share['T1'][0]:.2f} >= 0.6 -> {'SUPPORTED' if ok else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
