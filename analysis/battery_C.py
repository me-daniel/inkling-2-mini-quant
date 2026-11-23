"""Amendment 15, component C (secondary): task-trace KL of v2m against v1 and UD-IQ3_XXS per task, chunks resampled.

    python -I analysis/battery_C.py   (reads results/pilot/kl.tsv rows batC-<task>-<model>-P3; prints Markdown)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery_A import ratio_ci  # noqa: E402
from compare import load  # noqa: E402


def main() -> None:
    runs = load()
    print("| task | chunks | v1 KL | v2m KL | UD KL | v2m / v1 [95% CI] | v2m / UD [95% CI] |")
    print("|---|---|---|---|---|---|---|")
    for t in ("B1", "B2", "B3", "B4"):
        k = f"batC-{t}-{{}}-P3"
        if k.format("v2m") not in runs:
            continue
        c, m, u = (runs[k.format(n)] for n in ("cand", "v2m", "UD-IQ3_XXS"))
        r1 = ratio_ci({t: (m["chunks"], c["chunks"])}, per_doc=1)
        r2 = ratio_ci({t: (m["chunks"], u["chunks"])}, per_doc=1)
        print(f"| {t} | {len(m['chunks'])} | {c['chunks'].mean():.5f} | {m['chunks'].mean():.5f} | {u['chunks'].mean():.5f} | "
              f"{r1[0]:.3f} [{r1[1]:.3f}, {r1[2]:.3f}] | {r2[0]:.3f} [{r2[1]:.3f}, {r2[2]:.3f}] |")


if __name__ == "__main__":
    main()
