"""How much of v1's KL gap to UD-IQ3_XXS v2m closes, set by set, from the calibration-like sets out to fresh text.

    python -I analysis/gap_ud.py   (reads results/pilot/kl.tsv; prints Markdown)

Ratios are geometric means over chunk pairs with 95% CIs from analysis/battery_A.py's resampler (battery A: whole
documents of 4 chunks; every other set: single chunks). "Gap closed" is 1 - log(v2m / UD) / log(v1 / UD): the share
of v1's log-KL gap to UD that v2m removes. Descriptive only; it tests nothing registered.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery_A import ratio_ci  # noqa: E402
from compare import load  # noqa: E402

SETS = [("T1 (WikiText)", "final-{}-T1-P3", 1), ("T2 (Q8_0 reasoning traces)", "final-{}-T2-P3", 1)] + [
    (f"C {t} (Q8_0 task replies)", "batC-" + t + "-{}-P3", 1) for t in ("B1", "B2", "B3", "B4")] + [
    (f"A {d} (fresh)", "batA-" + d + "-{}-P3", 4) for d in ("math", "python", "rust", "german")]


def main() -> None:
    runs = load()
    print("| set | v2m / UD [95% CI] | v1 / UD [95% CI] | gap closed |\n|---|---|---|---|")
    for label, pat, per_doc in SETS:
        m, c, u = (runs[pat.format(n)]["chunks"] for n in ("v2m", "cand", "UD-IQ3_XXS"))
        a = ratio_ci({"x": (m, u)}, per_doc=per_doc)
        b = ratio_ci({"x": (c, u)}, per_doc=per_doc)
        print(f"| {label} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}] | "
              f"{1 - math.log(a[0]) / math.log(b[0]):.0%} |")


if __name__ == "__main__":
    main()
