"""Amendment 14 reading (CHANGELOG): v2m against mirror (UD's encoding made all-Metal) and against v2m-ud04 (UD's
gate_up in layers 0 and 4), paired over chunks on S, T1 and T2. Prints a Markdown table.

    python -I analysis/a14_report.py   (reads results/pilot/kl.tsv; run analysis/maptable.py first)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load, paired  # noqa: E402

RUN = {"S": "{}-S-P3", "T1": "final-{}-T1-P3", "T2": "final-{}-T2-P3"}
OTHERS = ("mirror", "v2m-ud04", "UD-IQ3_XXS")


def main() -> None:
    runs = load()
    print("| set | v2m | " + " | ".join(OTHERS) + " | " + " | ".join(f"v2m - {o}" for o in OTHERS) + " | mirror - UD-IQ3_XXS |")
    print("|---" * (3 + 2 * len(OTHERS)) + "|")
    for t, pat in RUN.items():
        m = runs[pat.format("v2m")]
        cells, diffs = [], []
        for o in OTHERS:
            r = runs.get(pat.format(o))
            cells.append(f"{r['chunks'].mean():.5f} ({r['top']:.2f}%)" if r else "n/a")
            diffs.append("{:+.5f} [{:+.5f}, {:+.5f}]".format(*paired(m["chunks"], r["chunks"])) if r else "n/a")
        mi, ud = runs.get(pat.format("mirror")), runs[pat.format("UD-IQ3_XXS")]
        diffs.append("{:+.5f} [{:+.5f}, {:+.5f}]".format(*paired(mi["chunks"], ud["chunks"])) if mi else "n/a")
        print(f"| {t} | {m['chunks'].mean():.5f} ({m['top']:.2f}%) | " + " | ".join(cells) + " | " + " | ".join(diffs) + " |")


if __name__ == "__main__":
    main()
