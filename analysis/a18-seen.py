"""Amendment 18 part A, descriptive: KL on the seen sets S, T1 and T2 (P = 3) for our files, ISTA's and UD.

    python -I analysis/a18-seen.py > results/a18/seen.md

S selected our files, so these sets favour them; the registered comparison is A3 (results/a18/a18.md).
"""

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maptable import parse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FILES = [("c1 (8.9G)", "c1"), ("ISTA IQ2_S", "ista-IQ2_S"), ("ISTA Q2_K_XL", "ista-Q2_K_XL"),
         ("ISTA IQ3_XXS", "ista-IQ3_XXS"), ("ISTA IQ3_S", "ista-IQ3_S"), ("v2c (refined2)", "v2c"),
         ("Unsloth UD-IQ3_XXS", "UD-IQ3_XXS")]


def log(name: str, s: str) -> Path:
    return ROOT / "logs" / (f"ppl-{name}-S-P3.log" if s == "S" else f"ppl-final-{name}-{s}-P3.log")


print("## KL on the seen sets (P = 3; mean KL, top-1 agreement %)\n")
print("S selected our files, so these sets favour them. Built by `analysis/a18-seen.py`.\n")
print("| file | S | T1 | T2 |\n|---|---|---|---|")
for label, name in FILES:
    cells = []
    for s in ("S", "T1", "T2"):
        r = parse(log(name, s)) if log(name, s).exists() else None
        cells.append(f"{r['mean_kl']:.5f} ({r['same_top_p']:.1f})" if r else "missing")
    print(f"| {label} | " + " | ".join(cells) + " |")
