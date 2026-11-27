"""Amendment 18: where each benched file's weights live, from the load-only runs (scripts/a18-load.sh, -lv 5).

    python -I analysis/a18-load.py > results/a18/load.md
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [("v2c (refined2)", "v2c"), ("ISTA IQ3_S", "ista-IQ3_S"), ("c1 (8.9G)", "c1"), ("ISTA IQ2_S", "ista-IQ2_S")]
MIB = 1024 ** 2

print("## Placement (load-only runs at -lv 5, `scripts/a18-load.sh`; built by `analysis/a18-load.py`)\n")
print("| file | file bytes | layers offloaded | CPU_Mapped GB | Metal (MTL0_Mapped) GB |\n|---|---|---|---|---|")
for label, n in FILES:
    text = (ROOT / f"logs/load-a18-{n}.log").read_text(errors="replace")
    off = re.findall(r"offloaded (\d+/\d+) layers to GPU", text)[-1]
    cpu = sum(float(x) for x in re.findall(r"CPU_Mapped model buffer size =\s+([\d.]+) MiB", text))
    mtl = sum(float(x) for x in re.findall(r"MTL0_Mapped model buffer size =\s+([\d.]+) MiB", text))
    size = (ROOT / f"models/{n}.gguf").stat().st_size
    print(f"| {label} | {size:,} | {off} | {cpu * MIB / 1e9:.3f} | {mtl * MIB / 1e9:.3f} |")
