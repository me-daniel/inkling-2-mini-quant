#!/bin/sh
# Addendum H amendment 21 (exploratory): battery B replies for c1, ISTA IQ2_S and ISTA IQ3_S (v2c's are on file
# from amendment 16), after amendment 18's parts B and C. Then scoring and the report.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
while pgrep -f 'a18-(score|swaps)\.sh' > /dev/null; do sleep 60; done
SPECS="c1:models/c1.gguf ista-IQ2_S:models/ista-IQ2_S.gguf ista-IQ3_S:models/ista-IQ3_S.gguf" sh scripts/battery-B.sh
.venv/bin/python -I analysis/battery_B.py > /dev/null
.venv/bin/python -I analysis/a21.py | tee results/a18/a21-tasks.md
