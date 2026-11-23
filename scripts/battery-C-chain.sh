#!/bin/sh
# Waits until the Q8_0 B replies are complete (300 lines), builds the C texts, then scores them (battery-C.sh).
set -eu
cd "$(dirname "$0")/.."
while [ "$(wc -l < results/battery/B-q8.jsonl 2>/dev/null || echo 0)" -lt 300 ]; do sleep 60; done
.venv/bin/python -I scripts/battery-C.py
sh scripts/battery-C.sh
