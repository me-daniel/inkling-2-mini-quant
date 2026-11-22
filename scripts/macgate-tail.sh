#!/bin/sh
# Waits until macgate.sh has scored prithiv-Q2_K on T2, stops it (bartowski Q2_K, 11.88 GB, is outside the 11.6 GB
# claim of amendment 11 and was dropped on 22 Nov), then runs macgate's report steps without bart-Q2_K.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); R="$ROOT/results/pilot"; PY="$ROOT/.venv/bin/python -I"
until [ -f "$ROOT/logs/ppl-final-prithiv-Q2_K-T2-P3.log" ] && grep -q 'Mean *KLD' "$ROOT/logs/ppl-final-prithiv-Q2_K-T2-P3.log"; do sleep 60; done
sleep 5; pkill -f "$ROOT/scripts/macgate.sh" || pkill -f "sh scripts/macgate.sh" || true
$PY "$ROOT/analysis/maptable.py" > /dev/null
$PY "$ROOT/analysis/compare.py" cand bart-IQ2_S atomic-AD-IQ2_S apex-I-Nano isvalorum-NanoPlus prithiv-Q2_K | tee "$R/mac-quality.txt"
$PY "$ROOT/analysis/compare.py" cand bart-IQ2_M | grep -v 'quality gate' > "$R/mac-context-IQ2_M.txt"
$PY "$ROOT/analysis/perpass_report.py" > /dev/null
echo "[$(date '+%H:%M:%S')] macgate tail done"
