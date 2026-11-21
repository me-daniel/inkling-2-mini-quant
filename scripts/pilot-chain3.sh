#!/bin/sh
# After pilot-chain2.sh (Addendum D map) and the optimized-Q2_0 encoding: the Addendum E map, then phase 4.
# Stops before any upload.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
while pgrep -f "scripts/pilot-chain2.sh" > /dev/null || pgrep -f "scripts/q2opt-all.sh" > /dev/null; do sleep 30; done
grep -q 'done' "$ROOT/logs/chain2.out" || { echo "map2 did not finish cleanly" >&2; exit 1; }
[ -f "$ROOT/models/q2opt/L23.gguf" ] || { echo "optimized Q2_0 encoding incomplete" >&2; exit 1; }
sh "$ROOT/scripts/map3.sh"
sh "$ROOT/scripts/phase4.sh"
