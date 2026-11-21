#!/bin/sh
# Runs the rest of the pilot back to back once the block map and pass sweep are done:
# the Q6_K non-expert entry, a rebuild with the full patch series, then the per-pass runs.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
while pgrep -f "scripts/map.sh" > /dev/null; do sleep 30; done
grep -q 'done' "$ROOT/logs/map.out" || { echo "map.sh did not finish cleanly" >&2; exit 1; }
sh "$ROOT/scripts/phase2b.sh"
sh "$ROOT/scripts/build-llama.sh" 2>&1 | grep -E 'loop-|wrote|failed'
sh "$ROOT/scripts/perpass.sh"
