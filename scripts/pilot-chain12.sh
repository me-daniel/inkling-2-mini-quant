#!/bin/sh
# Replaces chain11 (its fetch wait matched the sibling repo's scripts/fetch-ext.sh). Process checks use this repo's
# full path. Waits for the running decompose.sh, then v2a on S, the per-pass controls and the comparators; macgate
# itself waits for each comparator file, so no download wait is needed here.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
step() { echo "[$(date '+%H:%M:%S')] $*"; }
for s in decompose v2a perpass3 macgate; do sh -n "$ROOT/scripts/$s.sh"; done
while pgrep -f "$ROOT/scripts/decompose.sh" > /dev/null; do sleep 20; done
sh "$ROOT/scripts/decompose.sh"   # finishes gu-v2a if the running copy stopped before it; resumes, so no rerun
step "v2a on S"; sh "$ROOT/scripts/v2a.sh"
step "H1-H5 per-pass controls"; sh "$ROOT/scripts/perpass3.sh"
step "G/H6 comparators"; sh "$ROOT/scripts/macgate.sh"
step "chain12 done"
