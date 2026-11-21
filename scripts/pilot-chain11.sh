#!/bin/sh
# After the memory incident and amendment 4: every heavy run takes the shared GPU lock. Order: the remaining swaps and
# gu-v2a, v2a on S, the per-pass controls, then the comparators. Everything resumes.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
step() { echo "[$(date '+%H:%M:%S')] $*"; }
for s in decompose v2a perpass3 macgate; do sh -n "$ROOT/scripts/$s.sh"; done
step "swaps and gu-v2a"; sh "$ROOT/scripts/decompose.sh"
step "v2a on S"; sh "$ROOT/scripts/v2a.sh"
step "H1-H5 per-pass controls"; sh "$ROOT/scripts/perpass3.sh"
while pgrep -f "scripts/fetch-(ext|one|chain7).sh" > /dev/null; do sleep 60; done
step "G/H6 comparators"; sh "$ROOT/scripts/macgate.sh"
step "chain11 done"
