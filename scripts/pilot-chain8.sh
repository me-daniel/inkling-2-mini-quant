#!/bin/sh
# Replaces chain7 after H0: the layout decomposition first (it decides v2), then the per-pass controls, then the comparators.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
step() { echo "[$(date '+%H:%M:%S')] $*"; }
for s in decompose perpass3 macgate; do sh -n "$ROOT/scripts/$s.sh"; done
step "H amendment 2 decomposition"; sh "$ROOT/scripts/decompose.sh"
step "H1-H5 per-pass controls"; sh "$ROOT/scripts/perpass3.sh"
while pgrep -f "scripts/fetch-(ext|one|chain7).sh" > /dev/null; do sleep 60; done
step "G/H6 comparators"; sh "$ROOT/scripts/macgate.sh"
step "chain8 done"
