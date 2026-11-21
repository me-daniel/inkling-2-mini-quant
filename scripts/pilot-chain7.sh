#!/bin/sh
# Addendum H, tonight: the 2 x 2 on S, the per-pass controls, then the comparators once downloaded (no speed runs).
# Downloads run in a separate process (scripts/fetch-ext.sh, then fetch-one.sh for bartowski Q2_K).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
step() { echo "[$(date '+%H:%M:%S')] $*"; }
for s in hybrid mirror perpass3 macgate; do sh -n "$ROOT/scripts/$s.sh"; done
DRY=1 sh "$ROOT/scripts/perpass3.sh" > /dev/null
step "H0 hybrid"; sh "$ROOT/scripts/hybrid.sh"
step "H0 mirror"; sh "$ROOT/scripts/mirror.sh"
step "H1-H5 per-pass controls"; sh "$ROOT/scripts/perpass3.sh"
while pgrep -f "scripts/fetch-(ext|one).sh" > /dev/null; do sleep 60; done
step "G/H6 comparators"; sh "$ROOT/scripts/macgate.sh"
step "chain7 done"
