#!/bin/sh
# Reordered after the v2a build: amendment 3 (swaps, v2a on S) first, then the rest of the per-pass controls, then
# the comparators. Every step resumes, so nothing finished is redone.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
step() { echo "[$(date '+%H:%M:%S')] $*"; }
for s in decompose v2a perpass3 macgate; do sh -n "$ROOT/scripts/$s.sh"; done
rm -f "$ROOT/logs/gpu.lock"
step "amendment 3 swaps"; sh "$ROOT/scripts/decompose.sh"
step "v2a on S"; sh "$ROOT/scripts/v2a.sh"
step "H1-H5 per-pass controls"; sh "$ROOT/scripts/perpass3.sh"
while pgrep -f "scripts/fetch-(ext|one|chain7).sh" > /dev/null; do sleep 60; done
step "G/H6 comparators"; sh "$ROOT/scripts/macgate.sh"
step "chain10 done"
