#!/bin/sh
# Addendum H amendment 3 on the GPU, right after chain8's per-pass controls; holds logs/gpu.lock so macgate waits.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
step() { echo "[$(date '+%H:%M:%S')] $*"; }
touch "$ROOT/logs/gpu.lock"
trap 'rm -f "$ROOT/logs/gpu.lock"' EXIT
until grep -q "perpass3 done" "$ROOT/logs/chain8.out" || ! pgrep -f "scripts/pilot-chain8.sh" > /dev/null; do sleep 30; done
while pgrep -f "llama-quantize" > /dev/null; do sleep 30; done  # v2a must be built
step "amendment 3 swaps"; sh "$ROOT/scripts/decompose.sh"
step "v2a on S"; sh "$ROOT/scripts/v2a.sh"
step "chain9 done"
