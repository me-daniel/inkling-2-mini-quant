#!/bin/sh
# Replaces pilot-chain5 after the timing check: depth bench and memory first, then the new comparators when downloaded.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
while pgrep -f "scripts/p4-timing.sh" > /dev/null; do sleep 20; done
sh "$ROOT/scripts/depth-bench.sh"
while pgrep -f "scripts/fetch-ext.sh" > /dev/null; do sleep 60; done
sh "$ROOT/scripts/macgate.sh"
