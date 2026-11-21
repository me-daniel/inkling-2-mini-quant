#!/bin/sh
# Evening chain: wait for the KL-table check (CPU/IO), P = 2/3/4 timing, then the two new comparators (macgate resumes).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
while pgrep -f "scripts/perpass-kl.sh" > /dev/null; do sleep 20; done
sh "$ROOT/scripts/p4-timing.sh"
while pgrep -f "scripts/fetch-ext.sh" > /dev/null; do sleep 60; done
sh "$ROOT/scripts/macgate.sh"
