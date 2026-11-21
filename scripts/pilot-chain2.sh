#!/bin/sh
# After pilot-chain.sh: the Addendum D map levels.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
while pgrep -f "scripts/pilot-chain.sh" > /dev/null; do sleep 30; done
sh "$ROOT/scripts/map2.sh"
