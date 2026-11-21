#!/bin/sh
# Downloads for Addendum H6, one at a time after the running Amendment 2 downloads.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
while pgrep -f "scripts/fetch-ext.sh" > /dev/null; do sleep 30; done
sh "$ROOT/scripts/fetch-one.sh" bart-Q2_K bartowski/thinkingmachines_Inkling-2-Mini-GGUF thinkingmachines_Inkling-2-Mini-Q2_K.gguf
