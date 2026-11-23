#!/bin/sh
# Amendment 15 component B: task replies, Q8_0 first, then v1 (cand) and v2m; each server run under the GPU lock.
# SPECS="name:path ..." runs other models on the same items (amendment 16: SPECS=v2c:models/v2c.gguf).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
. scripts/gpu-lock.sh
for spec in ${SPECS:-q8:models/Inkling-2-Mini-Q8_0.gguf cand:models/cand.gguf v2m:models/v2m.gguf}; do
    n=${spec%%:*}; m=${spec#*:}
    mem_wait; gpu_take
    echo "[$(date '+%H:%M:%S')] battery B: $n"
    .venv/bin/python -I scripts/battery-B.py "$n" "$m"
    gpu_give
done
echo "[$(date '+%H:%M:%S')] battery B done"
