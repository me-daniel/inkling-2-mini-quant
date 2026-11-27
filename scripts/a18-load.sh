#!/bin/sh
# Amendment 18 part B's placement check: load-only runs at -lv 5 (the KL and bench logs print no buffer lines at default
# verbosity), as in scripts/nvfp4-parity.py. Each run loads, warms up and stops at "you need at least ... tokens".
# Writes logs/load-a18-<name>.log and prints each file's model buffers.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin
. scripts/gpu-lock.sh
head -c 2000 data/S.txt > models/tmp/a18-load.txt
for n in v2c ista-IQ3_S c1 ista-IQ2_S; do
    mem_wait; gpu_take
    $BIN/llama-perplexity -m models/$n.gguf -f models/tmp/a18-load.txt -c 4096 -ngl 99 \
        --override-kv inkling2_loop.loop.count=int:3 -lv 5 > logs/load-a18-$n.log 2>&1 || true
    gpu_give
    echo "== $n"; grep -E "model buffer size|offloaded" logs/load-a18-$n.log
done
rm -f models/tmp/a18-load.txt
