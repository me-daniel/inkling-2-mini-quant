#!/bin/sh
# v2r speed against v1 (cand), same binary and the same quiet machine: both locks held, so the v2m chain waits.
# cand is re-benched here so the pair shares conditions; the gate bench (results/pilot/bench-cand.txt) stays as it was.
# Same settings as the gate bench (pp512, tg128, 3 reps) and the depth bench (depth 0 / 4K / 16K, 2 reps).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; R=results/pilot; L=logs
. scripts/gpu-lock.sh
gpu_take; cpu_take  # GPU first: the chain then pauses before its next capture and frees the CPU after its check
echo "[$(date '+%H:%M:%S')] locks held; memory: $(memory_pressure | grep -o 'free percentage: [0-9]*%')"
for name in cand v2r; do
    echo "[$(date '+%H:%M:%S')] bench $name"
    $BIN/llama-bench -m models/$name.gguf -ngl 99 -p 512 -n 128 -r 3 -o md > $R/bench2-$name.txt 2> $L/bench2-$name.log
    cat $R/bench2-$name.txt
done
for name in cand v2r; do
    echo "[$(date '+%H:%M:%S')] depth bench $name"
    $BIN/llama-bench -m models/$name.gguf -ngl 99 -p 512 -n 128 -d 0,4096,16384 -r 2 -o md > $R/depth2-$name.txt 2> $L/depth2-$name.log
    cat $R/depth2-$name.txt
done
echo "[$(date '+%H:%M:%S')] bench done"
