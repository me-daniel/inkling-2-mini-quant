#!/bin/sh
# v2m speed and peak memory against v1 (cand), back to back on a quiet machine (both reviews of 23 Nov asked for it).
# Takes the GPU and CPU locks only when both are free at once and gives the GPU back if the CPU is taken in between,
# so it never holds an idle GPU (the sibling's bench2.sh pattern). Same settings as bench-v2r.sh, plus peak memory.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; R=results/pilot; L=logs
. scripts/gpu-lock.sh
while :; do
    if ! pgrep -x llama-bench > /dev/null && ! pgrep -x llama-perplexity > /dev/null && gpu_try; then
        cpu_try && break
        gpu_give
    fi
    sleep 30
done
mem_wait
echo "[$(date '+%H:%M:%S')] both locks held; memory: $(memory_pressure | grep -o 'free percentage: [0-9]*%')"
git -C vendor/llama.cpp rev-parse --short HEAD > $R/bench3-commit.txt
for name in cand v2m; do
    echo "[$(date '+%H:%M:%S')] bench $name"
    $BIN/llama-bench -m models/$name.gguf -ngl 99 -p 512 -n 128 -r 3 -o md > $R/bench3-$name.txt 2> $L/bench3-$name.log
    cat $R/bench3-$name.txt
done
for name in cand v2m; do
    echo "[$(date '+%H:%M:%S')] depth bench and peak memory $name"
    /usr/bin/time -l $BIN/llama-bench -m models/$name.gguf -ngl 99 -p 512 -n 128 -d 0,4096,16384 -r 2 -o md > $R/depth3-$name.txt 2> $L/depth3-$name.log
    cat $R/depth3-$name.txt; grep -E "maximum resident set size|peak memory footprint" $L/depth3-$name.log
done
echo "[$(date '+%H:%M:%S')] bench done"
