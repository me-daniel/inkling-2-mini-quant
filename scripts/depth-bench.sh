#!/bin/sh
# Addendum G extension (Daniel, 21 Nov evening): speed at context depth 0 / 4K / 16K and peak memory, same binary as
# the gate bench. Files: cand, UD-IQ3_XXS, AtomicChat AD-IQ2_S (lowest-KL all-Metal comparator). Resumes.
# Peak memory: macOS /usr/bin/time -l on one llama-bench run at depth 16K ("maximum resident set size" and
# "peak memory footprint"); the Metal buffer sizes llama.cpp logs are kept in the log.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"; R="$ROOT/results/pilot"; L="$ROOT/logs"
for spec in cand:models/cand.gguf UD-IQ3_XXS:models/UD-IQ3_XXS.gguf atomic-AD-IQ2_S:models/ext/atomic-AD-IQ2_S.gguf; do
    name=${spec%%:*}; f="$ROOT/${spec#*:}"
    if [ ! -f "$R/depth-$name.txt" ]; then
        echo "[$(date '+%H:%M:%S')] depth bench $name"
        "$BIN/llama-bench" -m "$f" -ngl 99 -p 512 -n 128 -d 0,4096,16384 -r 2 -o md > "$R/depth-$name.txt.part" 2> "$L/depth-$name.log"
        mv "$R/depth-$name.txt.part" "$R/depth-$name.txt"
    fi
    if [ ! -f "$R/memory-$name.txt" ]; then
        echo "[$(date '+%H:%M:%S')] peak memory $name"
        /usr/bin/time -l "$BIN/llama-bench" -m "$f" -ngl 99 -p 512 -n 128 -d 16384 -r 1 -o md > "$L/memory-$name.out" 2> "$L/memory-$name.log"
        grep -E "maximum resident set size|peak memory footprint" "$L/memory-$name.log" > "$R/memory-$name.txt"
    fi
done
echo "[$(date '+%H:%M:%S')] depth and memory done"
