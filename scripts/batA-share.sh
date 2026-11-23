#!/bin/sh
# Q8_0 reference logits of the battery A texts for the sibling repo's off-domain scores (agreed 23 Nov, ~12:00).
# After our battery B and C: one domain at a time, written as .part and renamed, kept until the sibling's
# results/batA/done-<domain> exists, then deleted. Same settings as scripts/battery-A.sh.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; K=models/logits; SIB="$HOME/GitHub/inkling-2-mini-under10/results/batA"
. scripts/gpu-lock.sh
until grep -q "battery C done" logs/battery-C.out 2>/dev/null; do sleep 60; done
for d in python rust german math; do
    base=$K/batA-$d-P3.kld
    if [ ! -f "$SIB/done-$d" ]; then
        if [ ! -s "$base" ]; then
            echo "[$(date '+%H:%M:%S')] $d: Q8_0 reference logits"
            mem_wait; gpu_take
            $BIN/llama-perplexity -m models/Inkling-2-Mini-Q8_0.gguf -f data/battery/A-$d.txt -c 1024 --chunks 32 -ngl 99 \
                --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$base.part" > logs/ppl-batA-share-$d-ref.log 2>&1
            gpu_give
            mv "$base.part" "$base"
        fi
        echo "[$(date '+%H:%M:%S')] $d: waiting for the sibling's done-$d"
        until [ -f "$SIB/done-$d" ]; do sleep 60; done
    fi
    rm -f "$base"; echo "[$(date '+%H:%M:%S')] $d: deleted $base"
done
echo "[$(date '+%H:%M:%S')] batA share done"
