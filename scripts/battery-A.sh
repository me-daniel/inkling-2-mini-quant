#!/bin/sh
# Amendment 15 component A: fresh-domain KL. Per domain: the Q8_0 reference logits (saved, then deleted), then v1
# (cand), v2m and, descriptive, UD-IQ3_XXS, each at P = 3, -c 1024 (second half of each chunk scored), 32 chunks.
# One GPU run at a time through the shared lock. Resumes; logs/ppl-batA-<domain>-<model>-P3.log.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits
. scripts/gpu-lock.sh
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
ppl() { # log model extra...
    log=$1; m=$2; shift 2
    mem_wait; gpu_take
    $BIN/llama-perplexity -m "$m" -f "$F" -c 1024 --chunks 32 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" "$@" > "$log" 2>&1
    gpu_give
}
for d in python rust german math; do
    F=data/battery/A-$d.txt; base=$K/batA-$d-P3.kld
    if done_log $L/ppl-batA-$d-cand-P3.log && done_log $L/ppl-batA-$d-v2m-P3.log && done_log $L/ppl-batA-$d-UD-IQ3_XXS-P3.log; then
        rm -f "$base"; continue
    fi
    if [ ! -s "$base" ]; then
        step "$d: Q8_0 reference logits"
        ppl $L/ppl-batA-$d-ref-P3.log $M/Inkling-2-Mini-Q8_0.gguf --kl-divergence-base "$base.part"
        mv "$base.part" "$base"
    fi
    for n in cand v2m UD-IQ3_XXS; do
        log=$L/ppl-batA-$d-$n-P3.log; done_log "$log" && continue
        step "$d: $n"
        ppl "$log" $M/$n.gguf --kl-divergence-base "$base" --kl-divergence
        grep -E 'Mean +KLD|Same top p' "$log"
    done
    rm -f "$base"
done
.venv/bin/python -I analysis/maptable.py > /dev/null
.venv/bin/python -I analysis/battery_A.py | tee results/battery/A.md
step "battery A done"
