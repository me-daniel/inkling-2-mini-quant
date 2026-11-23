#!/bin/sh
# Amendment 15 component C: task-trace KL (texts from scripts/battery-C.py). Per task: the Q8_0 reference logits (saved, then deleted), then v1
# (cand), v2m and, descriptive, UD-IQ3_XXS, each at P = 3, -c 1024 (second half of each chunk scored), up to 8 chunks.
# One GPU run at a time through the shared lock. Resumes; logs/ppl-batC-<domain>-<model>-P3.log.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits
. scripts/gpu-lock.sh
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
ppl() { # log model extra...
    log=$1; m=$2; shift 2
    mem_wait; gpu_take
    $BIN/llama-perplexity -m "$m" -f "$F" -c 1024 --chunks 8 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" "$@" > "$log" 2>&1
    gpu_give
}
for d in B1 B2 B3 B4; do
    F=data/battery/C-$d.txt; base=$K/batC-$d-P3.kld
    if done_log $L/ppl-batC-$d-cand-P3.log && done_log $L/ppl-batC-$d-v2m-P3.log && done_log $L/ppl-batC-$d-UD-IQ3_XXS-P3.log; then
        rm -f "$base"; continue
    fi
    if [ ! -s "$base" ]; then
        step "$d: Q8_0 reference logits"
        ppl $L/ppl-batC-$d-ref-P3.log $M/Inkling-2-Mini-Q8_0.gguf --kl-divergence-base "$base.part"
        mv "$base.part" "$base"
    fi
    for n in cand v2m UD-IQ3_XXS; do
        log=$L/ppl-batC-$d-$n-P3.log; done_log "$log" && continue
        step "$d: $n"
        ppl "$log" $M/$n.gguf --kl-divergence-base "$base" --kl-divergence
        grep -E 'Mean +KLD|Same top p' "$log"
    done
    rm -f "$base"
done
.venv/bin/python -I analysis/maptable.py > /dev/null
.venv/bin/python -I analysis/battery_C.py | tee results/battery/C.md
step "battery C done"
