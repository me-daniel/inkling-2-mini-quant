#!/bin/sh
# Addendum H amendment 16: fresh set A2 (scripts/a16-data.py). Per domain: the Q8_0 reference logits, then v1 (cand),
# v2m, v2c and, descriptive, UD-IQ3_XXS, each at P = 3, -c 1024 (second half of each chunk scored), 32 chunks; the
# reference is deleted once all four are scored. Runs after v2c exists (scripts/v2c.sh). Resumes;
# logs/ppl-a2-<domain>-<model>-P3.log. Then the report: analysis/a16.py -> results/battery/a16.md.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits
. scripts/gpu-lock.sh
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
[ -f $M/v2c.gguf ] || { step "STOP: models/v2c.gguf missing"; exit 1; }
ppl() { # log model extra...
    log=$1; m=$2; shift 2
    mem_wait; gpu_take
    $BIN/llama-perplexity -m "$m" -f "$F" -c 1024 --chunks 32 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" "$@" > "$log" 2>&1
    gpu_give
}
MODELS="cand v2m v2c UD-IQ3_XXS"
for d in python rust go typescript german; do
    F=data/battery/A2-$d.txt; [ $d = german ] && F=data/battery/A-german.txt  # battery A's German, reused as a guard
    base=$K/a2-$d-P3.kld
    all=1; for n in $MODELS; do done_log $L/ppl-a2-$d-$n-P3.log || all=0; done
    if [ $all = 1 ]; then rm -f "$base"; continue; fi
    if [ ! -s "$base" ]; then
        step "$d: Q8_0 reference logits"
        ppl $L/ppl-a2-$d-ref-P3.log $M/Inkling-2-Mini-Q8_0.gguf --kl-divergence-base "$base.part"
        mv "$base.part" "$base"
    fi
    for n in $MODELS; do
        log=$L/ppl-a2-$d-$n-P3.log; done_log "$log" && continue
        step "$d: $n"
        ppl "$log" $M/$n.gguf --kl-divergence-base "$base" --kl-divergence
        grep -E 'Mean +KLD|Same top p' "$log"
    done
    rm -f "$base"
done
.venv/bin/python -I analysis/maptable.py > /dev/null
.venv/bin/python -I analysis/a16.py | tee results/battery/a16.md
step "A2 done"
