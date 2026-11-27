#!/bin/sh
# Addendum H amendment 19, step 2: build the rule's subsets (2 per byte rule, from analysis/a19.py), score each on S,
# delete each after scoring. T1 only for a subset that beats v2c on S with the CI upper bound below 0 (reported by
# analysis/a19.py). Resumes.
set -euf
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits; PY=".venv/bin/python -I"
. scripts/gpu-lock.sh
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
for sub in $($PY analysis/a19.py picks); do
    log=$L/ppl-a19-$sub-S-P3.log; f=$M/a19-$sub.gguf
    done_log "$log" && { rm -f $f; continue; }
    if [ ! -f $f ]; then
        args=""
        for n in $(echo $sub | tr '+' ' '); do set -- $($PY analysis/a19.py spec $n); args="$args --take $1 $2"; done
        step "splice $sub (CPU lock)"; cpu_take
        $PY scripts/splice.py $M/v2c.gguf $M/tmp/a19-$sub.part.gguf $args --verify > $L/splice-a19-$sub.json
        cpu_give; mv $M/tmp/a19-$sub.part.gguf $f
    fi
    step "KL $sub on S"; mem_wait; gpu_take
    $BIN/llama-perplexity -m $f -f data/S.txt -c 4096 --chunks 16 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base $K/ref-S-P3.kld --kl-divergence > "$log" 2>&1
    gpu_give; grep -E 'Mean +KLD' "$log"; rm -f $f
done
$PY analysis/a19.py | tee results/a18/a19-singles.md
step "a19 subsets done"
