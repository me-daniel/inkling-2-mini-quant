#!/bin/sh
# Amendment 23: score files on the development set DV (data/battery/DV-<domain>.txt), A3's protocol: P = 3, -c 1024,
# 48 chunks per domain, second half of each chunk scored. The Q8_0 references stay in models/logits/dv-<domain>-P3.kld
# while the v3 track runs (DV is scored many times). Resumes; logs/ppl-dv-<domain>-<name>-P3.log.
#   NAMES="v2c v2c-A" sh scripts/dv-score.sh      (models/<name>.gguf)
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits
. scripts/gpu-lock.sh
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
ppl() { # log model extra...
    log=$1; m=$2; shift 2
    mem_wait; gpu_take
    $BIN/llama-perplexity -m "$m" -f "$F" -c 1024 --chunks 48 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" "$@" > "$log" 2>&1
    gpu_give
}
for d in python javascript german english math; do
    F=data/battery/DV-$d.txt; base=$K/dv-$d-P3.kld
    if [ ! -s "$base" ]; then
        echo "$d: Q8_0 reference"
        ppl $L/ppl-dv-$d-ref-P3.log $M/Inkling-2-Mini-Q8_0.gguf --kl-divergence-base "$base.part"
        mv "$base.part" "$base"
    fi
    for n in $NAMES; do
        log=$L/ppl-dv-$d-$n-P3.log; done_log "$log" && continue
        ppl "$log" $M/$n.gguf --kl-divergence-base "$base" --kl-divergence
        echo "$d $n: $(grep -E 'Mean +KLD' "$log")"
    done
done
