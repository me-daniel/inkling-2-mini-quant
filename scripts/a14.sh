#!/bin/sh
# Amendment 14 (CHANGELOG): mirror on T1/T2, v2m-ud04 on S/T1/T2. Diagnostics; each GGUF is deleted after scoring. Resumes.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits; PY=".venv/bin/python -I"
. scripts/gpu-lock.sh
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
score() { # name set
    log=$L/ppl-final-$1-$2-P3.log; [ "$2" = S ] && log=$L/ppl-$1-S-P3.log
    done_log "$log" && return 0
    c=4096; ch="--chunks 16"; [ "$2" = T2 ] && { c=2048; ch=""; }; [ "$2" = T1 ] && ch=""
    step "KL $1 on $2"; mem_wait; gpu_take
    $BIN/llama-perplexity -m $M/$1.gguf -f data/$2.txt -c $c $ch -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base $K/ref-$2-P3.kld --kl-divergence > "$log" 2>&1
    gpu_give; grep -E 'Mean +KLD|Same top p' "$log"
}
build() { # name splice-args...
    n=$1; shift; [ -f $M/$n.gguf ] && return 0
    step "splice $n (CPU lock)"; cpu_take
    $PY scripts/splice.py "$@" $M/tmp/$n.part.gguf --verify > $L/splice-$n.json
    $PY scripts/gguf-types.py $M/tmp/$n.part.gguf > $L/types-$n.txt
    cpu_give
    grep -q "tensors in types Metal cannot multiply: 0 " $L/types-$n.txt || { step "STOP: $n has off-Metal tensors"; exit 1; }
    mv $M/tmp/$n.part.gguf $M/$n.gguf
}
DOWN='^blk\.([0-9]|1[0-9])\.ffn_down_exps\.'
G04='^blk\.(0|4)\.ffn_gate_up_exps\.'
if ! { done_log $L/ppl-final-mirror-T1-P3.log && done_log $L/ppl-final-mirror-T2-P3.log; }; then
    build mirror $M/UD-IQ3_XXS.gguf --take $M/pool-3bit.gguf "${DOWN}weight\$" --drop "${DOWN}scale\$"
    score mirror T1; score mirror T2
fi
rm -f $M/mirror.gguf
if ! { done_log $L/ppl-v2m-ud04-S-P3.log && done_log $L/ppl-final-v2m-ud04-T1-P3.log && done_log $L/ppl-final-v2m-ud04-T2-P3.log; }; then
    build v2m-ud04 $M/v2m.gguf --take $M/UD-IQ3_XXS.gguf "${G04}weight\$" --add $M/UD-IQ3_XXS.gguf "${G04}scale\$"
    score v2m-ud04 S; score v2m-ud04 T1; score v2m-ud04 T2
fi
rm -f $M/v2m-ud04.gguf
$PY analysis/maptable.py > /dev/null
$PY analysis/a14_report.py | tee results/pilot/a14.md
step "a14 done"
