#!/bin/sh
# Amendment 18, parts B and C (registered order, after part A: scripts/a18-score.sh).
# B: speed for P1 and P2, back to back in both orders (scripts/bench-v2m.sh: quiet machine, both locks), T = 6..9.
# C: reciprocal same-type tensor swaps between v2c and ISTA IQ3_S, scored on S and T1 only:
#    gate_up in the 6 layers where both files hold the same type (6, 11, 13, 17, 20, 22), and the down projections in
#    the 21 layers where both are IQ4_NL (2, 4-23); fixed bytes, fixed type, so each swap compares the two encodings.
#    Added after the registration, before its data (labelled in the report): v2c with ISTA's BF16 token_embd
#    (+0.67 GB), which asks what ISTA's embedding spend is worth.
# Each swap file is deleted after scoring. Resumes.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits; PY=".venv/bin/python -I"
. scripts/gpu-lock.sh
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
[ -f results/a18/a18.md ] || { step "STOP: part A has not finished"; exit 1; }
for spec in "6 v2c ista-IQ3_S" "7 ista-IQ3_S v2c" "8 c1 ista-IQ2_S" "9 ista-IQ2_S c1"; do
    set -- $spec
    [ -s results/pilot/depth$1-$3.txt ] && continue
    step "bench T=$1: $2 then $3"; NAMES="$2 $3" T=$1 sh scripts/bench-v2m.sh > $L/a18-bench$1.out 2>&1
done
score() { # name set
    log=$L/ppl-final-$1-$2-P3.log; [ "$2" = S ] && log=$L/ppl-$1-S-P3.log
    done_log "$log" && return 0
    c=4096; ch="--chunks 16"; [ "$2" = T1 ] && ch=""
    step "KL $1 on $2"; mem_wait; gpu_take
    $BIN/llama-perplexity -m $M/$1.gguf -f data/$2.txt -c $c $ch -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base $K/ref-$2-P3.kld --kl-divergence > "$log" 2>&1
    gpu_give; grep -E 'Mean +KLD|Same top p' "$log"
}
swap() { # name base donor regex
    n=$1
    if done_log $L/ppl-$n-S-P3.log && done_log $L/ppl-final-$n-T1-P3.log; then rm -f $M/$n.gguf; return 0; fi
    if [ ! -f $M/$n.gguf ]; then
        step "splice $n (CPU lock)"; cpu_take
        $PY scripts/splice.py $M/$2.gguf $M/tmp/$n.part.gguf --take $M/$3.gguf "$4" --verify > $L/splice-$n.json
        cpu_give; mv $M/tmp/$n.part.gguf $M/$n.gguf
    fi
    score $n S; score $n T1; rm -f $M/$n.gguf
}
GU='^blk\.(6|11|13|17|20|22)\.ffn_gate_up_exps\.weight$'
DN='^blk\.(2|[4-9]|1[0-9]|2[0-3])\.ffn_down_exps\.weight$'
swap v2c-gu6i v2c ista-IQ3_S "$GU"
swap ista-gu6v ista-IQ3_S v2c "$GU"
swap v2c-dni v2c ista-IQ3_S "$DN"
swap ista-dnv ista-IQ3_S v2c "$DN"
swap v2c-embi v2c ista-IQ3_S '^token_embd\.weight$'
$PY analysis/maptable.py > /dev/null
step "swaps done"
