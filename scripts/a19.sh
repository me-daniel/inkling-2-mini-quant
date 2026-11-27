#!/bin/sh
# Addendum H amendment 19: single cherry-picked changes on v2c (s1-s8), each scored on S at P = 3, then deleted.
# Phase 0 (CPU lock, may run while amendment 18 holds the GPU): the BF16 rebuild and the IQ3_S donor for s8.
# Phase 1 (GPU): only after amendment 18's parts A-C (a18-score.sh and a18-swaps.sh no longer running and
# results/a18/a18.md present). Resumes: every step skips what is done.
set -euf  # no globbing: the specs carry regexes
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits; PY=".venv/bin/python -I"
BF=$M/Inkling-2-Mini-BF16.gguf; D8=$M/tmp/a19-iq3s-donor.gguf
. scripts/gpu-lock.sh
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
mkdir -p $M/tmp
free_gb() { df -g "$M" | awk 'NR==2{print $4}'; }

# ---- phase 0 -------------------------------------------------------------------------------------------------
if [ ! -f $BF ]; then
    [ "$(free_gb)" -ge 90 ] || { step "STOP: under 90 GB free for the BF16 rebuild"; exit 1; }
    step "convert to BF16 (CPU lock)"; cpu_take
    .venv/bin/python vendor/llama.cpp/convert_hf_to_gguf.py $M/Inkling-2-Mini --outtype bf16 --outfile $BF.part > $L/convert-a19.log 2>&1
    cpu_give; mv $BF.part $BF
fi
if [ ! -f $D8 ]; then
    step "IQ3_S donor (CPU lock)"; cpu_take
    $BIN/llama-quantize --imatrix $M/ggml-org/imatrix.gguf --tensor-type 'blk\.(9|10|15)\.ffn_gate_up_exps=iq3_s' \
        $BF $D8.part Q8_0 > $L/quant-a19-iq3s-donor.log 2>&1
    cpu_give; mv $D8.part $D8
fi
$PY scripts/gguf-types.py $D8 | grep -E 'IQ3_S|fallback' || { step "STOP: donor holds no IQ3_S"; exit 1; }

# ---- phase 1 -------------------------------------------------------------------------------------------------
while pgrep -f 'a18-(score|swaps)\.sh' > /dev/null; do sleep 60; done
[ -f results/a18/a18.md ] || { step "STOP: amendment 18 part A has not finished"; exit 1; }
score() { # name
    log=$L/ppl-a19-$1-S-P3.log; done_log "$log" && return 0
    step "KL $1 on S"; mem_wait; gpu_take
    $BIN/llama-perplexity -m $M/a19-$1.gguf -f data/S.txt -c 4096 --chunks 16 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base $K/ref-S-P3.kld --kl-divergence > "$log" 2>&1
    gpu_give; grep -E 'Mean +KLD|Same top p' "$log"
}
change() { # name (donor and regex from analysis/a19.py)
    n=$1; set -- $($PY analysis/a19.py spec $n); f=$M/a19-$n.gguf
    if done_log $L/ppl-a19-$n-S-P3.log; then rm -f $f; return 0; fi
    if [ ! -f $f ]; then
        step "splice $n (CPU lock)"; cpu_take
        $PY scripts/splice.py $M/v2c.gguf $M/tmp/a19-$n.part.gguf --take "$1" "$2" --verify > $L/splice-a19-$n.json
        cpu_give; mv $M/tmp/a19-$n.part.gguf $f
    fi
    score $n; rm -f $f
}
change s1
change s2
change s3
change s4
change s5
change s6
change s7
change s8
$PY analysis/maptable.py > /dev/null
$PY analysis/a19.py | tee results/a18/a19-singles.md
step "a19 singles done"
