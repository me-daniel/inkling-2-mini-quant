#!/bin/sh
# Addendum H amendment 20: joint scale + code refinement pilot, layers 6 and 20, arms A (codes only) and B (--scales),
# both from v2c's codes; then v2c with each arm's two tensors, KL on S. Resumes; waits for the BF16 rebuild
# (scripts/a19.sh phase 0) and for amendment 18's parts A-C.
set -euf
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits; PY=".venv/bin/python -I"
GD=$M/grams/gu-6-20-mixc; R=$M/refined
. scripts/gpu-lock.sh
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
while [ ! -f $M/Inkling-2-Mini-BF16.gguf ] || pgrep -f 'a18-(score|swaps)\.sh' > /dev/null; do sleep 60; done
[ -f results/a18/a18.md ] || { step "STOP: amendment 18 part A has not finished"; exit 1; }
LAYERS="6 20" CALIB=calib-mixc.txt CHUNKS=400 GSUF=-mixc sh scripts/gram-gu.sh > $L/gram-gu-6-20-mixc.out 2>&1
mkdir -p $R $M/tmp
for arm in A B; do
    out=$R/a20-$arm-6-20.gguf; [ -f $out ] && [ -f $out.json ] && continue
    flag=""; [ $arm = B ] && flag=--scales
    step "refine arm $arm (CPU lock, 4 workers)"; cpu_take
    $PY scripts/refine-codes.py --quant $M/v2c.gguf --bf16 $M/Inkling-2-Mini-BF16.gguf --grams $GD --layers 6 20 \
        --sweeps 4 --tol 1e-4 --shrink 0.1 --min-tokens 2304 --workers 4 $flag --out $out > $L/refine-a20-$arm.log 2>&1
    cpu_give
done
for arm in A B; do
    log=$L/ppl-a20-$arm-S-P3.log; f=$M/a20-$arm.gguf
    done_log $log && { rm -f $f; continue; }
    [ -f $f ] || { step "splice $arm"; cpu_take
        $PY scripts/splice.py $M/v2c.gguf $M/tmp/a20-$arm.part.gguf --take $R/a20-$arm-6-20.gguf '^blk\.(6|20)\.ffn_gate_up_exps\.weight$' --verify > $L/splice-a20-$arm.json
        cpu_give; mv $M/tmp/a20-$arm.part.gguf $f; }
    step "KL a20-$arm on S"; mem_wait; gpu_take
    $BIN/llama-perplexity -m $f -f data/S.txt -c 4096 --chunks 16 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base $K/ref-S-P3.kld --kl-divergence > $log 2>&1
    gpu_give; grep -E 'Mean +KLD' $log; rm -f $f
done
$PY analysis/a20.py | tee results/a18/a20-pilot.md
rm -rf $GD
step "a20 done"
