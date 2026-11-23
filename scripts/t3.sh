#!/bin/sh
# Addendum H amendment 1, T3, opened once (Daniel's go 23 Nov, frozen in CHANGELOG "T3 frozen" before this runs):
# Q8_0 reference logits on data/T3.txt, then v1 (cand), v2m and UD-IQ3_XXS, T1 protocol (P = 3, 4,096-token chunks).
# Each step skips when its log is complete, so a rerun after a crash scores nothing twice.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; K=models/logits; L=logs; PY=".venv/bin/python -I"
. scripts/gpu-lock.sh
OV="inkling2_loop.loop.count=int:3"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
[ "$(shasum -a 256 data/T3.txt | cut -d' ' -f1)" = 5a26eea1323fbade49482767b9ea4de14f539262b82135ab0e0e0a917be689a7 ] || { echo "T3 hash mismatch"; exit 1; }
if [ ! -f $K/ref-T3-P3.kld ]; then
    step "reference logits T3"
    gpu_take
    $BIN/llama-perplexity -m models/Inkling-2-Mini-Q8_0.gguf -f data/T3.txt -c 4096 -ngl 99 --override-kv "$OV" \
        --kl-divergence-base $K/ref-T3-P3.kld.part > $L/ppl-ref-T3-P3.log 2>&1
    mv $K/ref-T3-P3.kld.part $K/ref-T3-P3.kld
    gpu_give
fi
for N in cand v2m UD-IQ3_XXS; do
    log="$L/ppl-final-$N-T3-P3.log"
    [ -f "$log" ] && grep -q 'Mean *KLD' "$log" && continue
    step "KL $N on T3"
    gpu_take
    $BIN/llama-perplexity -m models/$N.gguf -f data/T3.txt -c 4096 -ngl 99 --override-kv "$OV" \
        --kl-divergence-base $K/ref-T3-P3.kld --kl-divergence > "$log" 2>&1
    gpu_give
    grep -E 'Mean +KLD|Same top p' "$log"
done
$PY analysis/maptable.py > /dev/null
$PY analysis/compare.py --sets=T3 v2m UD-IQ3_XXS cand | tee results/pilot/t3.txt
step "T3 done"
