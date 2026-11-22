#!/bin/sh
# Amendment 12 fallback reading: v2r (N=v2r) or another models/N.gguf on T1 and T2 (both already seen; reported as seen). T3 stays closed.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; K=models/logits; L=logs; PY=".venv/bin/python -I"
. scripts/gpu-lock.sh
N=${N:-v2r}
for t in T1 T2; do
    log="$L/ppl-final-$N-$t-P3.log"
    [ -f "$log" ] && grep -q 'Mean *KLD' "$log" && continue
    c=4096; [ "$t" = T2 ] && c=2048
    echo "[$(date '+%H:%M:%S')] KL $N on $t"
    gpu_take
    $BIN/llama-perplexity -m models/$N.gguf -f data/$t.txt -c $c -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base $K/ref-$t-P3.kld --kl-divergence > "$log" 2>&1
    gpu_give
    grep -E 'Mean +KLD|Same top p' "$log"
done
$PY analysis/maptable.py > /dev/null
$PY analysis/compare.py $N UD-IQ3_XXS cand
