#!/bin/sh
# Amendment 22: v2c with UD-IQ3_XXS's NVFP4 downs in layers 0-19 (diagnostic). Scores S, T1, T2; deletes the file after.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits; PY=".venv/bin/python -I"; n=a22-nv
. scripts/gpu-lock.sh
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
if [ ! -f $M/$n.gguf ]; then
    cpu_take; mkdir -p $M/tmp
    $PY scripts/splice.py $M/v2c.gguf $M/tmp/$n.part.gguf \
        --take $M/UD-IQ3_XXS.gguf '^blk\.1?[0-9]\.ffn_down_exps\.weight$' \
        --add $M/UD-IQ3_XXS.gguf '^blk\.1?[0-9]\.ffn_down_exps\.scale$' --verify > $L/splice-$n.json
    cpu_give; mv $M/tmp/$n.part.gguf $M/$n.gguf
fi
for s in S T1 T2; do
    log=$L/ppl-final-$n-$s-P3.log; ch=""; [ $s = S ] && { log=$L/ppl-$n-S-P3.log; ch="--chunks 16"; }
    done_log "$log" && continue
    mem_wait; gpu_take
    $BIN/llama-perplexity -m $M/$n.gguf -f data/$s.txt -c 4096 $ch -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base $K/ref-$s-P3.kld --kl-divergence > "$log" 2>&1
    gpu_give; echo "$s: $(grep -E 'Mean +KLD' "$log")"
done
rm -f $M/$n.gguf
