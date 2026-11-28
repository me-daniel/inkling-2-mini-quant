#!/bin/sh
# Amendment 23, part A: the base files with the release's own NVFP4 downs (repacked by scripts/nvfp4-repack.py,
# byte-identical to UD-IQ3_XXS's, logs/nvfp4-repack.log). v2c-A: all 20 downs of layers 0-19, one S smoke run.
# c1-A: the 8 layers of 0-19 where c1 holds IQ4_NL; S, T1, T2. Files are kept: they are the bases for what follows.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits; PY=".venv/bin/python -I"; R=$M/tmp/nvfp4-downs-0-19.gguf
. scripts/gpu-lock.sh
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
build() { # name base layer-regex
    [ -f $M/$1.gguf ] && return 0
    cpu_take
    $PY scripts/splice.py $M/$2.gguf $M/tmp/$1.part.gguf --take $R "^blk\.($3)\.ffn_down_exps\.weight\$" \
        --add $R "^blk\.($3)\.ffn_down_exps\.scale\$" --verify > $L/splice-$1.json
    cpu_give; mv $M/tmp/$1.part.gguf $M/$1.gguf
}
score() { # name set
    log=$L/ppl-final-$1-$2-P3.log; ch=""; c=4096
    [ $2 = S ] && { log=$L/ppl-$1-S-P3.log; ch="--chunks 16"; }
    [ $2 = T2 ] && c=2048
    done_log "$log" && return 0
    mem_wait; gpu_take
    $BIN/llama-perplexity -m $M/$1.gguf -f data/$2.txt -c $c $ch -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base $K/ref-$2-P3.kld --kl-divergence > "$log" 2>&1
    gpu_give; echo "$1 $2: $(grep -E 'Mean +KLD' "$log")"
}
build v2c-A v2c '1?[0-9]'
build c1-A c1 '5|8|9|11|13|17|18|19'
score v2c-A S
for s in S T1 T2; do score c1-A $s; done
