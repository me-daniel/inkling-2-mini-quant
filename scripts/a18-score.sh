#!/bin/sh
# Amendment 18, part A (registered order): step 0 packaging check, then KL on fresh set A3 for all 9 files
# (per domain: Q8_0 reference logits, every file, delete the reference), then S/T1/T2 for the files not yet scored
# there. P = 3, -c 1024, 48 chunks on A3 (second half of each chunk scored). Resumes; logs/ppl-a3-<domain>-<name>-P3.log.
# Then the report: analysis/a18.py -> results/a18/a18.md.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits; PY=".venv/bin/python -I"
. scripts/gpu-lock.sh
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
NAMES="cand v2m v2c c1 ista-IQ3_S ista-IQ3_XXS ista-Q2_K_XL ista-IQ2_S UD-IQ3_XXS"
# read-only links to files that live elsewhere
ln -sf ../../inkling-2-mini-under10/models/release/c1.gguf $M/c1.gguf
for t in IQ3_S IQ3_XXS Q2_K_XL IQ2_S; do ln -sf ista/Inkling-2-Mini-GSQ-RCO-$t.gguf $M/ista-$t.gguf; done
ln -sf unsloth/Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file $M/UD-IQ3_XXS.gguf
for n in $NAMES; do [ -s $M/$n.gguf ] || { step "STOP: $M/$n.gguf missing"; exit 1; }; done
mkdir -p results/a18
if [ ! -s results/a18/pack.txt ]; then
    step "packaging check"
    $PY scripts/a18-pack.py $M/Inkling-2-Mini-Q8_0.gguf $(for n in $NAMES; do printf '%s=%s ' $n $M/$n.gguf; done) > results/a18/pack.txt.part \
        || { cat results/a18/pack.txt.part; step "STOP: packaging differs"; exit 1; }
    for n in $NAMES; do echo "== $n"; $PY scripts/gguf-types.py $M/$n.gguf; done >> results/a18/pack.txt.part
    mv results/a18/pack.txt.part results/a18/pack.txt
fi
ppl() { # log model extra...
    log=$1; m=$2; shift 2
    mem_wait; gpu_take
    $BIN/llama-perplexity -m "$m" -f "$F" -c 1024 --chunks 48 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" "$@" > "$log" 2>&1
    gpu_give
}
for d in python javascript german english math; do
    F=data/battery/A3-$d.txt; base=$K/a3-$d-P3.kld
    all=1; for n in $NAMES; do done_log $L/ppl-a3-$d-$n-P3.log || all=0; done
    if [ $all = 1 ]; then rm -f "$base"; continue; fi
    if [ ! -s "$base" ]; then
        step "$d: Q8_0 reference logits"
        ppl $L/ppl-a3-$d-ref-P3.log $M/Inkling-2-Mini-Q8_0.gguf --kl-divergence-base "$base.part"
        mv "$base.part" "$base"
    fi
    for n in $NAMES; do
        log=$L/ppl-a3-$d-$n-P3.log; done_log "$log" && continue
        step "$d: $n"
        ppl "$log" $M/$n.gguf --kl-divergence-base "$base" --kl-divergence
        grep -q "offloaded 25/25 layers to GPU\|offloaded .*layers to GPU" "$log" || step "WARN: $n offload line missing"
        grep -E 'Mean +KLD|Same top p' "$log"
    done
    rm -f "$base"
done
# S, T1, T2 (seen for us; descriptive), settings as amendment 14's score()
for n in c1 ista-IQ3_S ista-IQ3_XXS ista-Q2_K_XL ista-IQ2_S; do
    for t in S T1 T2; do
        log=$L/ppl-final-$n-$t-P3.log; [ $t = S ] && log=$L/ppl-$n-S-P3.log
        done_log "$log" && continue
        c=4096; ch="--chunks 16"; [ $t = T2 ] && { c=2048; ch=""; }; [ $t = T1 ] && ch=""
        step "KL $n on $t"; mem_wait; gpu_take
        $BIN/llama-perplexity -m $M/$n.gguf -f data/$t.txt -c $c $ch -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
            --kl-divergence-base $K/ref-$t-P3.kld --kl-divergence > "$log" 2>&1
        gpu_give; grep -E 'Mean +KLD|Same top p' "$log"
    done
done
$PY analysis/maptable.py > /dev/null
$PY analysis/a18.py | tee results/a18/a18.md
step "A3 done"
