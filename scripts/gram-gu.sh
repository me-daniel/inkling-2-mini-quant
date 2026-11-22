#!/bin/sh
# Addendum H amendment 7: full per-expert input Grams of gate_up in layers 6 (looped, IQ2_S at UD's map, .scale in UD)
# and 20 (IQ3_XXS, no .scale in UD, MXFP8 source), for scripts/gram-error.py.
# Uses the sibling repo's llama-imatrix (its patch 0010, GRAM_DIR/GRAM_MATCH), read-only, on our cand.gguf at P = 3
# over the first 100 chunks of data/calib-wt.txt (WikiText-2 train, disjoint from S/T1/T2/T3). About 5.4 GB of Grams.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
SIBBIN="$HOME/GitHub/inkling-2-mini-under10/vendor/llama.cpp/build/bin/llama-imatrix"
D="$ROOT/models/grams/gu-6-20"; L="$ROOT/logs"
. "$ROOT/scripts/gpu-lock.sh"
[ -f "$D/done" ] && { echo "already captured"; exit 0; }
mkdir -p "$D"
echo "[$(date '+%H:%M:%S')] capture gate_up Grams, layers 6 and 20"
mem_wait; gpu_take
GRAM_DIR="$D" GRAM_MATCH='^blk\.(6|20)\.ffn_gate_up_exps\.weight$' "$SIBBIN" -m "$ROOT/models/cand.gguf" \
    -f "$ROOT/data/calib-wt.txt" --chunks 100 -c 512 -ngl 99 \
    --override-kv "inkling2_loop.loop.count=int:3" -o "$D/imatrix.gguf" > "$L/gram-gu-6-20.log" 2>&1
gpu_give
touch "$D/done"
ls -la "$D"
echo "[$(date '+%H:%M:%S')] gram-gu done"
