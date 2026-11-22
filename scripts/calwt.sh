#!/bin/sh
# Addendum H amendment 5, cal-wt: does the calibration text explain UD's encoding edge?
# 1. WikiText-2-raw train (Salesforce/wikitext rev b08601e, disjoint from the test and validation splits) ->
#    data/calib-wt.txt, the first 300 x 512 tokens by a fixed rule, with a 13-gram overlap check against S, T1, T2, T3.
# 2. upstream llama-imatrix on the Q8_0 reference (P = 3, plain sum over visits like the ggml-org imatrix) -> imatrix-wt.
# 3. llama-quantize at UD's gate_up map with that imatrix (the v2a recipe) -> v2a-wt; splice only its gate_up into cand.
# 4. cal-wt on S. Read against gu-v2a (same map, ggml-org imatrix). Resumes; deletes v2a-wt when scored.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; T="$M/tmp"
PY="$ROOT/.venv/bin/python -I"
RAW="$ROOT/data/raw/wikitext"
. "$ROOT/scripts/gpu-lock.sh"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
mkdir -p "$T"

if [ ! -f "$RAW/wikitext-2-raw-v1/train-00000-of-00001.parquet" ]; then
    step "fetch WikiText-2 train"
    HF_HUB_DISABLE_XET=1 hf download Salesforce/wikitext wikitext-2-raw-v1/train-00000-of-00001.parquet \
        --repo-type dataset --revision b08601e04326c79dfdd32d625aee71d232d685c3 --local-dir "$RAW" > /dev/null
fi
[ -f "$ROOT/data/calib-wt.txt" ] || $PY "$ROOT/scripts/calib-wt.py"

if [ ! -f "$M/imatrix-wt.gguf" ]; then
    step "imatrix on WikiText-2 train"
    mem_wait; gpu_take
    "$BIN/llama-imatrix" -m "$M/Inkling-2-Mini-Q8_0.gguf" -f "$ROOT/data/calib-wt.txt" -c 512 --chunks 300 -ngl 99 \
        -o "$M/imatrix-wt.gguf.part.gguf" > "$L/imatrix-wt.log" 2>&1
    gpu_give
    mv "$M/imatrix-wt.gguf.part.gguf" "$M/imatrix-wt.gguf"
fi

log="$L/ppl-cal-wt-S-P3.log"
if ! done_log "$log"; then
    if [ ! -f "$M/v2a-wt.gguf" ]; then
        step "quantize v2a-wt"
        mem_wait
        nice -n 5 "$BIN/llama-quantize" --imatrix "$M/imatrix-wt.gguf" \
            --tensor-type 'blk\.0\.ffn_gate_up_exps=iq2_xxs' \
            --tensor-type 'blk\.4\.ffn_gate_up_exps=iq2_xs' \
            --tensor-type 'blk\.(1|2|3|5|6|7|8|12|18)\.ffn_gate_up_exps=iq2_s' \
            --tensor-type 'blk\.(9|10|11|13|14|15|16|17|19|20|21|22|23)\.ffn_gate_up_exps=iq3_xxs' \
            --tensor-type 'ffn_down_exps=iq4_nl' \
            --token-embedding-type q4_k --output-tensor-type q5_k \
            "$M/Inkling-2-Mini-BF16.gguf" "$M/v2a-wt.gguf.part" Q6_K > "$L/quant-v2a-wt.log" 2>&1
        mv "$M/v2a-wt.gguf.part" "$M/v2a-wt.gguf"
    fi
    step "splice cal-wt"
    mem_wait
    $PY "$ROOT/scripts/splice.py" "$M/cand.gguf" "$T/cal-wt.gguf" --take "$M/v2a-wt.gguf" '^blk\.\d+\.ffn_gate_up_exps\.weight$' \
        --verify > "$L/splice-cal-wt.json"
    step "perplexity cal-wt (P=3)"
    gpu_take
    "$BIN/llama-perplexity" -m "$T/cal-wt.gguf" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$log" 2>&1
    gpu_give
    rm -f "$T/cal-wt.gguf"
fi
grep -E 'Mean +KLD|Same top p' "$log"
done_log "$log" && rm -f "$M/v2a-wt.gguf"
step "cal-wt done"
