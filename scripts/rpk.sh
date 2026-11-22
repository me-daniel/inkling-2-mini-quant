#!/bin/sh
# Addendum H amendment 9e: refined 9-11 gate_up rescaled per row to v2a's plain gain, spliced on the gu-v2a base, S.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"; M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; T="$M/tmp"; D="$M/refined"
PY="$ROOT/.venv/bin/python -I"
. "$ROOT/scripts/gpu-lock.sh"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
log="$L/ppl-rp-refined-k-9-10-11-S-P3.log"
if ! { [ -f "$log" ] && grep -q 'Mean *KLD' "$log"; }; then
    args=""
    for l in 9 10 11; do
        f="$D/rk-$l.gguf"
        [ -f "$f" ] || { mem_wait; nice -n 10 $PY "$ROOT/scripts/degain2.py" "$D/gu-refined-9-10-11.gguf" "$f" $l --kind ffn_gate_up_exps --target-from "$M/v2a.gguf" | tee -a "$L/rpk-degain.log"; }
        args="$args --take $f ^blk\\.$l\\.ffn_gate_up_exps\\.weight\$"
    done
    step "splice rp-refined-k-9-10-11"
    cpu_take
    # shellcheck disable=SC2086
    $PY "$ROOT/scripts/splice.py" "$M/cand.gguf" "$T/rpk.gguf" $args --take "$M/v2a.gguf" '^blk\.\d+\.ffn_gate_up_exps\.weight$' --verify > "$L/splice-rp-refined-k-9-10-11.json"
    $PY "$ROOT/scripts/gguf-types.py" "$T/rpk.gguf" > "$L/types-rp-refined-k-9-10-11.txt"
    cpu_give
    step "perplexity rp-refined-k-9-10-11 (P=3)"
    gpu_take
    "$BIN/llama-perplexity" -m "$T/rpk.gguf" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$log" 2>&1
    gpu_give
    rm -f "$T/rpk.gguf"
fi
grep -E 'Mean +KLD' "$log"
step "rpk done"
