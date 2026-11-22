#!/bin/sh
# Addendum H amendment 9 pilot, S runs on the gu-v2a base (cand + v2a's gate_up; (a) = logs/ppl-gu-v2a-S-P3.log):
#   (c) rp-ud:      UD's gate_up (+ its .scale) in layers 6 and 20, v2a's elsewhere
#   (b) rp-refined: refined gate_up (models/refined/gu-refined-6-20.gguf) in layers 6 and 20, v2a's elsewhere
# First --take match wins, so the layer-specific donor comes first. Builds what it can; resumes; deletes each GGUF after scoring.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"; M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; T="$M/tmp"
PY="$ROOT/.venv/bin/python -I"
. "$ROOT/scripts/gpu-lock.sh"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
L620='^blk\.(6|20)\.ffn_gate_up_exps\.weight$'
ALL='^blk\.\d+\.ffn_gate_up_exps\.weight$'
one() { # name splice-args...
    name=$1; shift; log="$L/ppl-$name-S-P3.log"
    done_log "$log" && { grep -E 'Mean +KLD' "$log"; return 0; }
    step "splice $name"
    if [ "${DRY:-0}" = 1 ]; then echo "  splice.py cand OUT $*"; return 0; fi
    cpu_take
    $PY "$ROOT/scripts/splice.py" "$M/cand.gguf" "$T/$name.gguf" "$@" --verify > "$L/splice-$name.json"
    $PY "$ROOT/scripts/gguf-types.py" "$T/$name.gguf" > "$L/types-$name.txt"
    cpu_give
    step "perplexity $name (P=3)"
    gpu_take
    "$BIN/llama-perplexity" -m "$T/$name.gguf" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$log" 2>&1
    gpu_give
    rm -f "$T/$name.gguf"
    grep -E 'Mean +KLD' "$log"
}
mkdir -p "$T"
one rp-ud --take "$M/UD-IQ3_XXS.gguf" "$L620" --take "$M/v2a.gguf" "$ALL" --add "$M/UD-IQ3_XXS.gguf" '^blk\.6\.ffn_gate_up_exps\.scale$'
R="$M/refined/gu-refined-6-20.gguf"
if [ -f "$R" ]; then one rp-refined --take "$R" "$L620" --take "$M/v2a.gguf" "$ALL"; else step "rp-refined: $R not there yet"; fi
step "refine-pilot done"
