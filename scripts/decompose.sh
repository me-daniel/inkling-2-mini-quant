#!/bin/sh
# Addendum H amendment 2: cand with UD-IQ3_XXS's non-expert tensors (ne-ud) or gate_up experts (gu-ud), S, P = 3.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; T="$M/tmp"
PY="$ROOT/.venv/bin/python -I"
UD="$M/UD-IQ3_XXS.gguf"
mkdir -p "$T"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
one() { # name splice-args...
    name=$1; shift
    log="$L/ppl-$name-S-P3.log"; done_log "$log" && return 0
    step "splice $name"
    $PY "$ROOT/scripts/splice.py" "$M/cand.gguf" "$T/$name.gguf" "$@" --verify > "$L/splice-$name.json"
    $PY "$ROOT/scripts/gguf-types.py" "$T/$name.gguf" > "$L/types-$name.txt"
    grep -q "tensors in types Metal cannot multiply: 0 " "$L/types-$name.txt" || { echo "$name has off-Metal tensors" >&2; exit 1; }
    step "perplexity $name (P=3)"
    "$BIN/llama-perplexity" -m "$T/$name.gguf" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$log" 2>&1
    grep -E 'Mean +KLD|Same top p' "$log"
    rm -f "$T/$name.gguf"
}
one ne-ud --take "$UD" '^(?!.*_exps\.)'
one gu-ud --take "$UD" '^blk\.\d+\.ffn_gate_up_exps\.weight$' --add "$UD" '^blk\.\d+\.ffn_gate_up_exps\.scale$'
# amendment 3: isolating swaps
one gu15-ud --take "$UD" '^blk\.[1-5]\.ffn_gate_up_exps\.weight$' --add "$UD" '^blk\.[1-5]\.ffn_gate_up_exps\.scale$'
one gu911-ud --take "$UD" '^blk\.(9|10|11)\.ffn_gate_up_exps\.weight$' --add "$UD" '^blk\.(9|10|11)\.ffn_gate_up_exps\.scale$'
step "decompose done"
