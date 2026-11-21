#!/bin/sh
# Addendum F (CHANGELOG.md): controls for the per-pass test, on the same IQ4_NL base, S, P = 3.
# 1. Layer split (non-shared comparator): the 2-bit block on layers 6-7 / 8-9 / 10-11 only, on all passes. The three
#    sum to the existing "all" condition, so the same residual analysis gives r for a split that involves no sharing.
# 2. Small-amplitude rung: pass 1 / 2 / 3 only and all passes at the 3-bit level (IQ3_XXS gate_up; down IQ4_NL as base).
# Each group is analysed as soon as its files exist, then its logits are deleted (disk). Resumes.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; T="$M/tmp"; R="$ROOT/results/pilot"
PY="$ROOT/.venv/bin/python -I"
BASE="$M/Inkling-2-Mini-IQ4_NL.gguf"
TWO="$M/pool-2bit.gguf"
THREE="$M/pool-3bit.gguf"
B2='^blk\.(6|7|8|9|10|11)\.ffn_(gate_up|down)_exps\.weight$'
mkdir -p "$T" "$K" "$R"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
slots() { for l in 6 7 8 9 10 11; do printf ' --slot %s %s %s' "$1" "$l" $((24 + ($2 - 2) * 12 + l - 6)); done; }
run() { # cond splice-args...
    cond=$1; shift
    kld="$K/pp-$cond.kld"
    [ -f "$kld" ] && return 0
    model="$T/pp-$cond.gguf"
    step "splice $cond"
    # shellcheck disable=SC2086
    $PY "$ROOT/scripts/splice.py" "$BASE" "$model" "$@" --verify > "$L/splice-pp-$cond.json"
    step "logits $cond"
    "$BIN/llama-perplexity" -m "$model" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$kld.part" > "$L/ppl-pp-$cond.log" 2>&1
    mv "$kld.part" "$kld"
    rm -f "$model"
}
analyse() { # out.json base s1 s2 s3 all
    out=$1; shift
    [ -f "$out" ] && return 0
    step "residual $(basename "$out")"
    $PY "$ROOT/analysis/per_pass.py" "$K/pp-$1.kld" "$K/pp-$2.kld" "$K/pp-$3.kld" "$K/pp-$4.kld" "$K/pp-$5.kld" > "$out.part"
    mv "$out.part" "$out"
}
kl_row() { # cond -> "cond <tab> KL against pp-base"
    printf '%s\t%s\n' "$1" "$($PY "$ROOT/scripts/kldfile.py" "$K/pp-base.kld" "$K/pp-$1.kld" | tail -1 | cut -d' ' -f3)"
}

# shellcheck disable=SC2046
{
run ls67    --take "$TWO" '^blk\.(6|7)\.ffn_(gate_up|down)_exps\.weight$'
run ls89    --take "$TWO" '^blk\.(8|9)\.ffn_(gate_up|down)_exps\.weight$'
run ls1011  --take "$TWO" '^blk\.(10|11)\.ffn_(gate_up|down)_exps\.weight$'
}
analyse "$R/layersplit-residual.json" base ls67 ls89 ls1011 all
[ -f "$R/perpass2-kl.tsv" ] || printf 'cond\tkl_vs_base\n' > "$R/perpass2-kl.tsv"
for c in ls67 ls89 ls1011; do grep -q "^$c	" "$R/perpass2-kl.tsv" || kl_row "$c" >> "$R/perpass2-kl.tsv"; done
rm -f "$K/pp-ls67.kld" "$K/pp-ls89.kld" "$K/pp-ls1011.kld"

# shellcheck disable=SC2046
{
run 3b-p1   --take "$THREE" "$B2" $(slots "$BASE" 2) $(slots "$BASE" 3)
run 3b-p2   $(slots "$THREE" 2)
run 3b-p3   $(slots "$THREE" 3)
run 3b-all  --take "$THREE" "$B2"
}
analyse "$R/perpass3bit-residual.json" base 3b-p1 3b-p2 3b-p3 3b-all
for c in 3b-p1 3b-p2 3b-p3 3b-all; do grep -q "^$c	" "$R/perpass2-kl.tsv" || kl_row "$c" >> "$R/perpass2-kl.tsv"; done
rm -f "$K/pp-3b-p1.kld" "$K/pp-3b-p2.kld" "$K/pp-3b-p3.kld" "$K/pp-3b-all.kld"
step "done"
