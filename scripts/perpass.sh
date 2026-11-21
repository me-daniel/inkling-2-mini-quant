#!/bin/sh
# Pilot run 4 with addenda A and B (CHANGELOG.md, 2026-11-21): block B2 (layers 6-11) at 2-bit on chosen passes,
# plus tied and independent noise, on the IQ4_NL base at P = 3, scored on S. Each condition's logits are saved
# (models/logits/pp-<cond>.kld); KL against the base and the Q8_0 reference is computed afterwards in Python.
# Slot of pass p over core layer l: 24 + (p - 2)*12 + (l - 6). Resumes per condition.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"
L="$ROOT/logs"
K="$M/logits"
T="$M/tmp"
PY="$ROOT/.venv/bin/python -I"
BASE="$M/Inkling-2-Mini-IQ4_NL.gguf"
TWO="$M/pool-2bit.gguf"
B2='^blk\.(6|7|8|9|10|11)\.ffn_(gate_up|down)_exps\.weight$'
mkdir -p "$T" "$K" "$ROOT/results/pilot"

step() { echo "[$(date '+%H:%M:%S')] $*"; }

slots() { # donor pass -> --slot args for layers 6-11
    for l in 6 7 8 9 10 11; do printf ' --slot %s %s %s' "$1" "$l" $((24 + ($2 - 2) * 12 + l - 6)); done
}

if [ ! -f "$M/noise-1.gguf" ]; then
    step "noise donors"
    $PY "$ROOT/scripts/noise.py" "$BASE" "$TWO" "$M/noise" --layers 6-11 --seeds 1 2 3 > "$L/noise.log" 2>&1
    cp "$M/noise.tsv" "$ROOT/results/pilot/noise-t.tsv"
fi

run() { # cond splice-args...
    cond=$1; shift
    kld="$K/pp-$cond.kld"
    [ -f "$kld" ] && return 0
    model="$BASE"
    if [ $# -gt 0 ]; then
        model="$T/pp-$cond.gguf"
        step "splice $cond"
        # shellcheck disable=SC2086
        $PY "$ROOT/scripts/splice.py" "$BASE" "$model" "$@" --verify > "$L/splice-pp-$cond.json"
    fi
    step "logits $cond"
    "$BIN/llama-perplexity" -m "$model" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$kld.part" > "$L/ppl-pp-$cond.log" 2>&1
    mv "$kld.part" "$kld"
    [ "$model" = "$BASE" ] || rm -f "$model"
}

# shellcheck disable=SC2046
{
run base
run noop        $(slots "$BASE" 2) $(slots "$BASE" 3)
run p1          --take "$TWO" "$B2" $(slots "$BASE" 2) $(slots "$BASE" 3)
run p2          $(slots "$TWO" 2)
run p3          $(slots "$TWO" 3)
run all         --take "$TWO" "$B2"
run allslots    --take "$TWO" "$B2" $(slots "$TWO" 2) $(slots "$TWO" 3)
run noise-tied  --take "$M/noise-1.gguf" "$B2"
run noise-indep --take "$M/noise-1.gguf" "$B2" $(slots "$M/noise-2.gguf" 2) $(slots "$M/noise-3.gguf" 3)
}

step "KL per chunk"
{
    printf 'cond\tagainst\tmean_kl\tper_chunk\n'
    for c in base noop p1 p2 p3 all allslots noise-tied noise-indep; do
        for ref in pp-base ref-S-P3; do
            out=$($PY "$ROOT/scripts/kldfile.py" "$K/$ref.kld" "$K/pp-$c.kld")
            printf '%s\t%s\t%s\t%s\n' "$c" "$ref" "$(echo "$out" | tail -1 | cut -d' ' -f3)" "$(echo "$out" | head -1 | tr ' ' ',')"
        done
    done
} > "$ROOT/results/pilot/perpass-kl.tsv"
cat "$ROOT/results/pilot/perpass-kl.tsv" | cut -f1-3
step "done"
