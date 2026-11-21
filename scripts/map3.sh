#!/bin/sh
# Addendum E (CHANGELOG.md): per-block levels with the imatrix-weighted Q2_0 encoder, S, P = 3, Q8_0 elsewhere.
# dnq2o: gate_up IQ3_XXS (3-bit pool), down optimized Q2_0. (q2o, Q2_0 on gate_up too, was dropped: dominated by IQ2_XXS.)
# First, Unsloth's UD-IQ3_XXS on S: the number to beat, and a real-file check of the .scale handling.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; T="$M/tmp"
PY="$ROOT/.venv/bin/python -I"
REF="$M/Inkling-2-Mini-Q8_0.gguf"
mkdir -p "$T"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }

if ! done_log "$L/ppl-UD-IQ3_XXS-S-P3.log"; then
    step "perplexity UD-IQ3_XXS (P=3)"
    "$BIN/llama-perplexity" -m "$M/UD-IQ3_XXS.gguf" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$L/ppl-UD-IQ3_XXS-S-P3.log" 2>&1
    grep -E 'Mean +KLD|Same top p' "$L/ppl-UD-IQ3_XXS-S-P3.log" || true
fi

for spec in B1:0-5 B2:6-11 B3:12-17 B4:18-19 B5:20-23; do
    b=${spec%%:*} range=${spec#*:}
    layers=$(seq -s '|' "${range%-*}" "${range#*-}")
    for lvl in dnq2o; do
        name="map-$b-$lvl-S-P3"
        done_log "$L/ppl-$name.log" && continue
        args=""
        for l in $(seq "${range%-*}" "${range#*-}"); do
            args="$args --take $M/q2opt/L$l.gguf ^blk\\.$l\\.ffn_down_exps\\.weight\$"
            [ "$lvl" = q2o ] && args="$args --take $M/q2opt/L$l.gguf ^blk\\.$l\\.ffn_gate_up_exps\\.weight\$"
        done
        [ "$lvl" = dnq2o ] && args="$args --take $M/pool-3bit.gguf ^blk\\.($layers)\\.ffn_gate_up_exps\\.weight\$"
        out="$T/$name.gguf"
        step "splice $name"
        # shellcheck disable=SC2086
        $PY "$ROOT/scripts/splice.py" "$REF" "$out" $args --verify > "$L/splice-$name.json"
        step "perplexity $name (P=3)"
        "$BIN/llama-perplexity" -m "$out" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
            --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$L/ppl-$name.log" 2>&1
        grep -E 'Mean +KLD|Same top p' "$L/ppl-$name.log" || true
        rm -f "$out"
    done
done
step "done"
