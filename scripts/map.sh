#!/bin/sh
# Pilot phase 2 (CHANGELOG.md, 2026-11-21): block map at P = 3 and pass sweep at P = 2/3/4, all on S.
# Block builds are spliced from the Q8_0 reference and a pool, scored, then deleted. Resumes: a run whose
# log already shows a KLD result is skipped. One heavy process at a time.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"
L="$ROOT/logs"
K="$M/logits"
T="$M/tmp"
PY="$ROOT/.venv/bin/python -I"
REF="$M/Inkling-2-Mini-Q8_0.gguf"
mkdir -p "$T" "$K"

step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }

ppl() { # name model passes extra-args...
    name=$1 model=$2 p=$3; shift 3
    log="$L/ppl-$name.log"
    if done_log "$log"; then return 0; fi
    step "perplexity $name (P=$p)"
    "$BIN/llama-perplexity" -m "$model" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:$p" "$@" > "$log" 2>&1
    grep -E 'Mean +KLD|Same top p' "$log" || true
}

refbase() { # passes
    [ -f "$K/ref-S-P$1.kld" ] && return 0
    step "reference logits P=$1"
    "$BIN/llama-perplexity" -m "$REF" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:$1" --kl-divergence-base "$K/ref-S-P$1.kld.part" > "$L/ppl-ref-S-P$1.log" 2>&1
    mv "$K/ref-S-P$1.kld.part" "$K/ref-S-P$1.kld"
}

# block map at P = 3
for spec in B1:0-5 B2:6-11 B3:12-17 B4:18-19 B5:20-23; do
    b=${spec%%:*} range=${spec#*:}
    layers=$(seq -s '|' "${range%-*}" "${range#*-}")
    for lvl in 2bit 3bit; do
        name="map-$b-$lvl-S-P3"
        done_log "$L/ppl-$name.log" && continue
        out="$T/$name.gguf"
        step "splice $name"
        $PY "$ROOT/scripts/splice.py" "$REF" "$out" \
            --take "$M/pool-$lvl.gguf" "^blk\.($layers)\.ffn_(gate_up|down)_exps\.weight$" --verify > "$L/splice-$name.json"
        ppl "$name" "$out" 3 --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence
        rm -f "$out"
    done
done

# pass sweep: uniform pools against the reference at the same pass count
for p in 2 3 4; do
    refbase "$p"
    for lvl in 2bit 3bit; do
        ppl "sweep-$lvl-S-P$p" "$M/pool-$lvl.gguf" "$p" --kl-divergence-base "$K/ref-S-P$p.kld" --kl-divergence
    done
done
step "done"
