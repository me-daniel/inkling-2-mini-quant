#!/bin/sh
# Addendum D (CHANGELOG.md): per-block gate_up-only 2-bit levels (IQ2_S or IQ2_XXS gate_up, IQ4_NL down), S, P = 3.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; T="$M/tmp"
PY="$ROOT/.venv/bin/python -I"
REF="$M/Inkling-2-Mini-Q8_0.gguf"
mkdir -p "$T"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }

if [ ! -f "$M/pool-gu2s.gguf" ]; then
    step "quantize pool-gu2s"
    "$BIN/llama-quantize" --imatrix "$M/ggml-org/imatrix.gguf" --tensor-type ffn_gate_up_exps=iq2_s --tensor-type ffn_down_exps=iq4_nl \
        "$M/Inkling-2-Mini-BF16.gguf" "$M/pool-gu2s.gguf.part" Q8_0 > "$L/quant-pool-gu2s.log" 2>&1
    mv "$M/pool-gu2s.gguf.part" "$M/pool-gu2s.gguf"
fi

for spec in B1:0-5 B2:6-11 B3:12-17 B4:18-19 B5:20-23; do
    b=${spec%%:*} range=${spec#*:}
    layers=$(seq -s '|' "${range%-*}" "${range#*-}")
    for lvl in gu2s gu2xxs; do
        name="map-$b-$lvl-S-P3"
        done_log "$L/ppl-$name.log" && continue
        donor="$M/pool-gu2s.gguf"; [ "$lvl" = gu2xxs ] && donor="$M/pool-2bit.gguf"
        out="$T/$name.gguf"
        step "splice $name"
        $PY "$ROOT/scripts/splice.py" "$REF" "$out" \
            --take "$donor" "^blk\.($layers)\.ffn_gate_up_exps\.weight$" \
            --take "$M/pool-3bit.gguf" "^blk\.($layers)\.ffn_down_exps\.weight$" --verify > "$L/splice-$name.json"
        step "perplexity $name (P=3)"
        "$BIN/llama-perplexity" -m "$out" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
            --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$L/ppl-$name.log" 2>&1
        grep -E 'Mean +KLD|Same top p' "$L/ppl-$name.log" || true
        rm -f "$out"
    done
done
step "done"
