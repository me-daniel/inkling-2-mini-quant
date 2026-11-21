#!/bin/sh
# Pilot G0 (CHANGELOG.md, 2026-11-21): BF16 conversion, Q8_0 reference, the IQ4_NL floor build and the
# 2-bit and 3-bit expert pools, then reference logits on S at P = 3 and the two floors.
# Each step is skipped when its output exists, so a rerun resumes. One heavy process at a time.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"
L="$ROOT/logs"
K="$ROOT/models/logits"
IMAT="$M/ggml-org/imatrix.gguf"
mkdir -p "$L" "$K"

step() { echo "[$(date '+%H:%M:%S')] $*"; }

if [ ! -f "$M/Inkling-2-Mini-BF16.gguf" ]; then
    step "convert to BF16"
    "$ROOT/.venv/bin/python" "$ROOT/vendor/llama.cpp/convert_hf_to_gguf.py" "$M/Inkling-2-Mini" \
        --outtype bf16 --outfile "$M/Inkling-2-Mini-BF16.gguf" > "$L/convert.log" 2>&1
fi

quant() { # out ftype [tensor-type overrides...]
    out=$1 ftype=$2; shift 2
    [ -f "$out" ] && return 0
    step "quantize $(basename "$out")"
    tt=""
    for o in "$@"; do tt="$tt --tensor-type $o"; done
    # shellcheck disable=SC2086
    "$BIN/llama-quantize" --imatrix "$IMAT" $tt "$M/Inkling-2-Mini-BF16.gguf" "$out" "$ftype" > "$L/quant-$(basename "$out" .gguf).log" 2>&1
    grep -E 'fallback|n_fallback' "$L/quant-$(basename "$out" .gguf).log" | tail -2 || true
}

quant "$M/Inkling-2-Mini-Q8_0.gguf" Q8_0
quant "$M/Inkling-2-Mini-IQ4_NL.gguf" IQ4_NL
quant "$M/pool-2bit.gguf" Q8_0 ffn_gate_up_exps=iq2_xxs ffn_down_exps=q2_0
quant "$M/pool-3bit.gguf" Q8_0 ffn_gate_up_exps=iq3_xxs ffn_down_exps=iq4_nl

ppl() { # name model passes extra-args...
    name=$1 model=$2 p=$3; shift 3
    step "perplexity $name (P=$p)"
    "$BIN/llama-perplexity" -m "$model" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:$p" "$@" > "$L/ppl-$name.log" 2>&1
    grep -E 'Final estimate|Mean +KLD|Same top p|llama_perf_context_print: +total' "$L/ppl-$name.log" || true
}

[ -f "$K/ref-S-P3.kld" ] || ppl ref-S-P3 "$M/Inkling-2-Mini-Q8_0.gguf" 3 --kl-divergence-base "$K/ref-S-P3.kld"
ppl floor-q8-S-P3    "$M/Inkling-2-Mini-Q8_0.gguf"   3 --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence
ppl floor-iq4nl-S-P3 "$M/Inkling-2-Mini-IQ4_NL.gguf" 3 --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence
step "done"
