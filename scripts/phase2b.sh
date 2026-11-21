#!/bin/sh
# Addendum C map entry (CHANGELOG.md): non-expert tensors at Q6_K, experts as in the Q8_0 reference, on S at P = 3.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"
L="$ROOT/logs"
K="$M/logits"
out="$M/tmp/ne-q6k.gguf"
log="$L/ppl-map-NE-Q6K-S-P3.log"
mkdir -p "$M/tmp"

step() { echo "[$(date '+%H:%M:%S')] $*"; }

if ! { [ -f "$log" ] && grep -q 'Mean *KLD' "$log"; }; then
    step "quantize non-experts to Q6_K"
    "$BIN/llama-quantize" --imatrix "$M/ggml-org/imatrix.gguf" \
        --tensor-type ffn_gate_up_exps=q8_0 --tensor-type ffn_down_exps=q8_0 \
        "$M/Inkling-2-Mini-BF16.gguf" "$out.part" Q6_K > "$L/quant-ne-q6k.log" 2>&1
    mv "$out.part" "$out"
    grep -E 'quant size' "$L/quant-ne-q6k.log" || true
    step "perplexity map-NE-Q6K-S-P3 (P=3)"
    "$BIN/llama-perplexity" -m "$out" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$log" 2>&1
    grep -E 'Mean +KLD|Same top p' "$log" || true
    rm -f "$out"
fi
step "done"
