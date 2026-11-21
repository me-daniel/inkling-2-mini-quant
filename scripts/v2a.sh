#!/bin/sh
# Addendum H amendment 3: build v2a (our quantization at UD-IQ3_XXS's gate_up allocation) and score it on S;
# also the isolating swaps gu15-ud and gu911-ud (via decompose.sh). Resumes. BUILD_ONLY=1 builds v2a and stops.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"
PY="$ROOT/.venv/bin/python -I"
LIMIT=11663310784
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
if [ ! -f "$M/v2a.gguf" ]; then
    step "quantize v2a"
    nice -n 5 "$BIN/llama-quantize" --imatrix "$M/ggml-org/imatrix.gguf" \
        --tensor-type 'blk\.0\.ffn_gate_up_exps=iq2_xxs' \
        --tensor-type 'blk\.4\.ffn_gate_up_exps=iq2_xs' \
        --tensor-type 'blk\.(1|2|3|5|6|7|8|12|18)\.ffn_gate_up_exps=iq2_s' \
        --tensor-type 'blk\.(9|10|11|13|14|15|16|17|19|20|21|22|23)\.ffn_gate_up_exps=iq3_xxs' \
        --tensor-type 'ffn_down_exps=iq4_nl' \
        --token-embedding-type q4_k --output-tensor-type q5_k \
        "$M/Inkling-2-Mini-BF16.gguf" "$M/v2a.gguf.part" Q6_K > "$L/quant-v2a.log" 2>&1
    mv "$M/v2a.gguf.part" "$M/v2a.gguf"
fi
$PY "$ROOT/scripts/gguf-types.py" "$M/v2a.gguf" > "$L/types-v2a.txt"
grep -q "tensors in types Metal cannot multiply: 0 " "$L/types-v2a.txt" || { echo "v2a has off-Metal tensors" >&2; exit 1; }
size=$(stat -f %z "$M/v2a.gguf"); [ "$size" -le "$LIMIT" ] || { echo "v2a is $size bytes, over $LIMIT" >&2; exit 1; }
# the gate_up type of every layer must equal UD-IQ3_XXS's
$PY - "$M/v2a.gguf" "$M/UD-IQ3_XXS.gguf" <<'PY'
import sys, gguf
a, b = ({t.name: t.tensor_type.name for t in gguf.GGUFReader(p).tensors if "gate_up_exps.weight" in t.name} for p in sys.argv[1:3])
bad = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
sys.exit(f"gate_up types differ from UD: {bad}") if bad else print(f"v2a gate_up types match UD in {len(a)} layers")
PY
[ "${BUILD_ONLY:-0}" = 1 ] && { step "v2a built ($size bytes)"; exit 0; }
log="$L/ppl-v2a-S-P3.log"
if ! done_log "$log"; then
    step "perplexity v2a (P=3)"
    "$BIN/llama-perplexity" -m "$M/v2a.gguf" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$log" 2>&1
fi
grep -E 'Mean +KLD|Same top p' "$log"
