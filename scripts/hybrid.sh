#!/bin/sh
# Addendum G amendment 3 (draft, CHANGELOG.md): diagnostic hybrid, S only, P = 3. Not a release candidate.
# cand.gguf with UD-IQ3_XXS's native NVFP4 ffn_down_exps of layers 0-19 and their per-expert .scale tensors.
# Asks how much of the cand vs UD-IQ3_XXS gap on S the down projections explain. NVFP4 runs on the CPU here
# (Addendum C), so expect a run as slow as UD-IQ3_XXS's own S run. Resumes; the GGUF is deleted after scoring.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; T="$M/tmp"
PY="$ROOT/.venv/bin/python -I"
UD="$M/UD-IQ3_XXS.gguf"
OUT="$T/hybrid.gguf"
LOG="$L/ppl-hybrid-S-P3.log"
mkdir -p "$T"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }

if done_log "$LOG"; then
    grep -E 'Mean +KLD|Same top p' "$LOG" || true
    rm -f "$OUT"
    step "done (already scored)"
    exit 0
fi

if ! { [ -f "$OUT" ] && [ -f "$L/splice-hybrid.json" ] && grep -q '"verified": true' "$L/splice-hybrid.json"; }; then
    step "splice hybrid"
    $PY "$ROOT/scripts/splice.py" "$M/cand.gguf" "$OUT" \
        --take "$UD" '^blk\.([0-9]|1[0-9])\.ffn_down_exps\.weight$' \
        --add "$UD" '^blk\.([0-9]|1[0-9])\.ffn_down_exps\.scale$' \
        --verify > "$L/splice-hybrid.json"
fi

step "types hybrid"
$PY "$ROOT/scripts/gguf-types.py" "$OUT" | tee "$L/types-hybrid.txt"
# 20 NVFP4 down tensors, each with its scale, or the run would score unscaled experts
$PY - "$OUT" <<'EOF'
import re, sys
import gguf
t = {x.name: x for x in gguf.GGUFReader(sys.argv[1]).tensors}
down = [n for n in t if re.fullmatch(r"blk\.([0-9]|1[0-9])\.ffn_down_exps\.weight", n)]
bad = [n for n in down if t[n].tensor_type.name != "NVFP4" or n.replace(".weight", ".scale") not in t]
if len(down) != 20 or bad:
    sys.exit(f"hybrid check failed: {len(down)} down tensors, without NVFP4 or scale: {bad}")
print("hybrid check: 20 NVFP4 ffn_down_exps (layers 0-19), each with its .scale")
EOF

step "perplexity hybrid (P=3)"
"$BIN/llama-perplexity" -m "$OUT" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
    --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$LOG" 2>&1
grep -E 'Mean +KLD|Same top p' "$LOG" || true
done_log "$LOG" && rm -f "$OUT"
step "done"
