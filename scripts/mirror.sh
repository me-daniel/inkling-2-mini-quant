#!/bin/sh
# Addendum H (CHANGELOG.md): the mirror half of the 2 x 2 on S, P = 3. Diagnostic, not a release candidate.
# UD-IQ3_XXS with our IQ4_NL down projections (pool-3bit, the same bytes as cand's) in layers 0-19, and the
# per-expert .scale of those down tensors dropped (our IQ4_NL values are already at full scale).
# Resumes; the GGUF is deleted after scoring.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; T="$M/tmp"
PY="$ROOT/.venv/bin/python -I"
OUT="$T/mirror.gguf"; LOG="$L/ppl-mirror-S-P3.log"
DOWN='^blk\.([0-9]|1[0-9])\.ffn_down_exps\.'
mkdir -p "$T"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
if done_log "$LOG"; then grep -E 'Mean +KLD|Same top p' "$LOG"; rm -f "$OUT"; exit 0; fi
step "splice mirror"
$PY "$ROOT/scripts/splice.py" "$M/UD-IQ3_XXS.gguf" "$OUT" --take "$M/pool-3bit.gguf" "${DOWN}weight\$" --drop "${DOWN}scale\$" \
    --verify > "$L/splice-mirror.json"
$PY "$ROOT/scripts/gguf-types.py" "$OUT" | tee "$L/types-mirror.txt"
grep -q "tensors in types Metal cannot multiply: 0 " "$L/types-mirror.txt" || { echo "mirror still has off-Metal tensors" >&2; exit 1; }
step "perplexity mirror (P=3)"
"$BIN/llama-perplexity" -m "$OUT" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
    --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$LOG" 2>&1
grep -E 'Mean +KLD|Same top p' "$LOG"
rm -f "$OUT"
