#!/bin/sh
# Addendum H (CHANGELOG.md): further per-pass controls on the IQ4_NL base, S, P = 3. Each group is analysed with
# analysis/per_pass.py (base, pass 1 only, pass 2 only, pass 3 only, all) as soon as its logits exist, its KL rows
# go to results/pilot/perpass3-kl.tsv, and its logits are deleted (pp-base.kld stays). Resumes. DRY=1 prints only.
#   H1  layer split at the 3-bit rung:       3b-ls67, 3b-ls89, 3b-ls1011, 3b-all
#   H2  tied Gaussian noise, one pass each:  g1-p1, g1-p2, g1-p3 (draw 1), noise-tied (draw 1 on all passes)
#   H3  independent Gaussian noise:          g2-p2 (draw 2), g3-p3 (draw 3); with g1-p1 and noise-indep (draws 1/2/3)
#   H4  two more independent orders:         indep-231, indep-312
#   H5  the pass split on B3 (layers 12-17): b3-p1, b3-p2, b3-p3, b3-all (2-bit pool)
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; T="$M/tmp"; R="$ROOT/results/pilot"
PY="$ROOT/.venv/bin/python -I"
BASE="$M/Inkling-2-Mini-IQ4_NL.gguf"; TWO="$M/pool-2bit.gguf"; THREE="$M/pool-3bit.gguf"
N1="$M/noise-1.gguf"; N2="$M/noise-2.gguf"; N3="$M/noise-3.gguf"
B2='^blk\.(6|7|8|9|10|11)\.ffn_(gate_up|down)_exps\.weight$'
B3='^blk\.(12|13|14|15|16|17)\.ffn_(gate_up|down)_exps\.weight$'
L2="6 7 8 9 10 11"; L3="12 13 14 15 16 17"
KL="$R/perpass3-kl.tsv"
mkdir -p "$T" "$K"

# shared with the sibling repo on this Mac: one heavy GPU run at a time (lock holds the owner's name)
GLOCK="$HOME/GitHub/.gpu.lock"
gpu_take() { while [ -f "$GLOCK" ] && [ "$(cat "$GLOCK" 2>/dev/null)" != "inkling-2-mini-quant" ]; do sleep 30; done; echo inkling-2-mini-quant > "$GLOCK"; }
gpu_give() { [ "$(cat "$GLOCK" 2>/dev/null)" = "inkling-2-mini-quant" ] && rm -f "$GLOCK"; return 0; }
trap gpu_give EXIT
step() { echo "[$(date '+%H:%M:%S')] $*"; }
# slots DONOR PASS LAYERS: --slot args putting DONOR's experts of each layer into that layer's cache slot for PASS
slots() { for l in $3; do printf ' --slot %s %s %s' "$1" "$l" $((24 + ($2 - 2) * 12 + l - 6)); done; }
run() { # cond splice-args...
    cond=$1; shift
    kld="$K/pp-$cond.kld"
    [ -f "$kld" ] && return 0
    grep -q "^$cond	" "$KL" 2>/dev/null && return 0  # already scored and deleted
    step "splice $cond"
    if [ "${DRY:-0}" = 1 ]; then echo "  splice.py BASE OUT $*"; return 0; fi
    # shellcheck disable=SC2086
    $PY "$ROOT/scripts/splice.py" "$BASE" "$T/pp-$cond.gguf" "$@" --verify > "$L/splice-pp-$cond.json"
    step "logits $cond"
    gpu_take
    "$BIN/llama-perplexity" -m "$T/pp-$cond.gguf" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$kld.part" > "$L/ppl-pp-$cond.log" 2>&1
    gpu_give
    mv "$kld.part" "$kld"
    rm -f "$T/pp-$cond.gguf"
}
kl_rows() { # conds... -> cond, mean KL against pp-base, per-chunk values
    [ -f "$KL" ] || printf 'cond\tagainst\tmean_kl\tper_chunk\n' > "$KL"
    for c in "$@"; do
        grep -q "^$c	" "$KL" && continue
        [ "${DRY:-0}" = 1 ] && { echo "  kl row $c"; continue; }
        out=$($PY "$ROOT/scripts/kldfile.py" "$K/pp-base.kld" "$K/pp-$c.kld")
        printf '%s\tpp-base\t%s\t%s\n' "$c" "$(echo "$out" | tail -1 | cut -d' ' -f3)" "$(echo "$out" | head -1 | tr ' ' ',')" >> "$KL"
    done
}
analyse() { # out.json s1 s2 s3 all
    out=$1; shift
    [ -f "$out" ] && return 0
    step "residual $(basename "$out")"
    [ "${DRY:-0}" = 1 ] && return 0
    $PY "$ROOT/analysis/per_pass.py" "$K/pp-base.kld" "$K/pp-$1.kld" "$K/pp-$2.kld" "$K/pp-$3.kld" "$K/pp-$4.kld" > "$out.part"
    mv "$out.part" "$out"
}
drop() { [ "${DRY:-0}" = 1 ] || for c in "$@"; do rm -f "$K/pp-$c.kld"; done; }

[ -f "$K/pp-base.kld" ] || { echo "pp-base.kld missing" >&2; exit 1; }

# shellcheck disable=SC2046
{
# H1
run 3b-ls67   --take "$THREE" '^blk\.(6|7)\.ffn_(gate_up|down)_exps\.weight$'
run 3b-ls89   --take "$THREE" '^blk\.(8|9)\.ffn_(gate_up|down)_exps\.weight$'
run 3b-ls1011 --take "$THREE" '^blk\.(10|11)\.ffn_(gate_up|down)_exps\.weight$'
run 3b-all    --take "$THREE" "$B2"
analyse "$R/layersplit3bit-residual.json" 3b-ls67 3b-ls89 3b-ls1011 3b-all
kl_rows 3b-ls67 3b-ls89 3b-ls1011 3b-all; drop 3b-ls67 3b-ls89 3b-ls1011 3b-all

# H2 and H3 share g1-p1
run g1-p1      --take "$N1" "$B2" $(slots "$BASE" 2 "$L2") $(slots "$BASE" 3 "$L2")
run g1-p2      $(slots "$N1" 2 "$L2")
run g1-p3      $(slots "$N1" 3 "$L2")
run noise-tied --take "$N1" "$B2"
analyse "$R/noise-tied-residual.json" g1-p1 g1-p2 g1-p3 noise-tied
kl_rows g1-p1 g1-p2 g1-p3 noise-tied; drop g1-p2 g1-p3 noise-tied
run g2-p2       $(slots "$N2" 2 "$L2")
run g3-p3       $(slots "$N3" 3 "$L2")
run noise-indep --take "$N1" "$B2" $(slots "$N2" 2 "$L2") $(slots "$N3" 3 "$L2")
analyse "$R/noise-indep-residual.json" g1-p1 g2-p2 g3-p3 noise-indep
kl_rows g2-p2 g3-p3 noise-indep; drop g1-p1 g2-p2 g3-p3 noise-indep

# H4
run indep-231 --take "$N2" "$B2" $(slots "$N3" 2 "$L2") $(slots "$N1" 3 "$L2")
run indep-312 --take "$N3" "$B2" $(slots "$N1" 2 "$L2") $(slots "$N2" 3 "$L2")
kl_rows indep-231 indep-312; drop indep-231 indep-312

# H5
run b3-p1  --take "$TWO" "$B3" $(slots "$BASE" 2 "$L3") $(slots "$BASE" 3 "$L3")
run b3-p2  $(slots "$TWO" 2 "$L3")
run b3-p3  $(slots "$TWO" 3 "$L3")
run b3-all --take "$TWO" "$B3"
analyse "$R/perpass-b3-residual.json" b3-p1 b3-p2 b3-p3 b3-all
kl_rows b3-p1 b3-p2 b3-p3 b3-all; drop b3-p1 b3-p2 b3-p3 b3-all
}
step "perpass3 done"
