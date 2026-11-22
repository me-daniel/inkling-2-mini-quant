#!/bin/sh
# Addendum H amendment 10: cand-u = cand with every expert row rescaled to gain 1 (scripts/degain2.py, codes unchanged),
# all 24 layers, gate_up and down; offline gains for cand (all layers) and UD (layers 0 6 12 19 23 where the type is
# supported) to results/pilot/cand-gain.tsv; then S, paired. Resumes. DRY=1 prints only.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"; M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; T="$M/tmp"; D="$M/degain"
PY="$ROOT/.venv/bin/python -I"
. "$ROOT/scripts/gpu-lock.sh"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p "$D" "$T"; G="$ROOT/results/pilot/cand-gain.tsv"
[ -f "$G" ] || printf 'file\ttensor\ttype\tgain_before_after\tenergy_before_after\n' > "$G"
dg() { # file src layer kind out
    grep -q "^$1	blk.$3.$4.weight	" "$G" && [ -f "$5" -o "$1" = UD ] && return 0
    [ "${DRY:-0}" = 1 ] && { echo "  degain $1 $3 $4"; return 0; }
    cpu_take
    r=$(nice -n 10 $PY "$ROOT/scripts/degain2.py" "$2" "$5" "$3" --kind "$4" 2>&1 | tail -1)
    cpu_give
    printf '%s\t%s\n' "$1" "$r" >> "$G"
}
step "degain donors (CPU lock per donor)"
for l in $(seq 0 23); do for k in ffn_gate_up_exps ffn_down_exps; do dg cand "$M/cand.gguf" $l $k "$D/cand-$l-$k.gguf"; done; done
n=$(ls "$D"/cand-*.gguf 2>/dev/null | wc -l | tr -d " ")
[ "${DRY:-0}" = 1 ] || [ "$n" -eq 48 ] || { echo "only $n of 48 cand donors; see $G" >&2; exit 1; }
for l in 0 6 12 19 23; do for k in ffn_gate_up_exps ffn_down_exps; do
    dg UD "$M/UD-IQ3_XXS.gguf" $l $k "$T/ud-degain.gguf" || echo "  UD $l $k: type not supported"; rm -f "$T/ud-degain.gguf"
done; done
args=""; for f in "$D"/cand-*.gguf; do l=$(basename "$f" | cut -d- -f2); k=$(basename "$f" .gguf | cut -d- -f3); args="$args --take $f ^blk\\.$l\\.$k\\.weight\$"; done
log="$L/ppl-cand-u-S-P3.log"
if ! { [ -f "$log" ] && grep -q 'Mean *KLD' "$log"; }; then
    step "splice cand-u"
    [ "${DRY:-0}" = 1 ] && { echo "  splice.py cand cand-u$args" | cut -c1-300; exit 0; }
    cpu_take
    # shellcheck disable=SC2086
    $PY "$ROOT/scripts/splice.py" "$M/cand.gguf" "$T/cand-u.gguf" $args --verify > "$L/splice-cand-u.json"
    cpu_give
    step "perplexity cand-u (P=3)"
    gpu_take
    "$BIN/llama-perplexity" -m "$T/cand-u.gguf" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$log" 2>&1
    gpu_give
    rm -f "$T/cand-u.gguf"
fi
grep -E 'Mean +KLD|Same top p' "$log"
step "cand-u done"
