#!/bin/sh
# Addendum G (CHANGELOG.md): the Mac track after the quality FAIL. Resumes; GPU jobs one at a time.
# 1. Science: noise-tied draws 2 and 3 (base IQ4_NL, B2, S, P = 3), then the per-pass KL table.
# 2. Registered phase-4 step 5: stability of the reference and the frozen candidate.
# 3. Outside comparators (scripts/fetch-ext.sh): tensor types, KL on T1/T2 against the Q8_0 reference, compare.
# 4. Registered phase-4 step 6, extended: bench and Metal coverage for every file.
# 5. CPU: per-pass residual with the full-vocabulary norm, and the report.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; T="$M/tmp"; R="$ROOT/results/pilot"; E="$M/ext"
PY="$ROOT/.venv/bin/python -I"
B2='^blk\.(6|7|8|9|10|11)\.ffn_(gate_up|down)_exps\.weight$'
EXT="bart-IQ2_S atomic-AD-IQ2_S apex-I-Nano bart-IQ2_M isvalorum-NanoPlus prithiv-Q2_K bart-Q2_K"
mkdir -p "$T"
# shared with the sibling repo on this Mac: one heavy GPU run at a time (lock holds the owner's name)
GLOCK="$HOME/GitHub/.gpu.lock"
gpu_take() { while [ -f "$GLOCK" ] && [ "$(cat "$GLOCK" 2>/dev/null)" != "inkling-2-mini-quant" ]; do sleep 30; done; echo inkling-2-mini-quant > "$GLOCK"; }
gpu_give() { [ "$(cat "$GLOCK" 2>/dev/null)" = "inkling-2-mini-quant" ] && rm -f "$GLOCK"; return 0; }
trap gpu_give EXIT
step() { echo "[$(date '+%H:%M:%S')] $*"; }
# one heavy GPU run at a time: wait while another chain holds the lock
while [ -f "$ROOT/logs/gpu.lock" ]; do sleep 30; done
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
ctx() { if [ "$1" = T2 ]; then echo 2048; else echo 4096; fi; }

# 1
for d in 2 3; do
    kld="$K/pp-noise-tied$d.kld"; [ -f "$kld" ] && continue
    step "splice noise-tied$d"
    $PY "$ROOT/scripts/splice.py" "$M/Inkling-2-Mini-IQ4_NL.gguf" "$T/pp-noise-tied$d.gguf" --take "$M/noise-$d.gguf" "$B2" --verify > "$L/splice-pp-noise-tied$d.json"
    step "logits noise-tied$d"
    gpu_take
    "$BIN/llama-perplexity" -m "$T/pp-noise-tied$d.gguf" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$kld.part" > "$L/ppl-pp-noise-tied$d.log" 2>&1
    gpu_give
    mv "$kld.part" "$kld"; rm -f "$T/pp-noise-tied$d.gguf"
done
grep -q '^noise-tied3	' "$R/perpass-kl.tsv" || { step "per-pass KL table"; sh "$ROOT/scripts/perpass-kl.sh" > /dev/null; }

# 2
for name in ref:Inkling-2-Mini-Q8_0 cand:cand; do
    [ -f "$R/stability-${name%%:*}.json" ] && continue
    step "stability ${name%%:*}"
    $PY "$ROOT/scripts/stability.py" "${name%%:*}" "$M/${name#*:}.gguf" > "$L/stability-${name%%:*}.out" 2>&1
done

# 3
for name in $EXT; do
    # already scored (the file may be deleted for disk since): skip without waiting for it
    if [ -f "$R/types-$name.txt" ] && done_log "$L/ppl-final-$name-T1-P3.log" && done_log "$L/ppl-final-$name-T2-P3.log"; then continue; fi
    while [ ! -e "$E/$name.gguf" ]; do sleep 60; done
    [ -f "$R/types-$name.txt" ] || $PY "$ROOT/scripts/gguf-types.py" "$E/$name.gguf" > "$R/types-$name.txt"
    for t in T1 T2; do
        log="$L/ppl-final-$name-$t-P3.log"; done_log "$log" && continue
        step "KL $name on $t"
        gpu_take
        "$BIN/llama-perplexity" -m "$E/$name.gguf" -f "$ROOT/data/$t.txt" -c "$(ctx "$t")" -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
            --kl-divergence-base "$K/ref-$t-P3.kld" --kl-divergence > "$log" 2>&1
    gpu_give
        grep -E 'Mean +KLD|Same top p' "$log" || true
    done
done
$PY "$ROOT/analysis/maptable.py" > /dev/null
# shellcheck disable=SC2086
$PY "$ROOT/analysis/compare.py" cand bart-IQ2_S atomic-AD-IQ2_S apex-I-Nano isvalorum-NanoPlus prithiv-Q2_K | tee "$R/mac-quality.txt"
$PY "$ROOT/analysis/compare.py" cand bart-IQ2_M | grep -v 'quality gate' > "$R/mac-context-IQ2_M.txt"
# Addendum H6: every all-Metal file under 12 GB we scored
$PY "$ROOT/analysis/compare.py" cand bart-IQ2_S atomic-AD-IQ2_S apex-I-Nano isvalorum-NanoPlus prithiv-Q2_K bart-IQ2_M bart-Q2_K | tee "$R/under12-quality.txt"

# 4 (only with BENCH=1, on an idle machine)
[ "${BENCH:-0}" = 1 ] && for name in cand uniform-IQ3_XXS UD-IQ3_XXS $EXT; do
    [ -f "$R/bench-$name.txt" ] && continue
    f="$M/$name.gguf"; [ -e "$f" ] || f="$E/$name.gguf"
    step "bench $name"
    # llama-bench has no --override-kv; every file is checked for loop.count = 3 in its types file
    "$BIN/llama-bench" -m "$f" -ngl 99 -p 512 -n 128 -r 3 -o md > "$R/bench-$name.txt" 2> "$L/bench-$name.log" || true
    $PY -c "import gguf,sys; r=gguf.GGUFReader(sys.argv[1]); print('expert tensors in types Metal cannot multiply:', sum(t.tensor_type.name in ('NVFP4','F8_E4M3','TQ1_0') for t in r.tensors if 'exps' in t.name))" "$f" >> "$R/bench-$name.txt"
done
# 5. CPU: the per-pass residual with the full-vocabulary norm (audit finding 3), then the report
if ! grep -q full_euclid "$R/perpass-residual.json"; then
    step "per-pass residual, full vocabulary"
    $PY "$ROOT/analysis/per_pass.py" "$K/pp-base.kld" "$K/pp-p1.kld" "$K/pp-p2.kld" "$K/pp-p3.kld" "$K/pp-all.kld" > "$R/perpass-residual.json.part"
    mv "$R/perpass-residual.json.part" "$R/perpass-residual.json"
fi
$PY "$ROOT/analysis/perpass_report.py" > /dev/null
step "macgate done"
