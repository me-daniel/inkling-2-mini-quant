#!/bin/sh
# Pilot phase 4 (CHANGELOG.md, entry and addenda C-F): T2 traces, the allocation, the candidate and the stock IQ3_XXS
# baseline, quality on T1 and T2 against Unsloth UD-IQ3_XXS, stability, speed. Stops before any upload. Resumes.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; R="$ROOT/results/pilot"
PY="$ROOT/.venv/bin/python -I"
REF="$M/Inkling-2-Mini-Q8_0.gguf"
LIMIT=11663310784  # bytes of Unsloth's UD-IQ3_XXS
mkdir -p "$R"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
ctx() { if [ "$1" = T2 ]; then echo 2048; else echo 4096; fi; }  # T2: 40 traces, scored at 2048 for enough chunks

# 1. T2 traces, frozen before any candidate exists
[ -f "$ROOT/data/T2.txt" ] || { step "T2 traces"; $PY "$ROOT/scripts/t2-traces.py" > "$L/t2.out" 2>&1; tail -1 "$L/t2.out"; }

# 2. reference logits on T1 and T2
for t in T1 T2; do
    [ -f "$K/ref-$t-P3.kld" ] && continue
    step "reference logits $t"
    "$BIN/llama-perplexity" -m "$REF" -f "$ROOT/data/$t.txt" -c "$(ctx "$t")" -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base "$K/ref-$t-P3.kld.part" > "$L/ppl-ref-$t-P3.log" 2>&1
    mv "$K/ref-$t-P3.kld.part" "$K/ref-$t-P3.kld"
done

# 3. allocation from the S map: 4 structurally different candidates (analysis/allocate.py), each spliced from the pools
#    and measured on S; the measured best becomes the candidate (Addendum F). T1 and T2 stay untouched until then.
$PY "$ROOT/analysis/maptable.py" > /dev/null
[ -f "$R/allocation-top.json" ] || { step "allocate"; $PY "$ROOT/analysis/allocate.py" | tee "$R/allocations.txt"; }
build() { # out ftype [extra llama-quantize args...]
    [ -f "$1" ] && return 0
    out=$1 ftype=$2; shift 2
    step "quantize $(basename "$out")"
    "$BIN/llama-quantize" --imatrix "$M/ggml-org/imatrix.gguf" "$@" "$M/Inkling-2-Mini-BF16.gguf" "$out.part" "$ftype" > "$L/quant-$(basename "$out" .gguf).log" 2>&1
    mv "$out.part" "$out"
}
jq_top() { $PY -c "import json,sys; c=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" "$R/allocation-top.json" "$1"; }
if [ ! -f "$M/cand.gguf" ]; then
    [ "$(jq_top "int(any(x['ne'] == 'Q6_K' for x in c))")" = 1 ] && \
        build "$M/ne-q6k-base.gguf" Q6_K --tensor-type ffn_gate_up_exps=q8_0 --tensor-type ffn_down_exps=q8_0
    N=$(jq_top "len(c)")
    best="" best_kl=""
    i=0
    while [ "$i" -lt "$N" ]; do
        name="sel-$i-S-P3" out="$M/sel-$i.gguf"
        if ! done_log "$L/ppl-$name.log"; then
            base="$REF"; [ "$(jq_top "c[$i]['ne']")" = Q6_K ] && base="$M/ne-q6k-base.gguf"
            step "splice sel-$i"
            # shellcheck disable=SC2046
            $PY "$ROOT/scripts/splice.py" "$base" "$out.part" $($PY "$ROOT/analysis/allocate.py" --splice-args "$i" "$M") --verify > "$L/splice-sel-$i.json"
            mv "$out.part" "$out"
            size=$(stat -f %z "$out")
            echo "  sel-$i: $size bytes (limit $LIMIT)"
            [ "$size" -le "$LIMIT" ] || { echo "  over the byte limit, stopping" >&2; exit 1; }
            step "perplexity $name"
            "$BIN/llama-perplexity" -m "$out" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
                --kl-divergence-base "$K/ref-S-P3.kld" --kl-divergence > "$L/ppl-$name.log" 2>&1
            grep -E 'Mean +KLD' "$L/ppl-$name.log" || true
        fi
        kl=$(grep -E 'Mean +KLD' "$L/ppl-$name.log" | awk '{print $3}')
        # keep only the best file so far on disk (each is ~11.6 GB)
        if [ -z "$best" ] || [ "$(echo "$kl < $best_kl" | bc -l)" = 1 ]; then
            [ -n "$best" ] && rm -f "$M/sel-$best.gguf"
            best=$i best_kl=$kl
        else
            rm -f "$out"
        fi
        i=$((i + 1))
    done
    [ -f "$M/sel-$best.gguf" ] || { echo "the best candidate file is missing (resumed after cleanup?); delete logs/ppl-sel-$best-S-P3.log and rerun" >&2; exit 1; }
    $PY "$ROOT/analysis/maptable.py" > /dev/null
    $PY -c "
import csv, json, sys
kl = {r['run']: float(r['mean_kl']) for r in csv.DictReader(open(sys.argv[1]), delimiter='\t')}
top = json.load(open(sys.argv[2]))
print('| candidate | predicted KL | measured KL on S | bytes | NE | choice |'); print('|---|---|---|---|---|---|')
for i, c in enumerate(top):
    print(f\"| sel-{i}{' (chosen)' if i == int(sys.argv[3]) else ''} | {c['predicted_kl']:.5f} | {kl[f'sel-{i}-S-P3']:.5f} | {c['bytes']:,} | {c['ne']} | {' '.join(f'{b}:{g}/{d}' for b, (g, d) in c['choice'].items())} |\")
" "$R/kl.tsv" "$R/allocation-top.json" "$best" > "$R/selection.md"
    cat "$R/selection.md"
    mv "$M/sel-$best.gguf" "$M/cand.gguf"
    rm -f "$M/ne-q6k-base.gguf"
fi
build "$M/uniform-IQ3_XXS.gguf" IQ3_XXS

# 4. quality on T1 and T2 at P = 3
for name in cand uniform-IQ3_XXS UD-IQ3_XXS; do
    for t in T1 T2; do
        log="$L/ppl-final-$name-$t-P3.log"
        done_log "$log" && continue
        step "KL $name on $t"
        "$BIN/llama-perplexity" -m "$M/$name.gguf" -f "$ROOT/data/$t.txt" -c "$(ctx "$t")" -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
            --kl-divergence-base "$K/ref-$t-P3.kld" --kl-divergence > "$log" 2>&1
        grep -E 'Mean +KLD|Same top p' "$log" || true
    done
done
$PY "$ROOT/analysis/maptable.py" > /dev/null
$PY "$ROOT/analysis/compare.py" cand uniform-IQ3_XXS UD-IQ3_XXS | tee "$R/gate-quality.txt"
grep -q "quality gate: PASS" "$R/gate-quality.txt" || { step "quality gate FAIL: no stability or bench runs for a loser"; exit 2; }

# 5. stability: reference and candidate (thresholds in scripts/stability.py)
for name in ref:Inkling-2-Mini-Q8_0 cand:cand; do
    [ -f "$R/stability-${name%%:*}.json" ] && continue
    step "stability ${name%%:*}"
    $PY "$ROOT/scripts/stability.py" "${name%%:*}" "$M/${name#*:}.gguf" > /dev/null
done

# 6. speed and Metal coverage
for name in cand uniform-IQ3_XXS UD-IQ3_XXS; do
    [ -f "$R/bench-$name.txt" ] && continue
    step "bench $name"
    # llama-bench has no --override-kv; every file here keeps the default loop.count = 3
    "$BIN/llama-bench" -m "$M/$name.gguf" -ngl 99 -p 512 -n 128 -r 3 -o md > "$R/bench-$name.txt" 2> "$L/bench-$name.log" || true
    $PY -c "import gguf,sys; r=gguf.GGUFReader(sys.argv[1]); print('expert tensors in types Metal cannot multiply:', sum(t.tensor_type.name in ('NVFP4','F8_E4M3','TQ1_0') for t in r.tensors if 'exps' in t.name))" \
        "$M/$name.gguf" >> "$R/bench-$name.txt"
done
step "phase 4 done, waiting for the upload decision"
