#!/bin/sh
# Pilot phase 4 (CHANGELOG.md, entry and addenda C, D): T2 traces, the allocation, the candidate and the stock IQ3_XXS
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

# 1. T2 traces, frozen before any candidate exists
[ -f "$ROOT/data/T2.txt" ] || { step "T2 traces"; $PY "$ROOT/scripts/t2-traces.py" > "$L/t2.out" 2>&1; tail -1 "$L/t2.out"; }

# 2. reference logits on T1 and T2
for t in T1 T2; do
    [ -f "$K/ref-$t-P3.kld" ] && continue
    step "reference logits $t"
    "$BIN/llama-perplexity" -m "$REF" -f "$ROOT/data/$t.txt" -c 4096 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base "$K/ref-$t-P3.kld.part" > "$L/ppl-ref-$t-P3.log" 2>&1
    mv "$K/ref-$t-P3.kld.part" "$K/ref-$t-P3.kld"
done

# 3. allocation from the S map (analysis/allocate.py), then the two builds
$PY "$ROOT/analysis/maptable.py" > /dev/null
[ -f "$R/allocation-best.json" ] || { step "allocate"; $PY "$ROOT/analysis/allocate.py" | tee "$R/allocations.txt"; }
build() { # out ftype [tensor-type-file]
    [ -f "$1" ] && return 0
    step "quantize $(basename "$1")"
    if [ $# -eq 3 ]; then tt="--tensor-type-file $3"; else tt=""; fi
    # shellcheck disable=SC2086
    "$BIN/llama-quantize" --imatrix "$M/ggml-org/imatrix.gguf" $tt "$M/Inkling-2-Mini-BF16.gguf" "$1.part" "$2" > "$L/quant-$(basename "$1" .gguf).log" 2>&1
    mv "$1.part" "$1"
    size=$(stat -f %z "$1")
    echo "  $(basename "$1"): $size bytes (limit $LIMIT)"
    [ "$size" -le "$LIMIT" ] || { echo "  over the byte limit, stopping" >&2; exit 1; }
}
$PY -c "import json,sys; a=json.load(open(sys.argv[1]))['rules']; open(sys.argv[2],'w').write(''.join(f'{r}={t.lower()}\n' for r,t in a['rules']))" \
    "$R/allocation-best.json" "$R/allocation-best.tensor-types.txt"
NE=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['ne'])" "$R/allocation-best.json")
build "$M/cand.gguf" "$NE" "$R/allocation-best.tensor-types.txt"
build "$M/uniform-IQ3_XXS.gguf" IQ3_XXS

# 4. quality on T1 and T2 at P = 3
for name in cand uniform-IQ3_XXS UD-IQ3_XXS; do
    for t in T1 T2; do
        log="$L/ppl-final-$name-$t-P3.log"
        done_log "$log" && continue
        step "KL $name on $t"
        "$BIN/llama-perplexity" -m "$M/$name.gguf" -f "$ROOT/data/$t.txt" -c 4096 -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
            --kl-divergence-base "$K/ref-$t-P3.kld" --kl-divergence > "$log" 2>&1
        grep -E 'Mean +KLD|Same top p' "$log" || true
    done
done
$PY "$ROOT/analysis/maptable.py" > /dev/null
$PY "$ROOT/analysis/compare.py" cand uniform-IQ3_XXS UD-IQ3_XXS | tee "$R/gate-quality.txt"

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
