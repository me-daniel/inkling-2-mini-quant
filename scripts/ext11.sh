#!/bin/sh
# Addendum H amendment 11: download and score the new all-Metal comparators under 11.6 GB on T1 and T2, exactly as
# scripts/macgate.sh scores its comparators (same logs, same reference, P = 3). One file at a time; each GGUF is
# deleted once both of its logs exist (types and logs stay). Resumes.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"; M="$ROOT/models"; L="$ROOT/logs"; K="$M/logits"; R="$ROOT/results/pilot"; E="$M/ext"
PY="$ROOT/.venv/bin/python -I"
. "$ROOT/scripts/gpu-lock.sh"
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
ctx() { if [ "$1" = T2 ]; then echo 2048; else echo 4096; fi; }
while read -r name repo rev file; do
    if done_log "$L/ppl-final-$name-T1-P3.log" && done_log "$L/ppl-final-$name-T2-P3.log"; then continue; fi
    if [ ! -e "$E/$name.gguf" ]; then
        step "fetch $name ($repo@$rev)"
        HF_HUB_DISABLE_XET=1 hf download "$repo" "$file" --revision "$rev" --local-dir "$E/$repo" > /dev/null
        ln -sf "$E/$repo/$file" "$E/$name.gguf"; echo "$repo sha=$rev $file" >> "$L/revisions.txt"
    fi
    [ -f "$R/types-$name.txt" ] || $PY "$ROOT/scripts/gguf-types.py" "$E/$name.gguf" > "$R/types-$name.txt"
    for t in T1 T2; do
        log="$L/ppl-final-$name-$t-P3.log"; done_log "$log" && continue
        step "KL $name on $t"
        gpu_take
        "$BIN/llama-perplexity" -m "$E/$name.gguf" -f "$ROOT/data/$t.txt" -c "$(ctx "$t")" -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
            --kl-divergence-base "$K/ref-$t-P3.kld" --kl-divergence > "$log" 2>&1
        gpu_give
        grep -E 'Mean +KLD' "$log" || true
    done
    done_log "$L/ppl-final-$name-T1-P3.log" && done_log "$L/ppl-final-$name-T2-P3.log" && rm -f "$(readlink "$E/$name.gguf")" && step "scored $name, GGUF deleted"
done <<'LIST'
arelius-Q2K-Q2_0 arelius/Inkling-2-Mini-Q2-3090-GGUF d1679c7 Inkling-2-Mini-Q2K-Q2_0.gguf
nikolaus-Q2_K nikolaus714/Inkling-2-Mini-GGUF 8375982 Inkling-2-Mini-Q2_K.gguf
devquasar-Q2_K DevQuasar/thinkingmachines.Inkling-2-Mini-GGUF fd1ba3a Q2_K/thinkingmachines.Inkling-2-Mini.f16.gguf.Q2_K.gguf
LIST
step "ext11 done"
