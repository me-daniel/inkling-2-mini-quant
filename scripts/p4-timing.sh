#!/bin/sh
# Stop-rule recheck (Daniel, 21 Nov evening): runtime of the 3-bit pool on S at P = 2, 3, 4, back to back, no KL
# (the P = 2 and P = 4 reference logits were deleted for disk), against layer visits 36 / 48 / 60. Writes results/pilot/p4-timing.txt.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
BIN="$ROOT/vendor/llama.cpp/build/bin"
out="$ROOT/results/pilot/p4-timing.txt"
printf 'P\tseconds\tvisits\n' > "$out.part"
for p in 2 3 4; do
    t0=$(date +%s)
    "$BIN/llama-perplexity" -m "$ROOT/models/pool-3bit.gguf" -f "$ROOT/data/S.txt" -c 4096 --chunks 16 -ngl 99 \
        --override-kv "inkling2_loop.loop.count=int:$p" > "$ROOT/logs/ppl-timing-3bit-S-P$p.log" 2>&1
    printf '%s\t%s\t%s\n' "$p" $(( $(date +%s) - t0 )) $(( 12 + 12 * p )) >> "$out.part"
done
mv "$out.part" "$out"
awk -F'\t' 'NR==2{} NR>1{s[$1]=$2; v[$1]=$3} END{for(p=2;p<=4;p++) printf "P = %d: %d s, ratio to P = 3 %.2f, visits ratio %.2f\n", p, s[p], s[p]/s[3], v[p]/v[3]}' "$out" >> "$out"
cat "$out"
