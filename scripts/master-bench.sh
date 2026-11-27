#!/bin/sh
# Speed of v2c and Unsloth UD-IQ3_XXS on unpatched llama.cpp master 12d47ba2f (Metal NVFP4 kernels, #32825).
# Same protocol as the sibling's scripts/master-nvfp4.sh. Checks first that no UD expert tensor stays on the CPU.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd); . "$ROOT/scripts/gpu-lock.sh"
BIN="$ROOT/../inkling-2-mini-under10/vendor/llama.cpp-master/build/bin"; R="$ROOT/results/master"; L="$ROOT/logs"; mkdir -p "$R"
git -C "$BIN/../.." log -1 --format='%h %cs' > "$R/commit.txt"
gpu_take
trap gpu_give EXIT
"$BIN/llama-perplexity" -m "$ROOT/models/UD-IQ3_XXS.gguf" -f "$ROOT/data/S.txt" -c 512 --chunks 1 -ngl 99 \
  --override-kv "inkling2_loop.loop.count=int:3" -v > "$L/master-splits-UD3.log" 2>&1
for n in v2c UD-IQ3_XXS; do
  "$BIN/llama-bench" -m "$ROOT/models/$n.gguf" -ngl 99 -p 512 -n 128 -r 3 -o md > "$R/bench-$n.txt.part" 2> "$L/master-bench-$n.log"
  mv "$R/bench-$n.txt.part" "$R/bench-$n.txt"
done
