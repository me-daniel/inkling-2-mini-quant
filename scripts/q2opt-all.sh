#!/bin/sh
# Encodes every layer's experts with scripts/q2opt.py, one donor file per layer (bounded memory). Resumes.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
mkdir -p "$ROOT/models/q2opt" "$ROOT/results/pilot"
for l in $(seq 0 23); do
    out="$ROOT/models/q2opt/L$l.gguf"
    [ -f "$out" ] && continue
    nice -n 10 "$ROOT/.venv/bin/python" -I "$ROOT/scripts/q2opt.py" "$ROOT/models/Inkling-2-Mini-BF16.gguf" "$ROOT/models/ggml-org/imatrix.gguf" \
        "$out.part" --layers "$l-$l" --kinds ffn_down_exps ffn_gate_up_exps 2>> "$ROOT/logs/q2opt.log"
    mv "$out.part" "$out"
    mv "$out.part.tsv" "$out.tsv"
done
{ printf 'layer\tkind\tt_stock\tt_opt\ttw_stock\ttw_opt\n'; for l in $(seq 0 23); do tail -n +2 "$ROOT/models/q2opt/L$l.gguf.tsv"; done; } > "$ROOT/results/pilot/q2opt-errors.tsv"
echo done
