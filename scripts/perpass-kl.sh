#!/bin/sh
# KL per chunk of every run-4 condition against the IQ4_NL base and the Q8_0 reference (results/pilot/perpass-kl.tsv).
# noop and allslots logits are deleted once found byte-identical to base and all (perpass-parity.txt); their rows say so.
# Other logits deleted for disk after scoring keep their existing rows.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
K="$ROOT/models/logits"
PY="$ROOT/.venv/bin/python -I"
{
    printf 'cond\tagainst\tmean_kl\tper_chunk\n'
    for c in base noop p1 p2 p3 all allslots noise-tied noise-tied2 noise-tied3 noise-indep; do
        for ref in pp-base ref-S-P3; do
            if [ ! -f "$K/pp-$c.kld" ]; then
                # logits deleted for disk: keep the row already scored, else mark the identical twin
                old=$(grep "^$c	$ref	" "$ROOT/results/pilot/perpass-kl.tsv" 2>/dev/null || true)
                if [ -n "$old" ]; then echo "$old"; else printf '%s\t%s\tdeleted\tidentical twin, see perpass-parity.txt\n' "$c" "$ref"; fi
                continue
            fi
            out=$($PY "$ROOT/scripts/kldfile.py" "$K/$ref.kld" "$K/pp-$c.kld")
            printf '%s\t%s\t%s\t%s\n' "$c" "$ref" "$(echo "$out" | tail -1 | cut -d' ' -f3)" "$(echo "$out" | head -1 | tr ' ' ',')"
        done
    done
} > "$ROOT/results/pilot/perpass-kl.tsv.part"
mv "$ROOT/results/pilot/perpass-kl.tsv.part" "$ROOT/results/pilot/perpass-kl.tsv"
cut -f1-3 "$ROOT/results/pilot/perpass-kl.tsv"
