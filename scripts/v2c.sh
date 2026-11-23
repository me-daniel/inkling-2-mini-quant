#!/bin/sh
# Addendum H amendment 16: v2c = v2m's recipe (v2a with refined gate_up in the same 22 layers) on Grams from
#   data/calib-mixc.txt (scripts/calib-mix.py --code: v2m's three sources plus 16 repositories of Python and Rust code,
#   scripts/a16-data.py; first 400 chunks of 512 tokens, so the three old sources keep about v2m's share and code adds
#   about a quarter). Derived from scripts/v2m.sh by renaming only. Groups of up to 4 layers: capture in the background
#   (scripts/gram-gu.sh, GPU lock) while the previous group is refined (CPU lock, 4 workers); each group's Gram .f32
#   files are deleted after its checks. Then splice, checks, sha256, S, T1 and T2 (all seen); A2 by scripts/a16-score.sh.
# Resumable (every step skips what is done). DRY=1 prints the plan and runs only the read-only preflight.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; R=models/refined; GR=models/grams; P=results/pilot
PY=".venv/bin/python -I"
DRY=${DRY:-0}
IQ3="9 10 11 13 14 15 16 17 19 20 21 22 23"
IQ2S="1 2 3 5 6 7 8 12 18"
. scripts/gpu-lock.sh

# ---- helpers ---------------------------------------------------------------------------------------------------
step() { echo "[$(date '+%H:%M:%S')] $*"; }
tag() { echo "$1" | tr ' ' '-'; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
est_min() { # layers -> expected refine minutes (pilot: ~31 min per IQ3_XXS layer; IQ2_S ~25 with the sign search)
    n=0
    for l in $1; do case " $IQ3 " in *" $l "*) n=$((n + 31)) ;; *) n=$((n + 25)) ;; esac; done
    echo "$n"
}
py_retry() { # log cmd... : rerun only when the script stopped on its own memory check at start
    log=$1; shift; n=0
    while :; do
        if "$@" > "$log" 2>&1; then return 0; fi
        if grep -q 'STOP: memory_pressure' "$log" && [ "$n" -lt 30 ]; then
            n=$((n + 1)); step "memory low, retry $n in 60 s ($log)"; sleep 60; mem_wait; continue
        fi
        step "FAIL: see $log"; tail -5 "$log"; return 1
    done
}
CAP=""
capture_bg() { # "layers": start the Gram capture in the background unless done
    t=$(tag "$1"); CAP=""
    if [ -f "$GR/gu-$t-mixc/done" ]; then step "grams $1: already captured"; return 0; fi
    step "capture grams $1 in the background (gram-gu.sh takes the GPU lock; ~30 min once it has it)"
    if [ "$DRY" = 1 ]; then echo "  LAYERS=\"$1\" CALIB=calib-mixc.txt CHUNKS=400 GSUF=-mixc sh scripts/gram-gu.sh > $L/gram-gu-$t-mixc.out 2>&1 &"; return 0; fi
    LAYERS="$1" CALIB=calib-mixc.txt CHUNKS=400 GSUF=-mixc sh scripts/gram-gu.sh > "$L/gram-gu-$t-mixc.out" 2>&1 &
    CAP=$!
}
capture_wait() {
    [ -n "$CAP" ] || return 0
    step "wait for capture (pid $CAP)"
    wait "$CAP"
    CAP=""
    step "capture done"
}
refine_group() { # "layers" gramdir delete(0/1)
    layers=$1; gd=$2; del=$3
    t=$(tag "$layers"); out="$R/gu-refined-mixc-$t.gguf"
    if [ -f "$out" ] && [ -f "$out.json" ]; then
        step "refined $layers: $out exists"
    else
        step "refine $layers on $gd (CPU lock, 4 workers; expected ~$(est_min "$layers") min)"
        if [ "$DRY" = 1 ]; then
            echo "  cpu_take; refine-codes.py --grams $gd --layers $layers --sweeps 4 --tol 1e-4 --shrink 0.1 --min-tokens 2304 --workers 4 --out $out > $L/refine-codes-v2c-$t.log; cpu_give"
        else
            [ -f "$gd/done" ] || { step "STOP: $gd/done missing"; exit 1; }
            cpu_take
            py_retry "$L/refine-codes-v2c-$t.log" $PY scripts/refine-codes.py --quant $M/v2a.gguf \
                --bf16 $M/Inkling-2-Mini-BF16.gguf --grams "$gd" --layers $layers --sweeps 4 --tol 1e-4 \
                --shrink 0.1 --min-tokens 2304 --workers 4 --out "$out"
            cpu_give
        fi
    fi
    if [ -f "$P/v2c-offline-$t.md" ] && grep -q '^OK ' "$L/v2c-check-$t.txt" 2>/dev/null; then
        step "check and offline table $layers: done"
    else
        step "check $out, offline table $P/v2c-offline-$t.md (CPU lock, ~$(( $(echo $layers | wc -w) * 2 + 1 )) min)"
        if [ "$DRY" = 1 ]; then
            echo "  cpu_take; v2r-check.py group $out $M/v2a.gguf $gd $layers > $L/v2c-check-$t.txt"
            echo "  gram-error.py --grams $gd --layers $layers --reg 0.1 --extra refined=$out --tsv $P/v2c-offline-$t.tsv --md $P/v2c-offline-$t.md; cpu_give"
        else
            cpu_take
            py_retry "$L/v2c-check-$t.txt" $PY scripts/v2r-check.py group "$out" $M/v2a.gguf "$gd" $layers
            py_retry "$L/v2c-offline-$t.log" $PY scripts/gram-error.py --grams "$gd" --layers $layers --reg 0.1 \
                --extra "refined=$out" --tsv "$P/v2c-offline-$t.tsv" --md "$P/v2c-offline-$t.md"
            cpu_give
            cat "$L/v2c-check-$t.txt"
        fi
    fi
    if [ "$del" = 1 ]; then
        step "delete Gram .f32 of $layers in $gd"
        for l in $layers; do
            f="$gd/blk.$l.ffn_gate_up_exps.weight.gram.f32"
            if [ "$DRY" = 1 ]; then echo "  rm -f $f"; else rm -f "$f"; fi
        done
    fi
}

# ---- preflight (read-only, also in DRY) --------------------------------------------------------------------------
if [ "$DRY" != 1 ]; then
    others=$(pgrep -f '^sh scripts/v2[rmc]\.sh|scripts/gram-gu\.sh|scripts/refine-codes\.py' | grep -vx "$$" || true)
    [ -z "$others" ] || { step "STOP: already running: $(echo $others)"; exit 1; }
fi
step "preflight: v2a types, data/calib-mixc.txt hash"
[ "$(shasum -a 256 data/calib-mixc.txt | cut -d' ' -f1)" = bafd8cade55dab290b96e89532ed4e4d237f8b352f37c92e496fe2cb8e3cce22 ] || { step "STOP: calib-mixc.txt hash"; exit 1; }
$PY - "$IQ3" "$IQ2S" <<'EOF'
import json, sys
import gguf
iq3, iq2s = [int(x) for x in sys.argv[1].split()], [int(x) for x in sys.argv[2].split()]
t = {x.name: x.tensor_type.name for x in gguf.GGUFReader("models/v2a.gguf").tensors}
want = {**{l: "IQ3_XXS" for l in iq3}, **{l: "IQ2_S" for l in iq2s}, 0: "IQ2_XXS", 4: "IQ2_XS"}
got = {l: t[f"blk.{l}.ffn_gate_up_exps.weight"] for l in range(24)}
assert got == want, ("v2a gate_up types", got)
print("preflight OK: v2a gate_up types as amendment 12")
EOF
mkdir -p "$R" "$M/tmp" "$P"

# ---- Grams and refinement: capture group i+1 while group i is refined -------------------------------------------
C1="1 2 3 5"; C2="6 7 8 9"; C3="10 11 12 13"; C4="14 15 16 17"; C5="18 19 20 21"; C6="22 23"
capture_bg "$C1"
capture_wait
prev=""
for g in "$C2" "$C3" "$C4" "$C5" "$C6" ""; do
    [ -n "$g" ] && capture_bg "$g"
    cur=${prev:-$C1}
    refine_group "$cur" "$GR/gu-$(tag "$cur")-mixc" 1
    capture_wait
    prev=$g
done

# ---- assemble v2c --------------------------------------------------------------------------------------------------
TAKES=""
add_take() { # file layers...
    f=$1; shift
    for l in "$@"; do TAKES="$TAKES --take $f ^blk\\.$l\\.ffn_gate_up_exps\\.weight\$"; done
}
for g in "$C1" "$C2" "$C3" "$C4" "$C5" "$C6"; do add_take "$R/gu-refined-mixc-$(tag "$g").gguf" $g; done
ALL="$IQ3 $IQ2S"
if [ -f "$M/v2c.gguf" ] && grep -q '^OK gate_up differs' "$L/v2c-check-final.txt" 2>/dev/null; then
    step "v2c: $M/v2c.gguf exists and passed its checks"
else
    step "splice v2c (CPU lock, ~5 min): base v2a, $(echo $ALL | wc -w | tr -d ' ') refined gate_up tensors"
    set -f
    if [ "$DRY" = 1 ]; then
        echo "  splice.py $M/v2a.gguf $M/tmp/v2c.part.gguf$TAKES --verify > $L/splice-v2c.json"
        echo "  gguf-types.py -> $L/types-v2c.txt; v2r-check.py final ... > $L/v2c-check-final.txt; mv -> $M/v2c.gguf"
    else
        for f in $(echo "$TAKES" | tr ' ' '\n' | grep '\.gguf$' | sort -u); do
            [ -f "$f" ] || { step "STOP: $f missing"; exit 1; }
        done
        cpu_take
        $PY scripts/splice.py $M/v2a.gguf $M/tmp/v2c.part.gguf $TAKES --verify > $L/splice-v2c.json
        $PY scripts/gguf-types.py $M/tmp/v2c.part.gguf > $L/types-v2c.txt
        $PY scripts/gguf-types.py $M/v2a.gguf > $L/types-v2a.txt
        $PY scripts/v2r-check.py final $M/tmp/v2c.part.gguf $M/v2a.gguf $ALL > $L/v2c-check-final.txt
        cpu_give
        cat $L/v2c-check-final.txt
        diff $L/types-v2a.txt $L/types-v2c.txt || { step "STOP: gguf-types of v2c differ from v2a"; exit 1; }
        step "gguf-types: v2c identical to v2a"
        mv $M/tmp/v2c.part.gguf $M/v2c.gguf
    fi
    set +f
fi
if [ "$DRY" = 1 ]; then echo "  shasum -a 256 $M/v2c.gguf > $P/v2c-sha256.txt"
elif [ ! -s "$P/v2c-sha256.txt" ]; then shasum -a 256 "$M/v2c.gguf" > "$P/v2c-sha256.txt"; fi
[ "$DRY" = 1 ] || cat "$P/v2c-sha256.txt"

# ---- score on S at P = 3 ---------------------------------------------------------------------------------------------
log="$L/ppl-v2c-S-P3.log"
if done_log "$log"; then
    step "v2c S score exists"
else
    step "perplexity v2c on S (P=3, GPU lock, ~15 min)"
    if [ "$DRY" = 1 ]; then
        echo "  gpu_take; llama-perplexity -m $M/v2c.gguf -f data/S.txt -c 4096 --chunks 16 -ngl 99 --override-kv inkling2_loop.loop.count=int:3 --kl-divergence-base $M/logits/ref-S-P3.kld --kl-divergence > $log; gpu_give"
    else
        mem_wait; gpu_take
        "$BIN/llama-perplexity" -m "$M/v2c.gguf" -f data/S.txt -c 4096 --chunks 16 -ngl 99 \
            --override-kv "inkling2_loop.loop.count=int:3" --kl-divergence-base "$M/logits/ref-S-P3.kld" \
            --kl-divergence > "$log" 2>&1
        gpu_give
    fi
fi
[ "$DRY" = 1 ] || grep -E 'Mean +KLD' "$log"
step "v2c done"
N=v2c sh scripts/v2r-t12.sh
step "v2c T1/T2 done"
