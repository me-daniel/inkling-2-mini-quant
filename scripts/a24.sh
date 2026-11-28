#!/bin/sh
# Addendum H amendment 24 (B): loop-aware gate_up exchange on v2c-A at fixed bytes. Waits for scripts/dv-score.sh,
# writes the DV baselines, rebuilds BF16, quantizes starting codes (v2a's command, B's gate_up types), captures Grams
# and refines with v2c's settings, scores U, Uc and d19..d23 on DV, selects D4 (analysis/a24.py select), builds and
# scores X on DV, S, T1, T2 (and v2c-A on T1, T2), then reports into results/a24/b.md. Resumes; every step skips what
# is done.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
BIN=vendor/llama.cpp/build/bin; M=models; L=logs; K=$M/logits; R=$M/refined; GR=$M/grams; PY=".venv/bin/python -I"
BF=$M/Inkling-2-Mini-BF16.gguf; INIT=$M/tmp/a24-init.gguf
. scripts/gpu-lock.sh
step() { echo "[$(date '+%H:%M:%S')] $*"; }
done_log() { [ -f "$1" ] && grep -q 'Mean *KLD' "$1"; }
tag() { echo "$1" | tr ' ' '-'; }
free_gb() { df -g "$M" | awk 'NR==2{print $4}'; }
G1="6 7 8 12"; G2="1 2 3 5"; G3="19 20 21 22"; G4="23"
rfile() { echo "$R/a24-$(tag "$1").gguf"; }

# ---- 0: after the DV baselines ------------------------------------------------------------------------------------
while pgrep -f 'scripts/dv-score\.sh' > /dev/null; do sleep 60; done
mkdir -p results/a23 results/a24 $R $M/tmp
[ -s results/a23/dv-baselines.md ] || $PY analysis/a23-dv.py > results/a23/dv-baselines.md
cat results/a23/dv-baselines.md

# ---- 1: starting codes ----------------------------------------------------------------------------------------------
all_refined() { for g in "$G1" "$G2" "$G3" "$G4"; do [ -f "$(rfile "$g").json" ] || return 1; done; }
if ! all_refined; then
    if [ ! -f $BF ]; then
        [ "$(free_gb)" -ge 90 ] || { step "STOP: under 90 GB free for the BF16 rebuild"; exit 1; }
        step "convert to BF16 (CPU lock)"; cpu_take
        .venv/bin/python vendor/llama.cpp/convert_hf_to_gguf.py $M/Inkling-2-Mini --outtype bf16 --outfile $BF.part > $L/convert-a24.log 2>&1
        cpu_give; mv $BF.part $BF
    fi
    if [ ! -f $INIT ]; then
        step "starting codes: v2a's command, gate_up 1-3, 5-17 IQ3_XXS and 18-23 IQ2_S (CPU lock)"; cpu_take
        nice -n 5 $BIN/llama-quantize --imatrix $M/ggml-org/imatrix.gguf \
            --tensor-type 'blk\.0\.ffn_gate_up_exps=iq2_xxs' \
            --tensor-type 'blk\.4\.ffn_gate_up_exps=iq2_xs' \
            --tensor-type 'blk\.(18|19|20|21|22|23)\.ffn_gate_up_exps=iq2_s' \
            --tensor-type 'blk\.(1|2|3|5|6|7|8|9|10|11|12|13|14|15|16|17)\.ffn_gate_up_exps=iq3_xxs' \
            --tensor-type 'ffn_down_exps=iq4_nl' \
            --token-embedding-type q4_k --output-tensor-type q5_k \
            $BF $INIT.part Q6_K > $L/quant-a24-init.log 2>&1
        cpu_give; mv $INIT.part $INIT
    fi
    $PY - $INIT <<'EOF'
import sys, gguf
t = {x.name: x.tensor_type.name for x in gguf.GGUFReader(sys.argv[1]).tensors}
want = {**{l: "IQ3_XXS" for l in (1, 2, 3, 5, 6, 7, 8, 12)}, **{l: "IQ2_S" for l in (19, 20, 21, 22, 23)}}
got = {l: t[f"blk.{l}.ffn_gate_up_exps.weight"] for l in want}
assert got == want, got
print("starting-code types OK")
EOF
fi

# ---- 2: Grams and refinement, v2c's settings; capture group i+1 while group i is refined ----------------------------
capture() { LAYERS="$1" CALIB=calib-mixc.txt CHUNKS=400 GSUF=-mixc sh scripts/gram-gu.sh > "$L/gram-gu-$(tag "$1")-mixc.out" 2>&1; }
refine() { # layers
    out=$(rfile "$1"); gd="$GR/gu-$(tag "$1")-mixc"; t=$(tag "$1")
    if [ ! -f "$out.json" ]; then
        [ -f "$gd/done" ] || { step "STOP: $gd/done missing"; exit 1; }
        step "refine $1 (CPU lock, 4 workers)"; cpu_take
        n=0  # refine-codes stops at start when memory is low (the capture of the next group runs beside it)
        until $PY scripts/refine-codes.py --quant $INIT --bf16 $BF --grams "$gd" --layers $1 --sweeps 4 --tol 1e-4 \
            --shrink 0.1 --min-tokens 2304 --workers 4 --out "$out" > "$L/refine-a24-$t.log" 2>&1; do
            grep -q 'STOP: memory_pressure' "$L/refine-a24-$t.log" && [ $n -lt 30 ] || { step "FAIL: $L/refine-a24-$t.log"; exit 1; }
            n=$((n + 1)); step "memory low, retry $n in 60 s"; sleep 60; mem_wait
        done
        $PY scripts/v2r-check.py group "$out" $INIT "$gd" $1 > "$L/a24-check-$t.txt"
        cpu_give; cat "$L/a24-check-$t.txt"
    fi
    rm -rf "$gd"
}
if ! all_refined; then
    prev=""
    for g in "$G1" "$G2" "$G3" "$G4" ""; do
        cap=""
        if [ -n "$g" ] && [ ! -f "$(rfile "$g").json" ]; then
            if [ -z "$prev" ]; then capture "$g"; else capture "$g" & cap=$!; fi
        fi
        [ -n "$prev" ] && refine "$prev"
        [ -n "$cap" ] && wait $cap
        prev=$g
    done
fi
all_refined || { step "STOP: a refined group is missing"; exit 1; }
rm -f $BF $INIT
step "refinement done; BF16 and starting codes deleted"

# ---- 3: files on DV -------------------------------------------------------------------------------------------------
takes() { # layer -> --take for that layer's refined tensor
    for g in "$G1" "$G2" "$G3" "$G4"; do
        case " $g " in *" $1 "*) echo "--take $(rfile "$g") ^blk\\.$1\\.ffn_gate_up_exps\\.weight\$"; return ;; esac
    done
    return 1
}
build() { # name layers...
    n=$1; shift; [ -f $M/$n.gguf ] && return 0
    T=""; for l in "$@"; do x=$(takes $l) || { step "STOP: no refined tensor for layer $l"; exit 1; }; T="$T $x"; done
    step "splice $n (CPU lock)"; cpu_take
    set -f; $PY scripts/splice.py $M/v2c-A.gguf $M/tmp/$n.part.gguf $T --verify > $L/splice-$n.json; set +f
    cpu_give; mv $M/tmp/$n.part.gguf $M/$n.gguf
}
dv_done() { for d in python javascript german english math; do done_log $L/ppl-dv-$d-$1-P3.log || return 1; done; }
dv() { # name layers...: build, score on DV, delete unless KEEP=1
    n=$1; dv_done $n || { build "$@"; NAMES=$n sh scripts/dv-score.sh; }
    [ "${KEEP:-0}" = 1 ] || rm -f $M/$n.gguf
}
dv b-U $G1
dv b-Uc $G2
for l in 19 20 21 22 23; do dv b-d$l $l; done

# ---- 4: X -----------------------------------------------------------------------------------------------------------
D4=$($PY analysis/a24.py select); step "D4 by rule: $D4"
KEEP=1 dv b-X $G1 $D4
[ "$(stat -f %z $M/b-X.gguf)" = "$(stat -f %z $M/v2c-A.gguf)" ] || { step "STOP: b-X bytes differ from v2c-A"; exit 1; }
score() { # name set
    log=$L/ppl-final-$1-$2-P3.log; ch=""; c=4096
    [ $2 = S ] && { log=$L/ppl-$1-S-P3.log; ch="--chunks 16"; }
    [ $2 = T2 ] && c=2048
    done_log "$log" && return 0
    mem_wait; gpu_take
    $BIN/llama-perplexity -m $M/$1.gguf -f data/$2.txt -c $c $ch -ngl 99 --override-kv "inkling2_loop.loop.count=int:3" \
        --kl-divergence-base $K/ref-$2-P3.kld --kl-divergence > "$log" 2>&1
    gpu_give; echo "$1 $2: $(grep -E 'Mean +KLD' "$log")"
}
for s in S T1 T2; do score b-X $s; done
for s in T1 T2; do score v2c-A $s; done
$PY analysis/a24.py > results/a24/b.md
cat results/a24/b.md
grep -q "X becomes the 11.5 GB base" results/a24/b.md || rm -f $M/b-X.gguf
step "a24 done"
