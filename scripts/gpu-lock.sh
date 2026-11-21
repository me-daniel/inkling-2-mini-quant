# Sourced by the run scripts: one heavy GPU run at a time on this Mac, shared with the sibling repo.
# Lock content "<pid> <owner> <time>"; created atomically; a lock whose pid is dead is stale; only the pid that took
# the lock releases it (a dry run or another process of this project cannot drop a real run's lock).
GLOCK="$HOME/GitHub/.gpu.lock"
gpu_take() {
    while :; do
        if ( set -C; echo "$$ inkling-2-mini-quant $(date '+%H:%M:%S')" > "$GLOCK" ) 2>/dev/null; then return 0; fi
        pid=$(cut -d' ' -f1 "$GLOCK" 2>/dev/null)
        [ "$pid" = "$$" ] && return 0
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then rm -f "$GLOCK"; continue; fi
        sleep 30
    done
}
gpu_give() { [ "$(cut -d' ' -f1 "$GLOCK" 2>/dev/null)" = "$$" ] && rm -f "$GLOCK"; return 0; }
trap gpu_give EXIT
# CPU- or IO-heavy steps outside the GPU lock (splices, quantize) wait for at least 30% free memory, as the sibling does
mem_wait() { while :; do f=$(memory_pressure 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'); [ "${f:-100}" -ge 30 ] && return 0; sleep 30; done; }
