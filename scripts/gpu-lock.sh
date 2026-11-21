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
