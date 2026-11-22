# Sourced by the run scripts: one heavy GPU run and one heavy CPU job at a time on this Mac, shared with the sibling repo.
# Lock content "<pid> <owner> <time>"; created atomically; a lock whose pid is dead is stale; only the pid that took
# the lock releases it (a dry run or another process of this project cannot drop a real run's lock).
#   GPU: ~/GitHub/.gpu.lock around llama-perplexity / llama-imatrix runs (gpu_take / gpu_give)
#   CPU: ~/GitHub/.cpu.lock around splices, llama-quantize and heavy CPU analyses (cpu_take / cpu_give); cpu_take
#        then waits for at least 30% free memory. mem_wait alone checks only once, so two jobs started seconds apart
#        both passed it (22 Nov, 05:20, swap full).
#   Small jobs (under 2 GB RSS and under 2 minutes) may run beside a held CPU lock while memory_pressure shows at
#   least 30% free; anything bigger takes the lock (agreed with the sibling repo, 22 Nov 09:00).
GLOCK="$HOME/GitHub/.gpu.lock"
CLOCK="$HOME/GitHub/.cpu.lock"
_take() { # lockfile
    while :; do
        if ( set -C; echo "$$ inkling-2-mini-quant $(date '+%H:%M:%S')" > "$1" ) 2>/dev/null; then return 0; fi
        _lk_pid=$(cut -d' ' -f1 "$1" 2>/dev/null)
        [ "$_lk_pid" = "$$" ] && return 0
        if [ -n "$_lk_pid" ] && ! kill -0 "$_lk_pid" 2>/dev/null; then rm -f "$1"; continue; fi
        sleep 30
    done
}
_give() { [ "$(cut -d' ' -f1 "$1" 2>/dev/null)" = "$$" ] && rm -f "$1"; return 0; }
# sh has no local variables: helper variables carry a _lk_ prefix so they never overwrite a caller's (a caller's $f
# once became the free-memory percentage, 22 Nov 11:00).
mem_wait() { while :; do _lk_free=$(memory_pressure 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'); [ "${_lk_free:-100}" -ge 30 ] && return 0; sleep 30; done; }
gpu_take() { _take "$GLOCK"; }
gpu_give() { _give "$GLOCK"; }
cpu_take() { _take "$CLOCK"; mem_wait; }
cpu_give() { _give "$CLOCK"; }
trap 'gpu_give; cpu_give' EXIT
