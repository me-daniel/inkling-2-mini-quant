# Sourced by the run scripts: one heavy GPU run and one heavy CPU job at a time on this Mac, shared with the sibling repo.
#   GPU: ~/GitHub/.gpu.lock around llama-perplexity / llama-imatrix runs (gpu_take / gpu_give)
#   CPU: ~/GitHub/.cpu.lock around splices, llama-quantize and heavy CPU analyses (cpu_take / cpu_give); cpu_take
#        then waits for at least 30% free memory. mem_wait alone checks only once, so two jobs started seconds apart
#        both passed it (22 Nov, 05:20, swap full).
#   gpu_try / cpu_try take a lock only if it is free right now (1 otherwise), for jobs that need both at once.
#   Small jobs (under 2 GB RSS and under 2 minutes) may run beside a held CPU lock while memory_pressure shows at
#   least 30% free; anything bigger takes the lock (agreed with the sibling repo, 22 Nov 09:00).
# Kernel locks since 23 Nov (agreed with the sibling repo): a background `lockf -k FILE` holds an flock on FILE and
# watches this script's pid; the kernel drops the lock when the holder ends, so a dead job leaves no stale lock and
# nothing is ever removed by hand. The file stays on disk and only names the last holder ("<holder pid> <owner>
# <time>"); its existence means nothing. The old pid files lost a lock to a stale-lock rm racing a new set -C.
GLOCK="$HOME/GitHub/.gpu.lock"
CLOCK="$HOME/GitHub/.cpu.lock"
_take() { # lockfile [lockf options]: start a holder and wait until it holds the lock; 1 if the holder gave up
    _lk_v=_lk_holder_$(echo "$1" | tr -c 'a-zA-Z0-9' _)
    eval "_lk_h=\${$_lk_v:-}"
    [ -n "$_lk_h" ] && kill -0 "$_lk_h" 2>/dev/null && return 0  # already ours (a caller holding it for a window)
    _lk_f=$1; shift
    /usr/bin/lockf -k "$@" "$_lk_f" sh -c 'echo "$PPID $2 $(date +%H:%M:%S)" > "$3"; while kill -0 "$1" 2>/dev/null; do sleep 2; done' \
        _ "$$" inkling-2-mini-quant "$_lk_f" 2>/dev/null &
    _lk_h=$!
    while [ "$(cut -d' ' -f1 "$_lk_f" 2>/dev/null)" != "$_lk_h" ]; do kill -0 "$_lk_h" 2>/dev/null || return 1; sleep 1; done
    eval "$_lk_v=$_lk_h"
}
_give() { # lockfile: end our holder (its watcher child first, so lockf exits and the kernel drops the lock)
    _lk_v=_lk_holder_$(echo "$1" | tr -c 'a-zA-Z0-9' _)
    eval "_lk_h=\${$_lk_v:-}"
    [ -n "$_lk_h" ] || return 0
    { pkill -P "$_lk_h"; kill "$_lk_h"; wait "$_lk_h"; } 2>/dev/null || :  # || : so set -e callers survive
    eval "$_lk_v="; return 0
}
# sh has no local variables: helper variables carry a _lk_ prefix so they never overwrite a caller's (a caller's $f
# once became the free-memory percentage, 22 Nov 11:00).
mem_wait() { while :; do _lk_free=$(memory_pressure 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'); [ "${_lk_free:-100}" -ge 30 ] && return 0; sleep 30; done; }
gpu_take() { _take "$GLOCK"; }
gpu_give() { _give "$GLOCK"; }
cpu_take() { _take "$CLOCK"; mem_wait; }
gpu_try() { _take "$GLOCK" -t 0; }
cpu_try() { _take "$CLOCK" -t 0; }
cpu_give() { _give "$CLOCK"; }
trap 'gpu_give; cpu_give' EXIT
