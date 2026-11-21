#!/bin/sh
# Waits up to SECONDS (default 540) for a shell condition, checking every 20 s. Exit 0: condition true; 3: timed out.
#   sh scripts/wait-for.sh '! pgrep -f scripts/pilot-chain4.sh' 540
cond=$1; limit=${2:-540}; t=0
cd "$(dirname "$0")/.." || exit 1
while ! sh -c "$cond" > /dev/null 2>&1; do
    [ "$t" -ge "$limit" ] && { echo "still waiting ($(date '+%H:%M')); last log lines:"; tail -n 2 logs/chain*.out 2>/dev/null | tail -n 6; exit 3; }
    sleep 20; t=$((t + 20))
done
echo "condition met ($(date '+%H:%M'))"
