#!/bin/sh
# Addendum G: downloads the outside comparators into models/ext/NAME.gguf and records each repo revision. Resumes.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
E="$ROOT/models/ext"; mkdir -p "$E"
export HF_HUB_DISABLE_XET=1
while read -r name repo file; do
    [ -f "$E/$name.gguf" ] && continue
    sha=$(curl -s "https://huggingface.co/api/models/$repo" | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])")
    echo "[$(date '+%H:%M:%S')] fetch $name ($repo@$sha)"
    hf download "$repo" "$file" --revision "$sha" --local-dir "$E/$repo" > /dev/null
    ln -sf "$E/$repo/$file" "$E/$name.gguf"
    echo "$repo sha=$sha $file" >> "$ROOT/logs/revisions.txt"
done <<LIST
bart-IQ2_S bartowski/thinkingmachines_Inkling-2-Mini-GGUF thinkingmachines_Inkling-2-Mini-IQ2_S.gguf
atomic-AD-IQ2_S AtomicChat/Inkling-2-Mini-GGUF Inkling-2-Mini-AD-IQ2_S.gguf
apex-I-Nano mudler/Inkling-2-Mini-APEX-GGUF Inkling-2-Mini-APEX-I-Nano.gguf
bart-IQ2_M bartowski/thinkingmachines_Inkling-2-Mini-GGUF thinkingmachines_Inkling-2-Mini-IQ2_M.gguf
LIST
echo "[$(date '+%H:%M:%S')] fetch done"
