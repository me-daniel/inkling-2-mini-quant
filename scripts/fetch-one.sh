#!/bin/sh
# Downloads one outside comparator into models/ext/NAME.gguf at the repo's current revision and records it in
# logs/revisions.txt (like scripts/fetch-ext.sh, for files registered later). Resumes.
#   sh scripts/fetch-one.sh NAME REPO FILE
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
E="$ROOT/models/ext"; mkdir -p "$E"
name=$1 repo=$2 file=$3
[ -f "$E/$name.gguf" ] && exit 0
export HF_HUB_DISABLE_XET=1
sha=$(curl -s "https://huggingface.co/api/models/$repo" | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])")
echo "[$(date '+%H:%M:%S')] fetch $name ($repo@$sha)"
hf download "$repo" "$file" --revision "$sha" --local-dir "$E/$repo" > /dev/null
ln -sf "$E/$repo/$file" "$E/$name.gguf"
echo "$repo sha=$sha $file" >> "$ROOT/logs/revisions.txt"
echo "[$(date '+%H:%M:%S')] fetch $name done"
