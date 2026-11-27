#!/bin/sh
# Amendment 18: ISTA-DASLab's GSQ-RCO GGUFs of Inkling 2 Mini at a pinned revision, sha256 checked against the
# Hub's LFS oid. Resumes. Writes models/ista/<file> and results/a18/ista-sha256.txt.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P); cd "$ROOT"
REV=f0da1097633dd7c2f5733e4077cff861be122245
mkdir -p models/ista results/a18
HF_HUB_DISABLE_XET=1 .venv/bin/python -I - "$REV" <<'PY'
import sys, hashlib
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
R, rev = "ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF", sys.argv[1]
info = {s.rfilename: s for s in HfApi().model_info(R, revision=rev, files_metadata=True).siblings}
out = []
for t in ("IQ3_S", "IQ3_XXS", "Q2_K_XL", "IQ2_S"):
    f = f"Inkling-2-Mini-GSQ-RCO-{t}.gguf"
    p = Path(hf_hub_download(R, f, revision=rev, local_dir="models/ista"))
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
    ok = h.hexdigest() == info[f].lfs.sha256 and p.stat().st_size == info[f].size
    out.append(f"{h.hexdigest()}  {p.stat().st_size}  {f}  {'OK' if ok else 'MISMATCH'}")
    print(out[-1], flush=True)
    if not ok: sys.exit(1)
Path("results/a18/ista-sha256.txt").write_text(f"# {R} @ {rev}\n" + "\n".join(out) + "\n")
PY
