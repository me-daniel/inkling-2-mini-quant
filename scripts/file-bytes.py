"""Writes results/pilot/file-bytes.tsv: the exact byte size of every compared GGUF, so cards and figures do not need
the (large, partly deleted) files. Local files are measured with stat; outside files are read from the Hugging Face
API at the revision pinned in logs/revisions.txt. Existing rows are kept.

    python scripts/file-bytes.py
"""

import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/pilot/file-bytes.tsv"
LOCAL = ["cand", "UD-IQ3_XXS", "uniform-IQ3_XXS"]
EXT = {"bart-IQ2_S": "thinkingmachines_Inkling-2-Mini-IQ2_S.gguf", "bart-IQ2_M": "thinkingmachines_Inkling-2-Mini-IQ2_M.gguf",
       "atomic-AD-IQ2_S": "Inkling-2-Mini-AD-IQ2_S.gguf", "apex-I-Nano": "Inkling-2-Mini-APEX-I-Nano.gguf",
       "isvalorum-NanoPlus": "Inkling-2-Mini-APEX-I-NanoPlus-V2.1.gguf", "prithiv-Q2_K": "Inkling-2-Mini.Q2_K.gguf"}


def main() -> None:
    rows = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines()[1:]:
            k, b, src = line.split("\t")
            rows[k] = (int(b), src)
    for k in LOCAL:
        p = ROOT / f"models/{k}.gguf"
        if p.exists():
            rows[k] = (p.resolve().stat().st_size, "stat")
    if "uniform-IQ3_XXS" not in rows:  # deleted for disk; gate.md recorded its bytes while it existed
        m = re.search(r"^\| stock IQ3_XXS[^|]*\| ([\d,]+) \|", (ROOT / "results/pilot/gate.md").read_text(), re.M)
        rows["uniform-IQ3_XXS"] = (int(m.group(1).replace(",", "")), "gate.md")
    revs = {m[3]: (m[1], m[2]) for m in re.finditer(r"^(\S+/\S+) sha=([0-9a-f]+) (\S+\.gguf)$", (ROOT / "logs/revisions.txt").read_text(), re.M)}
    for k, fname in EXT.items():
        if k in rows or fname not in revs:
            continue
        repo, sha = revs[fname]
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{sha}", timeout=30) as r:
            size = next(f["size"] for f in json.load(r) if f["path"] == fname)
        rows[k] = (size, f"hf:{repo}@{sha[:7]}")
    OUT.write_text("name\tbytes\tsource\n" + "".join(f"{k}\t{b}\t{s}\n" for k, (b, s) in sorted(rows.items())))
    print(OUT.read_text())


if __name__ == "__main__":
    main()
