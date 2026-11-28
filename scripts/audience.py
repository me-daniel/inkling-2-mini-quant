"""Audience check for amendment 23 (28 Nov 2026), from saved Hugging Face API replies in results/a23/audience/.

    python -I scripts/audience.py

Downloads are Hugging Face's per-repo counter (it counts GGUF file downloads; no per-file numbers are public), so the
size-tier split inside a repo is not observable. Prints the top repos and, for selected repos, the GGUF sizes offered.
"""

import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "results/a23/audience"
models = [m for m in json.loads((D / "hf-models-2026-11-28.json").read_text()) if "gguf" in m.get("tags", []) or "GGUF" in m["id"]]
models.sort(key=lambda m: -m.get("downloads", 0))
total = sum(m.get("downloads", 0) for m in models)
print(f"{len(models)} Inkling-2-Mini GGUF repos, {total:,} downloads in all\n")
print("| rank | repo | downloads | share | likes |\n|---|---|---|---|---|")
for i, m in enumerate(models):
    if i < 10 or m["id"].startswith(("ISTA-DASLab/", "Me-Daniel/")):
        print(f"| {i + 1} | {m['id']} | {m.get('downloads', 0):,} | {m.get('downloads', 0) / total:.1%} | {m.get('likes', 0)} |")
print("\n| repo | GGUF sizes offered (GB, language model only) |\n|---|---|")
for f in sorted(D.glob("tree-*.json")):
    sizes = sorted(x.get("lfs", {}).get("size", 0) / 1e9 for x in json.loads(f.read_text())
                   if x["path"].endswith((".gguf", ".gguf_file")) and not any(k in x["path"] for k in ("mmproj", "dspark", "LoRA", "imatrix", "Old_Support", "Mainline_Support", "BF16/")))
    print(f"| {f.stem[5:].replace('_', '/', 1)} | {', '.join(f'{s:.1f}' for s in sizes)} |")
