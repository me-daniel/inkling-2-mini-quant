"""analysis/allocate.py: rerun with ROOT pointed at a temp copy of kl.tsv (nothing in the repo is written), compare
with the stored picks, recompute the predictions by hand, and compare the BUDGET overhead with real files' overhead.
Reads GGUF headers only (memory-mapped).

    .venv/bin/python -I reviews/2026-11-21-code-review-analysis-opus/check_allocate.py
"""

import contextlib
import csv
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
import allocate  # noqa: E402
import budget  # noqa: E402  (scripts/, put on the path by allocate)

tmp = Path(tempfile.mkdtemp())
(tmp / "results/pilot").mkdir(parents=True)
shutil.copy(ROOT / "results/pilot/kl.tsv", tmp / "results/pilot/kl.tsv")
allocate.ROOT = tmp
out = io.StringIO()
with contextlib.redirect_stdout(out):
    allocate.main()
top = out.getvalue().splitlines()
new = json.loads((tmp / "results/pilot/allocation-top.json").read_text())
old = json.loads((ROOT / "results/pilot/allocation-top.json").read_text())
print("== allocator rerun ==")
print("stored allocation-top.json reproduced exactly:", new == old)
spare = [int(l.split("(")[1].split(" MB")[0]) for l in top if l.startswith("predicted")]
print(f"spare MB of the printed top {len(spare)}: {spare}; smallest {min(spare)} MB")
print("gate_up options in code:", list(allocate.GU), "; down options:", list(allocate.DN), "; docstring says '4 gate_up'")

kl = {r["run"]: float(r["mean_kl"]) for r in csv.DictReader(open(ROOT / "results/pilot/kl.tsv"), delimiter="\t")}
lvl = {"IQ3_XXS": "3bit", "IQ2_S": "gu2s", "IQ2_XXS": "gu2xxs"}
print("\n== hand prediction, Addendum E: KL(3bit) + [gu level - 3bit] + [dnq2o - 3bit], plus NE Q6_K ==")
for i, c in enumerate(old):
    pred = (kl["map-NE-Q6K-S-P3"] if c["ne"] == "Q6_K" else 0.0)
    for b, (gu, dn) in c["choice"].items():
        k3 = kl[f"map-{b}-3bit-S-P3"]
        pred += k3 + (kl[f"map-{b}-{lvl[gu]}-S-P3"] - k3) + ((kl[f"map-{b}-dnq2o-S-P3"] - k3) if dn == "Q2_0O" else 0.0)
    print(f"sel-{i}: hand {pred:.6f}, stored {c['predicted_kl']:.6f}, measured on S {kl[f'sel-{i}-S-P3']:.6f}")

print("\n== bytes ==")
for name in ("cand.gguf", "uniform-IQ3_XXS.gguf", "UD-IQ3_XXS.gguf"):
    budget.check([str(ROOT / "models" / name)])
assumed = 11_663_310_784 - allocate.BUDGET
cand_over = (ROOT / "models/cand.gguf").stat().st_size - sum(int(t.n_bytes) for t in budget.gguf.GGUFReader(ROOT / "models/cand.gguf").tensors)
print(f"overhead assumed by BUDGET {assumed:,}; cand.gguf overhead {cand_over:,}; difference {cand_over - assumed:,} bytes")
print(f"allocator bytes of sel-2 {old[2]['bytes']:,}; cand.gguf tensor bytes "
      f"{sum(int(t.n_bytes) for t in budget.gguf.GGUFReader(ROOT / 'models/cand.gguf').tensors):,}")

print("\n== missing map runs: does the allocator notice? ==")
lines = (tmp / "results/pilot/kl.tsv").read_text().splitlines()
drop = ("map-B3-gu2s-S-P3", "map-NE-Q6K-S-P3")
(tmp / "results/pilot/kl.tsv").write_text("\n".join(l for l in lines if not l.startswith(drop)) + "\n")
out = io.StringIO()
try:
    with contextlib.redirect_stdout(out):
        allocate.main()
    picks = json.loads((tmp / "results/pilot/allocation-top.json").read_text())
    print(f"without {drop}: allocator ran, no error, {len(picks)} picks, NE types used {sorted({c['ne'] for c in picks})}, "
          f"B3 gate_up types used {sorted({c['choice']['B3'][0] for c in picks})}")
except SystemExit as e:
    print("allocator exited:", e)
