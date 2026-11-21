"""analysis/noise.py and analysis/perpass_report.py: stored noise ratio against a recomputation, duplicate-row handling,
the tied/independent ratio for each tied draw, and the control-section sentence when the two CI readings differ.
CPU only, small files, writes only to a temp dir.

    .venv/bin/python -I reviews/2026-11-21-code-review-analysis-opus/check_report.py
"""

import ast
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
import noise  # noqa: E402
import perpass_report as pr  # noqa: E402

R = ROOT / "results/pilot"
t = R / "perpass-kl.tsv"
stored = ast.literal_eval((R / "noise-ratio.txt").read_text().strip())
fresh = noise.ratio(noise.chunks(t, "noise-tied"), noise.chunks(t, "noise-indep"))
print("== noise ratio ==")
print(f"stored noise-ratio.txt equals a recomputation from perpass-kl.tsv: {stored == fresh}")
for c in ("noise-tied", "noise-tied2", "noise-tied3"):
    o = noise.ratio(noise.chunks(t, c), noise.chunks(t, "noise-indep"))
    print(f"{c} / noise-indep: {o['ratio']:.3f} [{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}]")

print("\n== duplicate rows: which one wins ==")
tmp = Path(tempfile.mkdtemp()) / "kl.tsv"
tmp.write_text("cond\tagainst\tmean_kl\tper_chunk\nx\tpp-base\t0.1\t0.1,0.1\nx\tpp-base\t0.2\t0.2,0.2\n")
print(f"noise.chunks -> {noise.chunks(tmp, 'x').tolist()}; perpass_report.kl_chunks -> {pr.kl_chunks(tmp)['x'].tolist()}; "
      f"perpass_report.kl_table -> {pr.kl_table(tmp)[('x', 'pp-base')]}")

print("\n== control_section sentence when CIs are disjoint but a point lies inside the other's CI ==")
main = {"fisher": {"r": 0.50, "ci95": {"r": [0.40, 0.45]}}}  # a percentile CI need not contain its point
other = {"fisher": {"r": 0.47, "ci95": {"r": [0.46, 0.60]}}}
disjoint, points = pr.apart(main, other, "fisher", "r")
sentence = ("do not overlap." if disjoint else "overlap.") + ("" if disjoint == points else " Each point lies outside the other's CI.")
print(f"apart -> disjoint {disjoint}, points outside {points}; control_section would print: 'CIs {sentence}'")

print("\n== text in per_pass.py's docstring vs CHANGELOG ==")
doc = (ROOT / "analysis/per_pass.py").read_text().split('"""')[1]
print("per_pass.py says 'not recorded in CHANGELOG.md':", "not\nrecorded in CHANGELOG.md" in doc or "not recorded in CHANGELOG.md" in doc,
      "; CHANGELOG has Addendum G section 2:", "**2. Per-pass analysis deviations" in (ROOT / "CHANGELOG.md").read_text())
