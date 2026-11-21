"""Writes reviews/2026-11-21-decision-brief-evening.md. Every number in the brief is read here from a file.

Run: .venv/bin/python -I reviews/2026-11-21-decision-brief-evening/build.py
Reads result files, logs, source files and the saved PR record. Starts no GPU job; runs pytest (CPU) once.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R, L, M = ROOT / "results/pilot", ROOT / "logs", ROOT / "models"
sys.path.insert(0, str(ROOT / "analysis"))
import maptable  # noqa: E402


def lineno(path: str, needle: str) -> int:
    for i, line in enumerate((ROOT / path).read_text().splitlines(), 1):
        if needle in line:
            return i
    raise SystemExit(f"{path}: no line with {needle!r}")


# paired comparisons (compare.py output)
CMP = re.compile(r"^(T\d) cand vs (\S+): KL ([\d.]+) vs ([\d.]+), diff ([+-][\d.]+) \[")
diff = {}
for f in ("gate-quality.txt", "mac-quality.txt"):
    for line in (R / f).read_text().splitlines():
        if m := CMP.match(line):
            diff[m.group(1), m.group(2)] = m.group(5)
gate_last = (R / "gate-quality.txt").read_text().strip().splitlines()[-1]
mac_last = (R / "mac-quality.txt").read_text().strip().splitlines()[-1]


def bench(name: str) -> dict:
    out = {}
    for line in (R / f"bench-{name}.txt").read_text().splitlines():
        if m := re.search(r"\|\s+(pp512|tg128)\s+\|\s+([\d.]+) ±", line):
            out[m.group(1)] = float(m.group(2))
    return out


bc, bu = bench("cand"), bench("UD-IQ3_XXS")

# P = 2/3/4 timing
t = {int(a): (int(b), int(c)) for a, b, c in (ln.split("\t") for ln in (R / "p4-timing.txt").read_text().splitlines()[1:4])}

# S scores
kl = {ln.split("\t")[0]: float(ln.split("\t")[1]) for ln in (R / "kl.tsv").read_text().splitlines()[1:]}

# file sizes
size = {n: os.path.getsize(M / f"{n}.gguf") for n in ("cand", "UD-IQ3_XXS", "pool-3bit")}

# upstream PR record saved by the direction review
pr = json.loads((ROOT / "reviews/2026-11-21-direction-edge-fable/pr-20456.json").read_text())

# Amendment 2 comparators
names = ("isvalorum-NanoPlus", "prithiv-Q2_K")
logs = [L / f"ppl-final-{n}-{s}-P3.log" for n in names for s in ("T1", "T2")]
scored = sum(1 for p in logs if p.exists() and "Mean" in p.read_text(errors="replace") and re.search(r"Mean\s+KLD", p.read_text(errors="replace")))
present = sum(1 for n in names if (M / "ext" / f"{n}.gguf").exists())

# upload log
up = L / "hf-upload-v1.log"
up_time = time.strftime("%H:%M", time.localtime(up.stat().st_mtime))
up_has_url = up.read_text().startswith("url=https://huggingface.co/")

# release README against the Addendum G card rule
readme = (ROOT / "drafts/release/README.md").read_text().splitlines()
fail_line = next(i for i, ln in enumerate(readme, 1) if re.search(r"fail", ln, re.I))
margins = [diff["T1", "UD-IQ3_XXS"].lstrip("+"), diff["T2", "UD-IQ3_XXS"].lstrip("+")]
margins_in = sum(1 for m in margins if any(m in ln for ln in readme))

# run-time estimates from existing S logs
def eta(log: str) -> str:
    return re.search(r"seconds per pass - ETA ([\d.]+) minutes", (L / log).read_text(errors="replace")).group(1)


# maptable: logs that lose chunk lines
kl_logs, dropped = 0, []
for p in sorted(L.glob("ppl-*.log")):
    text = p.read_text(errors="replace")
    if not re.search(r"Mean\s+KLD", text):
        continue
    kl_logs += 1
    declared = int(re.search(r"over (\d+) chunks", text).group(1))
    kept = sum(1 for ln in text.splitlines() if maptable.CHUNK.match(ln))
    if kept != declared:
        dropped.append((p.name, kept, declared))

# tests
out = subprocess.run([str(ROOT / ".venv/bin/python"), "-I", "-m", "pytest", "-q", "tests/"], cwd=ROOT, capture_output=True, text=True).stdout
passed = int(re.search(r"(\d+) passed", out).group(1))
gate_tests = sum(1 for p in (ROOT / "tests").glob("test_*.py") if re.search(r"maptable|allocate|compare", p.read_text()))

# slot range of pass 4 (CLAUDE.md formula: 24 + (p - 2) * 12 + (l - 6), core layers 6 to 17)
slot = lambda p, l: 24 + (p - 2) * 12 + (l - 6)  # noqa: E731

P3 = "patches/llama.cpp/0003-model-inkling2_loop-cache-slots-can-carry-their-own-.patch"
chain = (L / "chain6.out").read_text().strip().splitlines()[-1]
dn, dk, dd = dropped[0] if dropped else ("none", 0, 0)

q8_max = max(float(x) for x in next(ln for ln in (R / "kl.tsv").read_text().splitlines() if ln.startswith("floor-q8-S-P3\t")).split("\t")[5].split(","))

brief = f"""# Decision brief, evening of 2026-11-21

Written by `reviews/2026-11-21-decision-brief-evening/build.py` at {time.strftime("%H:%M")}. The script reads every number from a file. "Inference" and "estimate" mark what is not measured. No GPU job was started.

## 1. Answer

There is an edge, and it is narrow: among GGUFs under {size["UD-IQ3_XXS"]:,} bytes that run fully on Metal, the candidate has the lowest held-out KL, ahead of AtomicChat AD-IQ2_S by {diff["T1", "atomic-AD-IQ2_S"].lstrip("-")} on T1 and {diff["T2", "atomic-AD-IQ2_S"].lstrip("-")} on T2 at the same speed. Against UD-IQ3_XXS it loses by {margins[0]} and {margins[1]}, and leads only on Mac speed ({bc["pp512"] / bu["pp512"]:.2f}x pp512, {bc["tg128"] / bu["tg128"]:.2f}x tg128), because Metal has no NVFP4 matmul. Upstream PR #{pr["number"]} ({pr["additions"]} added lines, {pr["state"]}, last updated {pr["updated_at"][:10]}) would end that lead (inference). Do not redo the candidate: it is frozen, T1 and T2 are seen, and our {size["pool-3bit"] / 1e9:.2f} GB all-3-bit build scores {kl["sweep-3bit-S-P3"]:.5f} on S against UD's {kl["UD-IQ3_XXS-S-P3"]:.5f}, so no reallocation inside Metal types passes the gate (inference). The model is a stopgap for llama.cpp on a Mac at {size["cand"] / 1e9:.2f} GB, and the pilot's per-pass method is worth more than the file.

**P = 4 speed adds up.** {t[4][0]} s against {t[3][0]} s at P = 3 is a ratio of {t[4][0] / t[3][0]:.2f}. Layer visits ({t[4][1]} against {t[3][1]}) give {t[4][1] / t[3][1]:.2f}. Timing only, no KL at P = 4.

## 2. Your decisions

1. **The upload.** `logs/hf-upload-v1.log` (written {up_time}) {"holds a Hugging Face commit URL" if up_has_url else "holds no URL"}. I did not open it. {scored} of {len(logs)} Amendment 2 comparator runs have a KL, so the Mac-track rule is not complete. Options: leave it (private, or approved by you), or set it private until the rule is complete. Recommendation: private until then.
2. **Addendum G.2 deviations.** Approve or reject. Recommendation: approve, and keep the full-vocabulary form as the sensitivity check.
3. **Diagnostic on S.** (a) Hybrid: the candidate with UD's native NVFP4 down. (b) Hybrid plus mirror: UD's gate_up and non-experts with our IQ4_NL down. (c) Neither. Recommendation: (b). Paste the Amendment 3 draft and write a dated mirror entry before any score. Mirror is the obvious rival to the Mac claim.
4. **Release.** (a) One model upload, no upkeep. (b) Measurements only. (c) Nothing. Recommendation: (a) only if both pending comparators are beaten and mirror does not beat the candidate, otherwise (b). `drafts/release/README.md` first mentions the failure on line {fail_line} of {len(readme)} and states {margins_in} of {len(margins)} margins. Addendum G wants the FAIL first, with both.
5. **Metal NVFP4 kernel (Claude) and the NVFP4 to IQ4_NL re-encode (yours).** (a) Start now (kernel spike estimate: half a day). (b) Wait for decision 3: cand minus hybrid on S is the prize for the kernel and the ceiling for the re-encode. (c) Drop both from the pilot. Recommendation: (b).
6. **Gate-path tests.** Claude adds tests for `maptable.py`, `allocate.py`, `compare.py` and a chunk-count check now (CPU only), or later. Recommendation: now, before anything is published.

## 3. GPU queue, in order

1. **Running:** `scripts/pilot-chain6.sh`, last log line "{chain}". It ends with `macgate.sh`: IsValorum APEX-I-NanoPlus-V2.1 and prithivMLmods Q2_K on T1 and T2. {present} of {len(names)} files are in place under `models/ext`.
2. **`scripts/hybrid.sh`.** Ready, not run. Needs Amendment 3 in CHANGELOG first. About {eta("ppl-UD-IQ3_XXS-S-P3.log")} min, the ETA of UD's own S run (inference).
3. **Mirror.** No script yet. `scripts/splice.py --take` and `--add` cover it, with `cand.gguf` as base. About {eta("ppl-sel-2-S-P3.log")} min, the ETA of an all-Metal S run (inference).
4. **Per-pass controls,** each registered first: 3-bit layer split, Gaussian single-pass, more noise draws, then B3. At P = 4 a per-pass file leaves pass 4 (slots {slot(4, 6)} to {slot(4, 17)}) on the physical experts (inference from the patch review).

## 4. Code-review findings to look at

No finding changes a reported number. A second agent rechecked the first four and rated each minor.

- `analysis/maptable.py:{lineno("analysis/maptable.py", "CHUNK = re.compile")}` and `:{lineno("analysis/maptable.py", "per_chunk.append(")}`. Chunk lines the regex cannot parse are dropped without a warning, and the next chunk gets a wrong value. Fired on {len(dropped)} of {kl_logs} KL logs: `{dn}` keeps {dk} of {dd} chunks (largest kept value {q8_max:.6f}).
- `tests/`. {gate_tests} test files touch `maptable.py`, `allocate.py` or `compare.py` ({passed} tests pass in total). The gate and Mac verdicts run through all three.
- `tests/test_per_pass.py:{lineno("tests/test_per_pass.py", "def test_shift_invariance")}`: `test_shift_invariance` cannot fail. `:{lineno("tests/test_per_pass.py", "assert np.all(full > head)")}`: the head norm, which scores the registration, is checked only as `full > head`. Removing the centring at `analysis/per_pass.py:{lineno("analysis/per_pass.py", "fisher = float(")}` or `:{lineno("analysis/per_pass.py", "euclid = float(np.sum(np.where(head")}` passes every test.
- `{P3}:{lineno(P3, "diff --git a/src/models/inkling2-loop.cpp")}`. On upstream 690b87d the series applies without a conflict, but this hunk lands in the wrong function. Keep the pin at `scripts/build-llama.sh:{lineno("scripts/build-llama.sh", "BASE=")}`.
- `analysis/compare.py:{lineno("analysis/compare.py", '"quality gate:"')}` prints one label for two rules: `mac-quality.txt` ends with "{mac_last}", `gate-quality.txt` with "{gate_last}". From the review only.
"""
assert "—" not in brief
dst = ROOT / "reviews/2026-11-21-decision-brief-evening.md"
dst.write_text(brief)
print(brief)
print(f"words: {len(brief.split())}", file=sys.stderr)
