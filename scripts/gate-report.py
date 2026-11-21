"""Writes results/pilot/gate.md, the release-gate summary (CHANGELOG.md pilot entry, Addenda C-F), from files only.

    python scripts/gate-report.py

Reads results/pilot/kl.tsv (rebuilt from logs/ by analysis/maptable.py first), allocation-top.json, selection.md,
gate-quality.txt, stability-*.json, bench-*.txt, logs/splice-sel-*.json and the GGUF files in models/. Missing
inputs are reported as not run. Teacher-2 rows (Addendum F) are used when logs/ppl-teach2-*.log exist.
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import compare  # noqa: E402
import maptable  # noqa: E402

R = ROOT / "results/pilot"
M = ROOT / "models"
LIMIT = 11663310784
_u = (ROOT / "models/uniform-IQ3_XXS.gguf")
# Addendum F calls it "stock IQ3_XXS (smaller)"; the label follows the file's real size
_ulabel = "stock IQ3_XXS" + ("" if not _u.exists() else " (smaller)" if _u.stat().st_size < LIMIT else " (larger than UD-IQ3_XXS)")
NAMES = {"cand": "cand", "uniform-IQ3_XXS": _ulabel, "UD-IQ3_XXS": "UD-IQ3_XXS"}
TEACHER_MARGIN = 0.005


def read(p: Path) -> str | None:
    return p.read_text(encoding="utf-8") if p.exists() else None


def margins(runs: dict, prefix: str, cand: str, base: str) -> list[str]:
    out = []
    for t in ("T1", "T2"):
        c, o = runs.get(f"{prefix}{cand}-{t}-P3"), runs.get(f"{prefix}{base}-{t}-P3")
        if c is None or o is None:
            out.append(f"| {t} | {NAMES.get(base, base)} | not run | | | | |")
            continue
        d, lo, hi = compare.paired(c["chunks"], o["chunks"])
        out.append(f"| {t} | {NAMES.get(base, base)} | {c['chunks'].mean():.5f} vs {o['chunks'].mean():.5f} | {d:+.5f} [{lo:+.5f}, {hi:+.5f}] "
                   f"| {c['top']:.2f}% vs {o['top']:.2f}% | {c['chunks'].size} | {int((c['chunks'] > o['chunks']).sum())} |")
    return out


def stability() -> tuple[list[str], str]:
    ref, cand = (json.loads(t) if (t := read(R / f"stability-{n}.json")) else None for n in ("ref", "cand"))
    if ref is None or cand is None:
        return ["Not run (phase 4 stops after a quality FAIL, Addendum F)." if cand is None else "Reference battery missing."], "not run"
    th = cand["thresholds"]
    checks = []
    for name, s in (("ref", ref), ("cand", cand)):
        g = s["long_generations"]
        checks.append((f"{name}: min distinct 4-gram ratio >= {th['distinct4_min']}", f"{min(x['distinct4'] for x in g):.4f}",
                       min(x["distinct4"] for x in g) >= th["distinct4_min"], name == "cand"))
        checks.append((f"{name}: max repeated-line run <= {th['repeat_run_max']}", str(max(x["repeat_run"] for x in g)),
                       max(x["repeat_run"] for x in g) <= th["repeat_run_max"], name == "cand"))
    checks.append(("planted fact found if the reference finds it",
                   f"ref {ref['long_context']['found']}, cand {cand['long_context']['found']} ({cand['long_context']['prompt_tokens']} prompt tokens)",
                   cand["long_context"]["found"] or not ref["long_context"]["found"], True))
    checks.append((f"valid tool calls at most {th['tool_drop_max']} fewer",
                   f"ref {ref['tool_calls']['valid']}/{ref['tool_calls']['n']}, cand {cand['tool_calls']['valid']}/{cand['tool_calls']['n']}",
                   cand["tool_calls"]["valid"] >= ref["tool_calls"]["valid"] - th["tool_drop_max"], True))
    checks.append(("both non-English replies pass", ", ".join(f"{x['chars']} chars / {x['letter_share']}" for x in cand["other_languages"]),
                   all(x["ok"] for x in cand["other_languages"]), True))
    checks.append((f"GSM8K 21-50 at most {th['gsm8k_drop_max']} below the reference",
                   f"ref {ref['gsm8k']['correct']}/{ref['gsm8k']['n']}, cand {cand['gsm8k']['correct']}/{cand['gsm8k']['n']}",
                   cand["gsm8k"]["correct"] >= ref["gsm8k"]["correct"] - th["gsm8k_drop_max"], True))
    checks.append((f"replies at the cap at most {th['cap_rise_max']} more",
                   f"ref {ref['gsm8k']['capped']}, cand {cand['gsm8k']['capped']}",
                   cand["gsm8k"]["capped"] <= ref["gsm8k"]["capped"] + th["cap_rise_max"], True))
    lines = ["| check | value | result |", "|---|---|---|"]
    lines += [f"| {c} | {v} | {'pass' if ok else 'FAIL'}{'' if gate else ' (reference, information only)'} |" for c, v, ok, gate in checks]
    lines.append("| llama-server chat smoke test | battery completed and JSON written | pass |")
    verdict = "PASS" if all(ok for _, _, ok, gate in checks if gate) else "FAIL"
    return lines, verdict


def bench(name: str) -> tuple[str, str]:
    text = read(R / f"bench-{name}.txt")
    if text is None:
        return "not run", "not run"
    vals = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[-2] in ("pp512", "tg128"):
            vals[cells[-2]] = cells[-1]
    return vals.get("pp512", "missing (see logs/bench-%s.log)" % name), vals.get("tg128", "missing")


def off_metal(path: Path) -> str:
    # same count as phase4.sh step 6, read from the file's tensor types
    import gguf
    if not path.exists():
        return "file missing"
    return str(sum(t.tensor_type.name in ("NVFP4", "F8_E4M3", "TQ1_0") for t in gguf.GGUFReader(path).tensors if "exps" in t.name))


def expert_types(path: Path) -> str:
    import gguf
    if not path.exists():
        return "file missing"
    counts: dict[str, int] = {}
    for t in gguf.GGUFReader(path).tensors:
        if "exps" in t.name and t.name.endswith(".weight"):
            k = f"{t.name.split('.')[2]} {t.tensor_type.name}"
            counts[k] = counts.get(k, 0) + 1
    return ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))


def main() -> None:
    maptable.collect()
    runs = compare.load()
    gq = read(R / "gate-quality.txt")
    verdict = re.search(r"quality gate: (PASS|FAIL)", gq or "")
    quality = verdict.group(1) if verdict else "not computed"
    out = ["# Release gate (pilot, P = 3)", "",
           "Written by `scripts/gate-report.py` from `results/pilot/` and `logs/`. Do not edit by hand. Exploratory pilot (CHANGELOG.md, 21 Nov 2026).", ""]

    # selection
    top = json.loads(read(R / "allocation-top.json") or "[]")
    rows = {r["run"]: r for r in csv.DictReader(open(R / "kl.tsv"), delimiter="\t")}
    kl = {k: float(r["mean_kl"]) for k, r in rows.items()}
    file_bytes = dict(re.findall(r"^  (sel-\d+): (\d+) bytes", read(ROOT / "logs/chain4.out") or "", re.M))
    out += ["## Selection on S (Addendum F)", "",
            "| candidate | predicted KL | measured KL on S (± SE) | measured / predicted | allocator bytes | file bytes | spliced and verified |",
            "|---|---|---|---|---|---|---|"]
    for i, c in enumerate(top):
        m = kl.get(f"sel-{i}-S-P3")
        sj = read(ROOT / f"logs/splice-sel-{i}.json")
        ver = "yes" if sj and json.loads(sj.strip().splitlines()[-1]).get("verified") else "no"
        fb = file_bytes.get(f"sel-{i}")
        out.append(f"| sel-{i} | {c['predicted_kl']:.5f} | {'not run' if m is None else f'{m:.5f} ± {float(rows[f"sel-{i}-S-P3"]["se"]):.5f}'} | "
                   f"{'' if m is None else f'{m / c['predicted_kl']:.3f}'} | {c['bytes']:,} | {'' if fb is None else f'{int(fb):,}'} | {ver} |")
    if "UD-IQ3_XXS-S-P3" in kl:
        out += ["", f"UD-IQ3_XXS on S (not used for selection): {kl['UD-IQ3_XXS-S-P3']:.5f}."]
    sel = read(R / "selection.md")
    if sel:
        chosen = re.search(r"\| (sel-\d+) \(chosen\)", sel)
        out += ["", f"Chosen: {chosen.group(1) if chosen else 'unknown'} (from `selection.md`)."]
    out.append("")

    # files
    out += ["## Files", "", "| build | bytes | under the UD-IQ3_XXS limit | expert tensor types |", "|---|---|---|---|"]
    for name in NAMES:
        p = M / f"{name}.gguf"
        size = p.resolve().stat().st_size if p.exists() else None
        out.append(f"| {NAMES[name]} | {'missing' if size is None else f'{size:,}'} | {'' if size is None else ('yes' if size <= LIMIT else 'no')} | {expert_types(p)} |")
    out.append("")

    # quality
    out += [f"## Quality gate on T1 and T2: {quality}", "",
            "Paired difference cand minus baseline, 95% bootstrap CI over chunks (10,000 resamples, seed 0). Negative favours cand.", "",
            "| set | baseline | mean KL cand vs baseline | diff [95% CI] | top-1 cand vs baseline | chunks | chunks where cand is worse |", "|---|---|---|---|---|---|---|"]
    for b in ("UD-IQ3_XXS", "uniform-IQ3_XXS"):
        out += margins(runs, "final-", "cand", b)
    out += ["", "Raw output of `analysis/compare.py`:", "", "```", (gq or "missing").strip(), "```", ""]

    # teacher check
    out += ["## Teacher check (Addendum F)", ""]
    small = []
    for t in ("T1", "T2"):
        c, o = runs.get(f"final-cand-{t}-P3"), runs.get(f"final-UD-IQ3_XXS-{t}-P3")
        if c and o:
            small.append(abs(c["chunks"].mean() - o["chunks"].mean()) < TEACHER_MARGIN)
    if any(k.startswith("teach2-") for k in runs):
        out += ["Second teacher: Q8_0 reference with UD-IQ3_XXS's NVFP4 ffn_down_exps (layers 0-19) and their .scale tensors spliced in.", "",
                "| set | baseline | mean KL cand vs baseline | diff [95% CI] | top-1 cand vs baseline | chunks | chunks where cand is worse |", "|---|---|---|---|---|---|---|"]
        out += margins(runs, "teach2-", "cand", "UD-IQ3_XXS")
        out.append("")
    elif quality != "PASS":
        gaps = [f"{t} {runs[f'final-cand-{t}-P3']['chunks'].mean() - runs[f'final-UD-IQ3_XXS-{t}-P3']['chunks'].mean():+.5f}"
                for t in ("T1", "T2") if f"final-cand-{t}-P3" in runs and f"final-UD-IQ3_XXS-{t}-P3" in runs]
        out += [f"Not run: the quality gate did not pass, and the cand minus UD-IQ3_XXS gaps ({', '.join(gaps)} KL) are not "
                f"margins below {TEACHER_MARGIN} in cand's favour.", ""]
    elif not any(small):
        out += [f"Not needed: both margins over UD-IQ3_XXS are at least {TEACHER_MARGIN} KL.", ""]
    else:
        out += [(read(R / "teacher-skip.txt") or "Needed but not run.").strip(), ""]

    # stability
    lines, sv = stability()
    out += [f"## Stability (cand against the Q8_0 reference): {sv}", ""] + lines + [""]

    # speed
    out += ["## Speed and Metal coverage (Addendum C)", "", "`llama-bench` -ngl 99 -p 512 -n 128 -r 3, default loop.count = 3. Phase 4 runs it only after a quality PASS. The Metal count is read from each file's tensor types.", "",
            "| build | pp512 t/s | tg128 t/s | expert tensors Metal cannot multiply |", "|---|---|---|---|"]
    for name in NAMES:
        pp, tg = bench(name)
        out.append(f"| {NAMES[name]} | {pp} | {tg} | {off_metal(M / f'{name}.gguf')} |")
    out.append("")

    final = "PASS" if quality == "PASS" and sv == "PASS" else "FAIL"
    out.insert(4, f"**Verdict: {final}.** Quality gate {quality}, stability {sv}. Uploads still need Daniel's go.\n")
    (R / "gate.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
