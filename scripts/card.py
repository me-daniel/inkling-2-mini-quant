"""Renders drafts/MODEL_CARD.md and drafts/note.md (pilot write-up drafts, never uploaded by this script) from files.

    .venv/bin/python -I scripts/card.py

Every number comes from results/ (pilot tables, env.md, data.md, t2.md, convert-check.md), logs/revisions.txt, the
llama-perplexity logs (context and chunk counts) and the imatrix download metadata in models/. Gate differences are
recomputed with analysis/compare.py (same paired bootstrap), per-pass ratios and the Addendum F test with
analysis/perpass_report.py. Card wording follows CHANGELOG.md (pilot entry, Addenda C, E, F). If results/pilot/gate.md
does not say PASS, the card is a stub marked NOT FOR UPLOAD and the note carries the measurements. A PASS card needs
the stability and bench files and is not drafted here.

Mac-track mode (Addendum G): when gate-quality.txt says FAIL and mac-quality.txt exists, the card is a full draft for
review. It states the FAIL against UD-IQ3_XXS, discloses that the comparison is post hoc, and gives the seven-file table
(bytes from the files, compare.py lines rechecked against kl.tsv, llama-bench output, off-Metal counts), the speed
ratios, the stability battery judged by scripts/stability.py judge(), setup and provenance. The note gets the same
table. The output is checked for wording the CHANGELOG forbids, private paths and dashes; any hit exits 1 without
writing. The one draft line naming the reviewer is prepended to the card after that check.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import compare  # noqa: E402
import noise  # noqa: E402
import perpass_report as pr  # noqa: E402

R = ROOT / "results/pilot"
L = ROOT / "logs"
OUT = ROOT / "drafts"
FORBIDDEN = [r"loop-aware", r"pass-aware", r"\bfirst\b", r"\bbest\b", r"same[- ]size", r"runs fully on metal", r"\bGSQ\b",
             r"\bGPTQ\b", r"\bRCO\b", r"once per visit", r"per visit", r"dominat", r"compounds", r"loop-specific effect",
             r"/Users/", r"private/", r"scratch/", r"(?<!me-)Daniel", r"—", r"–", r"\bSOTA\b", r"state of the art",
             r"\bfastest\b", r"beats unsloth", r"\bbeats? UD"]


REL = "[b12588-mix](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8)"
PR = "[unslothai/llama.cpp#348](https://github.com/unslothai/llama.cpp/pull/348)"


def text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def table(md: str, first: str) -> list[dict[str, str]]:
    """Rows of the markdown table whose header starts with `| first |`."""
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(f"| {first} |"):
            head = [c.strip() for c in ln.strip("|").split("|")]
            rows = []
            for row in lines[i + 2:]:
                if not row.startswith("|"):
                    break
                rows.append(dict(zip(head, (c.strip() for c in row.strip("|").split("|")))))
            return rows
    raise SystemExit(f"table '{first}' not found")


def one(pattern: str, s: str, flags: int = 0) -> re.Match:
    m = re.search(pattern, s, flags)
    if not m:
        raise SystemExit(f"pattern not found: {pattern}")
    return m


def ctx(log: str) -> tuple[int, int]:
    m = one(r"computing over (\d+) chunks, n_ctx=(\d+)", text(L / log))
    return int(m[1]), int(m[2])


def ci(d: dict, k: str, f: str = ".3f") -> str:
    lo, hi = d["ci95"][k]
    return f"{d[k]:{f}} [{lo:{f}}, {hi:{f}}]"


def n(v: int) -> str:
    return f"{v:,}"


def load() -> dict:
    v: dict = {}
    # engine and model
    env = text(ROOT / "results/env.md")
    m = one(r"llama\.cpp base: `([0-9a-f]+)` \((b\d+)\)", env)
    v["base"], v["tag"] = m[1], m[2]
    v["head"] = one(r"Patched HEAD: `([0-9a-f]+)`", env)[1]
    v["patches"] = re.findall(r"^  - `(\d{4}-[^`]+\.patch)` `([0-9a-f]+)`", env, re.M)
    v["build"] = one(r"^- Build: (.+)$", env, re.M)[1]
    v["machine"] = one(r"^- Machine: (.+)$", env, re.M)[1]
    v["loop_tests"] = len(re.findall(r"\(OK\)$", env, re.M))
    nmse = [float(x) for x in re.findall(r"nmse (\S+) \(OK\)$", env, re.M)]
    assert len(nmse) == v["loop_tests"], "loop test lines without an nmse value"
    v["loop_zero"], v["loop_max_nmse"] = sum(x == 0 for x in nmse), max(nmse)
    v["calib_col"] = next(k for k in table(text(ROOT / "results/data.md"), "File")[0] if k.startswith("13-gram"))
    v["explore_bench"] = sorted(p.name for p in (R / "exploratory").glob("bench-*.md"))
    v["rev"] = dict(re.findall(r"^(\S+/\S+) sha=([0-9a-f]+)$", text(L / "revisions.txt"), re.M))
    v["model"] = next(k for k in v["rev"] if not k.startswith("unsloth/"))
    v["ud_repo"] = next(k for k in v["rev"] if k.startswith("unsloth/"))
    meta = text(ROOT / "models/ggml-org/.cache/huggingface/download/imatrix.gguf.metadata").split()
    conv = text(ROOT / "results/convert-check.md")
    v["imat_repo"] = one(r"(ggml-org/[\w.-]+)", conv)[1]
    assert meta[0].startswith(one(r"revision `([0-9a-f]+)`", conv)[1]), "imatrix revision differs from convert-check.md"
    v["imat_rev"], v["imat_sha"] = meta[0], meta[1]
    # data and scoring setup
    v["data"] = table(text(ROOT / "results/data.md"), "File")
    t2 = text(ROOT / "results/t2.md")
    m = one(r"at P = (\d+), reasoning_effort (\w+), temperature (\S+), cap ([\d,]+) tokens", t2)
    v["t2_P"], v["t2_effort"], v["t2_temp"], v["t2_cap"] = m[1], m[2], m[3], m[4]
    m = one(r"Prompts: (\d+) \(([^)]+)\), completion tokens ([\d,]+), hit the cap (\d+)", t2)
    v["t2_prompts"], v["t2_src"], v["t2_tokens"], v["t2_capped"] = m[1], m[2], m[3], m[4]
    v["ctx"] = {s: ctx(f"ppl-final-cand-{s}-P3.log") for s in ("T1", "T2")}
    v["ctx"]["S"] = ctx("ppl-sel-0-S-P3.log")
    gate = text(R / "gate.md")
    v["P"] = one(r"^# Release gate \(pilot, P = (\d+)\)", gate, re.M)[1]
    v["P_default"] = one(r"default loop\.count = (\d+)", gate)[1]
    m = one(r"\(([\d,]+) resamples, seed (\d+)\)", gate)
    v["boot"], v["seed"] = m[1], m[2]
    v["verdict"] = one(r"\*\*Verdict: (\w+)\.\*\*", gate)[1]
    v["stability_run"] = any(R.glob("stability-*.json"))
    v["bench_run"] = any(R.glob("bench-*.txt"))
    v["files"] = {r["build"]: r for r in table(gate, "build")}
    v["metal"] = {r["build"]: r["expert tensors Metal cannot multiply"] for r in table(gate, "build | pp512 t/s")}
    v["sel_gate"] = {r["candidate"]: r for r in table(gate, "candidate")}
    v["teacher_margin"] = one(r"margin over UD-IQ3_XXS is below ([\d.]+) KL", text(ROOT / "CHANGELOG.md"))[1]
    m = one(r"our IQ4_NL against Q8_0 is outside ([\d.]+) to ([\d.]+) KL", text(ROOT / "CHANGELOG.md"))
    v["floor_band"] = (m[1], m[2])
    # KL runs
    rows = {r["run"]: r for r in csv.DictReader(open(R / "kl.tsv"), delimiter="\t")}
    v["kl"] = {k: float(r["mean_kl"]) for k, r in rows.items()}
    v["se"] = {k: float(r["se"]) for k, r in rows.items()}
    v["top"] = {k: float(r["same_top_p"]) for k, r in rows.items()}
    runs = compare.load()
    v["gate"] = []
    for t in ("T1", "T2"):
        c = runs[f"final-cand-{t}-P{v['P']}"]
        for b in ("UD-IQ3_XXS", "uniform-IQ3_XXS"):
            o = runs[f"final-{b}-{t}-P{v['P']}"]
            d, lo, hi = compare.paired(c["chunks"], o["chunks"])
            v["gate"].append(dict(set=t, base=b, kc=c["chunks"].mean(), kb=o["chunks"].mean(), d=d, lo=lo, hi=hi,
                                  tc=c["top"], tb=o["top"], n=c["chunks"].size, worse=int((c["chunks"] > o["chunks"]).sum())))
    # selection
    top = json.loads(text(R / "allocation-top.json"))
    chosen = int(one(r"\| sel-(\d+) \(chosen\)", text(R / "selection.md"))[1])
    v["chosen"] = chosen
    v["sel"] = [dict(i=i, pred=c["predicted_kl"], meas=v["kl"][f"sel-{i}-S-P3"], se=v["se"][f"sel-{i}-S-P3"],
                     abytes=c["bytes"], fbytes=int(v["sel_gate"][f"sel-{i}"]["file bytes"].replace(",", "")), ne=c["ne"],
                     choice=c["choice"]) for i, c in enumerate(top)]
    # phase 2
    p2 = text(R / "phase2.md")
    v["blocks"] = table(p2, "Block")
    v["sweep"] = table(p2, "Level")
    v["timing"] = re.findall(r"^- P = (\d): (\d+) s, ratio to P = 3 ([\d.]+) \(visits ratio ([\d.]+)\)", p2, re.M)
    v["additivity"] = re.findall(r"(\d)-bit ([\d.]+) vs ([\d.]+) \(ratio ([\d.]+)\)", p2)
    assert all(abs(float(u) / float(s_) - float(r_)) < 0.01 for _, s_, u, r_ in v["additivity"]), "map ratio is not uniform / sum"
    # our Q2_0 encoder
    q = list(csv.DictReader(open(R / "q2opt-errors.tsv"), delimiter="\t"))
    v["q2"] = {k: (np.mean([float(r["t_stock"]) for r in q if r["kind"] == k]),
                   np.mean([float(r["t_opt"]) for r in q if r["kind"] == k]), sum(r["kind"] == k for r in q))
               for k in ("ffn_down_exps", "ffn_gate_up_exps")}
    # per-pass
    v["pp"] = pr.with_derived(json.loads(text(R / "perpass-residual.json")))
    v["ls"] = pr.with_derived(json.loads(text(R / "layersplit-residual.json")))
    v["b3"] = pr.with_derived(json.loads(text(R / "perpass3bit-residual.json")))
    v["ppkl"] = pr.kl_table(R / "perpass-kl.tsv")
    kc = pr.kl_chunks(R / "perpass-kl.tsv")
    v["kl_ratio"] = noise.paired_ratio([kc["all"]], [kc[c] for c in ("p1", "p2", "p3")])
    v["kl2"] = {r["cond"]: float(r["kl_vs_base"]) for r in csv.DictReader(open(R / "perpass2-kl.tsv"), delimiter="\t")}
    v["noise"] = ast.literal_eval(text(R / "noise-ratio.txt").strip())
    v["parity"] = "byte-identical" in text(R / "perpass-parity.txt")
    v["ls_pairs"] = [f"{c[2:][:len(c[2:]) // 2]}-{c[2:][len(c[2:]) // 2:]}" for c in v["kl2"] if c.startswith("ls")]
    v["head_nats"] = one(r"the (\d+)-nat head", text(R / "perpass.md"))[1]
    nt = list(csv.DictReader(open(R / "noise-t.tsv"), delimiter="\t"))
    v["t2bit"] = {k: np.mean([float(r["t_2bit"]) for r in nt if r["kind"] == k]) for k in sorted({r["kind"] for r in nt})}
    tb = list(csv.DictReader(open(R / "t-blocks.tsv"), delimiter="\t"))
    b2 = one(r"(\d+)-(\d+)", next(b["Layers"] for b in v["blocks"] if b["Block"] == "B2"))
    v["b2_layers"] = (int(b2[1]), int(b2[2]))
    v["t3bit"] = np.mean([float(r["t"]) for r in tb if r["pool"] == "3bit" and r["kind"] == "ffn_gate_up_exps"
                          and v["b2_layers"][0] <= int(r["layer"]) <= v["b2_layers"][1]])
    v["t2gu_tb"] = np.mean([float(r["t"]) for r in tb if r["pool"] == "2bit" and r["kind"] == "ffn_gate_up_exps"
                            and v["b2_layers"][0] <= int(r["layer"]) <= v["b2_layers"][1]])
    # noise: every tied draw against the one independent draw
    v["tied"] = {(1 if c == "noise-tied" else int(c[len("noise-tied"):])): kc[c] for c in kc if c.startswith("noise-tied")}
    v["tied_ratio"] = {d: noise.paired_ratio([x], [kc["noise-indep"]]) for d, x in sorted(v["tied"].items())}
    assert abs(v["tied_ratio"][1]["ratio"] - v["noise"]["ratio"]) < 1e-9, "draw-1 ratio differs from noise-ratio.txt"
    # Addendum G section 2 (deviations) and the Mac track
    cl = text(ROOT / "CHANGELOG.md")
    m = one(r"fixed in commit (\w+) at ([\d:]+), before the first per-pass logits at ([\d:]+); (awaiting \w+'s approval)", cl)
    v["dev_commit"], v["dev_time"], v["dev_logits"] = m[1], m[2], m[3]
    v["dev_open"] = "awaiting" in m[4]
    m = one(r"### Addendum G \(([\d-]+), ~([\d:]+), after the quality gate FAILed", cl)
    v["g_date"], v["g_time"] = m[1], m[2]
    v["g_amend"] = one(r"\*\*Amendment \(~([\d:]+), before any download or score", cl)[1]
    v["gq"] = cmp_file(R / "gate-quality.txt")
    assert v["gq"]["verdict"] == v["verdict"], "gate-quality.txt and gate.md disagree"
    v["mac"] = (R / "mac-quality.txt").exists() and v["gq"]["verdict"] == "FAIL"
    v["pw"] = prior_work(ROOT / "reviews/2026-11-21-scoop-check.md")
    if v["mac"]:
        load_mac(v, runs)
    return v


CMP = re.compile(r"^(T\d) (\S+) vs (\S+): KL ([\d.]+) vs ([\d.]+), diff ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\], "
                 r"top-1 ([\d.]+)% vs ([\d.]+)% -> (better|not better)$", re.M)


def cmp_file(p: Path) -> dict:
    """Lines of analysis/compare.py output: rows keyed by (set, baseline), the baselines in order, the gate line."""
    s = text(p)
    rows = {(m[1], m[3]): dict(cand=m[2], kc=m[4], ko=m[5], d=m[6], lo=m[7], hi=m[8], tc=m[9], to=m[10], win=m[11] == "better")
            for m in CMP.finditer(s)}
    assert rows and len(rows) == len([ln for ln in s.splitlines() if " vs " in ln]), f"unparsed line in {p.name}"
    g = re.search(r"^quality gate: (PASS|FAIL)$", s, re.M)
    return {"rows": rows, "others": list(dict.fromkeys(o for _, o in rows)), "verdict": g[1] if g else None, "name": p.name}


def load_mac(v: dict, runs: dict) -> None:
    v["mq"] = cmp_file(R / "mac-quality.txt")
    v["mc"] = cmp_file(R / "mac-context-IQ2_M.txt")
    rows = {**v["gq"]["rows"], **v["mq"]["rows"], **v["mc"]["rows"]}
    # every difference recomputed from kl.tsv with the same bootstrap must match the printed line
    for (t, o), r in rows.items():
        c, b = runs[f"final-cand-{t}-P{v['P']}"], runs[f"final-{o}-{t}-P{v['P']}"]
        d, lo, hi = compare.paired(c["chunks"], b["chunks"])
        assert (f"{d:+.5f}", f"{lo:+.5f}", f"{hi:+.5f}", f"{b['chunks'].mean():.5f}") == (r["d"], r["lo"], r["hi"], r["ko"]), (t, o)
    v["mrows"] = rows
    v["mnames"] = ["cand"] + v["gq"]["others"] + v["mq"]["others"] + v["mc"]["others"]
    v["counted"] = v["mq"]["others"]
    v["context"] = v["mc"]["others"]
    revs = {m[3]: (m[1], m[2]) for m in re.finditer(r"^(\S+/\S+) sha=([0-9a-f]+) (\S+\.gguf)$", text(L / "revisions.txt"), re.M)}
    reps = {int(x) for x in re.findall(r"llama-bench\" .* -r (\d+)", text(ROOT / "scripts/macgate.sh"))}
    assert len(reps) == 1, "bench repetitions not unique in macgate.sh"
    v["bench_reps"] = reps.pop()
    v["mf"] = {}
    sizes = {r[0]: int(r[1]) for r in (l.split("\t") for l in text(R / "file-bytes.tsv").splitlines()[1:])}
    spec_fb = importlib.util.spec_from_file_location("file_bytes", ROOT / "scripts/file-bytes.py")
    fb = importlib.util.module_from_spec(spec_fb)
    spec_fb.loader.exec_module(fb)
    for name in v["mnames"]:
        ext = name in fb.EXT  # outside files: sizes from file-bytes.tsv, the files themselves may be deleted
        b = text(R / f"bench-{name}.txt")
        sp = {m[1]: (float(m[2]), m[2], m[3]) for m in re.finditer(r"\|\s*(pp512|tg128)\s*\|\s*([\d.]+) ± ([\d.]+)\s*\|", b)}
        assert set(sp) == {"pp512", "tg128"}, name
        f = dict(bytes=sizes[name], ext=ext, sp=sp, build=one(r"^build: ([0-9a-f]+)", b, re.M)[1],
                 off=int(one(r"expert tensors in types Metal cannot multiply: (\d+)", b)[1]))
        assert v["head"].startswith(f["build"]), f"{name}: bench build is not the patched HEAD"
        if ext:
            ty = text(R / f"types-{name}.txt")
            f["loop"] = one(r"loop\.count (\S+)", ty)[1]
            f["mtp"] = int(one(r"^mtp tensors: (\d+)", ty, re.M)[1])
            assert int(one(r"\(experts: (\d+) \)", ty)[1]) == f["off"], f"{name}: types and bench disagree on Metal"
            f["repo"], f["rev"] = revs[fb.EXT[name]]
            f["file"] = fb.EXT[name]
        v["mf"][name] = f
    assert v["mf"]["cand"]["bytes"] == int(v["files"]["cand"]["bytes"].replace(",", "")), "cand bytes differ from gate.md"
    assert v["mf"]["UD-IQ3_XXS"]["bytes"] == int(v["files"]["UD-IQ3_XXS"]["bytes"].replace(",", "")), "UD bytes differ"
    spec = importlib.util.spec_from_file_location("stability", ROOT / "scripts/stability.py")
    stab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stab)
    v["stab"] = {k: json.loads(text(R / f"stability-{k}.json")) for k in ("ref", "cand")}
    v["judge"] = stab.judge(v["stab"]["ref"], v["stab"]["cand"])


def prior_work(p: Path) -> dict:
    """Links, dates and the claims quoted in the prior-work check (reviews/*scoop-check.md)."""
    s = text(p)
    out = {"date": one(r"^# Scoop and prior-work check, (.+)$", s, re.M)[1], "file": p.relative_to(ROOT).as_posix()}
    for m in re.finditer(r"^\d+\. \*\*\[([\w ]+), (arXiv [\d.]+)\]\((\S+?)\), ([\w ]+)\.?\*\*", s, re.M):
        out[m[1]] = dict(arxiv=m[2], url=m[3], date=m[4])
    m = one(r"^\| \[(shinmegamiboson/[\w.-]+)\]\((\S+?)\) \| ([^|]+?) \|", s, re.M)
    out["LookAtMyQuant"] = dict(repo=m[1], url=m[2], date=m[3])
    out["lamq_quotes"] = re.findall(r"\"((?:Measured per-layer|48-visit)[^\"]+)\"", s)
    assert len(out["lamq_quotes"]) == 2, "LookAtMyQuant quotes not found"
    return out


def gate_table(v: dict, label: dict) -> list[str]:
    out = ["| Set | Baseline | KL cand | KL baseline | cand minus baseline [95% CI] | Chunks where cand is worse | "
           "Top-1 cand vs baseline |", "|---|---|---|---|---|---|---|"]
    for g in v["gate"]:
        out.append(f"| {g['set']} | {label[g['base']]} | {g['kc']:.5f} | {g['kb']:.5f} | {g['d']:+.5f} [{g['lo']:+.5f}, "
                   f"{g['hi']:+.5f}] | {g['worse']} of {g['n']} | {g['tc']:.2f}% vs {g['tb']:.2f}% |")
    return out


def provenance(v: dict) -> list[str]:
    out = [f"- Model: `{v['model']}` at revision `{v['rev'][v['model']]}`.",
           f"- Comparison file: `UD-IQ3_XXS` from `{v['ud_repo']}` at revision `{v['rev'][v['ud_repo']]}`.",
           f"- Engine: llama.cpp `{v['base']}` ({v['tag']}) plus {len(v['patches'])} patches, patched HEAD "
           f"`{v['head']}`. {v['build']}.",
           f"- Machine: {v['machine']}.",
           f"- Importance matrix: `imatrix.gguf` from `{v['imat_repo']}` at revision `{v['imat_rev']}`, sha256 "
           f"`{v['imat_sha']}`.",
           "", "| Patch | sha256 prefix |", "|---|---|"]
    out += [f"| `{p}` | `{h}` |" for p, h in v["patches"]]
    return out


def setup(v: dict) -> list[str]:
    c = v["ctx"]
    return [f"- Pass count P = {v['P']} for every run (the model's default loop count is {v['P_default']}).",
            f"- Reference: a Q8_0 GGUF converted from the release, scored at the same P.",
            f"- Metric: mean KL divergence and top-1 agreement against the reference, from `llama-perplexity "
            f"--kl-divergence`. The text is fixed, so scoring involves no sampling.",
            "- Context: " + ", ".join(f"{k} in {c[k][0]} chunks of {n(c[k][1])} tokens" for k in ("S", "T1", "T2")) + ".",
            f"- T2 text: reasoning traces from the reference at P = {v['t2_P']} on {v['t2_prompts']} prompts "
            f"({v['t2_src']}). Effort {v['t2_effort']}, temperature {v['t2_temp']}, cap {v['t2_cap']} tokens. "
            f"{v['t2_tokens']} completion tokens; {v['t2_capped']} traces hit the cap. The traces were made before "
            "any candidate existed.",
            f"- Statistics: paired bootstrap over chunks, {v['boot']} resamples, seed {v['seed']}, 95% percentile CI."]


def data_table(v: dict) -> list[str]:
    col = v["calib_col"]
    out = [f"| File | Tokens | sha256 prefix | Exact word {col.replace('bartowski v6', 'bartowski v6 calibration text')} |",
           "|---|---|---|---|"]
    out += [f"| {r['File'].replace('data/', '')} | {r['Tokens']} | {r['sha256 (first 16)']} | {r[col]} |" for r in v["data"]]
    return out


def choice(c: dict) -> str:
    return ", ".join(f"{b} {g}/{d.replace('Q2_0O', 'Q2_0 (ours)')}" for b, (g, d) in c.items())


LABEL = {"cand": "cand (this file)", "UD-IQ3_XXS": "Unsloth UD-IQ3_XXS", "bart-IQ2_S": "bartowski IQ2_S",
         "atomic-AD-IQ2_S": "AtomicChat AD-IQ2_S", "apex-I-Nano": "mudler APEX-I-Nano", "bart-IQ2_M": "bartowski IQ2_M"}


def mac_label(v: dict, k: str, card: bool = False) -> str:
    if k == "cand":
        return LABEL[k] if card else "cand"
    if k == "uniform-IQ3_XXS":
        return "stock IQ3_XXS (smaller)" if v["mf"][k]["bytes"] < v["mf"]["UD-IQ3_XXS"]["bytes"] else "stock IQ3_XXS (larger)"
    return LABEL[k]


def mac_table(v: dict, card: bool = False) -> list[str]:
    """Addendum G comparison: all seven files, numbers from the files, compare.py lines and llama-bench output."""
    limit = v["mf"]["UD-IQ3_XXS"]["bytes"]
    out = ["| File | Bytes | Size limit | Off-Metal expert tensors | KL T1 | KL T2 | cand minus file, T1 [95% CI] | "
           "cand minus file, T2 [95% CI] | Top-1 T1 / T2 | pp512 t/s | tg128 t/s |", "|---|" + "---|" * 10]
    c1, c2 = (next(r for (t, _), r in v["mrows"].items() if t == s) for s in ("T1", "T2"))
    for k in v["mnames"]:
        f = v["mf"][k]
        flag = ("sets the limit" if f["bytes"] == limit else "under" if f["bytes"] < limit else
                "over, context only" if k in v["context"] else "over")
        if k == "cand":
            kl1, kl2, d1, d2, top = c1["kc"], c2["kc"], "n/a", "n/a", f"{c1['tc']}% / {c2['tc']}%"
        else:
            r1, r2 = v["mrows"][("T1", k)], v["mrows"][("T2", k)]
            kl1, kl2, top = r1["ko"], r2["ko"], f"{r1['to']}% / {r2['to']}%"
            d1, d2 = (f"{r['d']} [{r['lo']}, {r['hi']}]" for r in (r1, r2))
        sp = " | ".join(f"{f['sp'][t][1]} ± {f['sp'][t][2]}" for t in ("pp512", "tg128"))
        out.append(f"| {mac_label(v, k, card)} | {n(f['bytes'])} | {flag} | {f['off']} | {kl1} | {kl2} | {d1} | {d2} | {top} | {sp} |")
    return out


def mac_facts(v: dict) -> dict:
    mf = v["mf"]
    allm = [k for k in v["mnames"] if mf[k]["off"] == 0]
    sp = {t: [mf[k]["sp"][t][0] for k in allm] for t in ("pp512", "tg128")}
    ud = mf["UD-IQ3_XXS"]["sp"]
    rank = {t: 1 + sum(x > mf["cand"]["sp"][t][0] for x in sp[t]) for t in sp}
    return dict(allm=allm, sp=sp, rank=rank, spread={t: max(x) / min(x) for t, x in sp.items()},
                vs_ud={t: (mf["cand"]["sp"][t][0] / ud[t][0], min(sp[t]) / ud[t][0], max(sp[t]) / ud[t][0]) for t in sp},
                counted_ok=[k for k in v["counted"] if mf[k]["off"] == 0],
                n_under=sum(mf[k]["bytes"] < mf["UD-IQ3_XXS"]["bytes"] for k in v["counted"]))


def speed_par(v: dict) -> list[str]:
    f, mf = mac_facts(v), v["mf"]
    ud = mf["UD-IQ3_XXS"]
    (pc, plo, phi), (tc, tlo, thi) = f["vs_ud"]["pp512"], f["vs_ud"]["tg128"]
    return [f"The {len(f['allm'])} files with all expert tensors in Metal-supported types run at about the same speed as "
            f"each other. Their pp512 lies between {min(f['sp']['pp512']):.2f} and {max(f['sp']['pp512']):.2f} t/s "
            f"(largest over smallest {f['spread']['pp512']:.2f}), and their tg128 between {min(f['sp']['tg128']):.2f} and "
            f"{max(f['sp']['tg128']):.2f} t/s ({f['spread']['tg128']:.2f}). Among them the candidate ranks "
            f"{f['rank']['pp512']} of {len(f['allm'])} on pp512 and {f['rank']['tg128']} of {len(f['allm'])} on tg128.",
            "",
            f"The speed difference is against UD-IQ3_XXS only. Its {ud['off']} NVFP4 expert tensors run off Metal, and it "
            f"measured {ud['sp']['pp512'][1]} t/s pp512 and {ud['sp']['tg128'][1]} t/s tg128. The candidate is {pc:.2f} "
            f"times as fast on pp512 and {tc:.2f} times as fast on tg128. For the all-Metal files the ratios to UD-IQ3_XXS "
            f"run from {plo:.2f} to {phi:.2f} (pp512) and {tlo:.2f} to {thi:.2f} (tg128). This would change if the Metal "
            "backend gained NVFP4 matrix multiplies.",
            "",
            f"Measured with `llama-bench -ngl 99 -p 512 -n 128 -r {v['bench_reps']}` at each file's default loop count, one "
            f"binary for all files (build `{ud['build']}` as printed by llama-bench, the patched HEAD below). The ± is "
            f"llama-bench's standard deviation over the {v['bench_reps']} repetitions. Machine: {v['machine']}. Engine: "
            f"llama.cpp `{v['base'][:7]}` ({v['tag']}) plus {len(v['patches'])} patches "
            f"({', '.join(p[:4] for p, _ in v['patches'])}, listed under Provenance)."
            + ("" if memory_rows() else " Peak memory was not recorded.")]


def stability_lines(v: dict) -> list[str]:
    j, s = v["judge"], v["stab"]
    ok = sum(c["pass"] for c in j["checks"])
    out = ["| Check | Q8_0 reference | Candidate | Rule | Result |", "|---|---|---|---|---|"]
    fmt = lambda x: ("yes" if x else "no") if isinstance(x, bool) else (f"{x:.4f}" if isinstance(x, float) else str(x))
    out += [f"| {c['name']} | {fmt(c['ref'])} | {fmt(c['cand'])} | {c['rule']} | {'pass' if c['pass'] else 'FAIL'} |"
            for c in j["checks"]]
    lc = s["cand"]["long_context"]["prompt_tokens"]
    assert lc == s["ref"]["long_context"]["prompt_tokens"]
    return out + ["", f"**Stability verdict: {'PASS' if j['pass'] else 'FAIL'}** ({ok} of {len(j['checks'])} checks, "
                      "thresholds fixed in `scripts/stability.py` before any candidate existed, applied by its `judge` "
                      f"function). The long prompt has {n(lc)} tokens. {s['cand']['gsm8k']['n']} GSM8K items are a smoke "
                      "test, not an accuracy estimate. Both runs use llama-server's chat endpoint at P = 3."]


def ext_revisions(v: dict) -> list[str]:
    out = ["| File | Repo | File name | Revision | loop.count | MTP tensors |", "|---|---|---|---|---|---|"]
    out += [f"| {mac_label(v, k)} | `{f['repo']}` | `{f['file']}` | `{f['rev']}` | {f['loop']} | {f['mtp']} |"
            for k, f in v["mf"].items() if f["ext"]]
    return out


def mac_rule(v: dict) -> str:
    f = mac_facts(v)
    return (f"The Addendum G rule needs the candidate to have lower KL than each of the {len(v['counted'])} comparators under the "
            f"limit ({', '.join(mac_label(v, k) for k in v['counted'])}) on both T1 and T2, with the 95% CI of the "
            f"difference below 0 and top-1 not worse (`analysis/compare.py`), and each comparator must have all expert "
            f"tensors in Metal-supported types. {len(f['counted_ok'])} of {len(v['counted'])} are confirmed from their "
            f"tensor types. The result in `{v['mq']['name']}` is **{v['mq']['verdict']}**.")


def prior_section(v: dict) -> list[str]:
    pw = v["pw"]
    mark = "**[link: reopen before publishing]**"
    desc = {"LoopQ": "Post-training quantization of weights and activations for looped transformers. It names recursive "
                     "error accumulation as one source of degradation.",
            "Feedback Exposure": "Integer quantization of a recurrent-depth model (Huginn). Hessians summed over "
                                 "recurrence steps beat one-step Hessians. It perturbs by component, always on all steps. It "
                                 "has no condition where the weights differ on one step only, no MoE and no GGUF.",
            "Tilted Bowl": "Tests the accumulation account across several model families. A fixed rounding error does "
                           "not accumulate when the loop settles. It contrasts fixed with fresh noise and runs the last "
                           "loops at higher weight precision, so pass-specific weight precision is already published.",
            "Loopy": "Rankings of candidate low-bit representations change with recurrent depth, so it selects the shared "
                     "representation by final loss at the deployment depth.",
            "LoopQuant": "Allocates activation precision per loop from Shapley-style marginals, with a pairwise term that "
                         "flags stronger-than-additive degradation. That is a per-loop additivity analysis, on activations."}
    out = ["## Prior work", "",
           f"From the prior-work check of {pw['date']} (`{pw['file']}`). Every link below was opened then and must be "
           "reopened and found live before this note is published.", ""]
    for k, d in desc.items():
        p = pw[k]
        out.append(f"- **{k}** ({p['arxiv']}, {p['date']}), [{p['url']}]({p['url']}) {mark}. {d}")
    lq = pw["LookAtMyQuant"]
    out += [f"- **LookAtMyQuant** (`{lq['repo']}`, created {lq['date']}), [{lq['url']}]({lq['url']}) {mark}. A Mini "
            f"quantization whose card says it \"{pw['lamq_quotes'][0]}\" and then ran QAT with "
            f"\"{pw['lamq_quotes'][1]}\". The card shows no table by visit, no single-pass intervention, no additivity "
            "test and no control without weight sharing.", "",
            "Our per-pass test differs from these in design. It applies the same quantized expert block on one pass "
            "only, compares the sum of single-pass changes with the all-pass change, and sets that against a layer split "
            "that shares no weights.", ""]
    return out


def render(v: dict) -> tuple[str, str]:
    ud_bytes = int(v["files"]["UD-IQ3_XXS"]["bytes"].replace(",", ""))
    stock_key = next(k for k in v["files"] if k.startswith("stock IQ3_XXS"))
    stock_bytes = int(v["files"][stock_key]["bytes"].replace(",", ""))
    stock = "stock IQ3_XXS (smaller)" if stock_bytes < ud_bytes else "stock IQ3_XXS (larger)"
    label = {"UD-IQ3_XXS": "UD-IQ3_XXS", "uniform-IQ3_XXS": stock}
    cand_bytes = int(v["files"]["cand"]["bytes"].replace(",", ""))
    s = v["sel"][v["chosen"]]
    passed = v["verdict"] == "PASS"
    ud = [g for g in v["gate"] if g["base"] == "UD-IQ3_XXS"]
    st = [g for g in v["gate"] if g["base"] == "uniform-IQ3_XXS"]
    uses_ours = [x["i"] for x in v["sel"] if any(d == "Q2_0O" for _, d in x["choice"].values())]
    cand_ours = v["chosen"] in uses_ours

    # selection facts
    ratios = [x["meas"] / x["pred"] for x in v["sel"]]
    by_pred = min(v["sel"], key=lambda x: x["pred"])["i"]
    by_meas = sorted(v["sel"], key=lambda x: x["meas"])
    over = [x["fbytes"] - x["abytes"] for x in v["sel"]]

    pp, ls, b3 = v["pp"]["fisher"], v["ls"]["fisher"], v["b3"]["fisher"]
    tests = {k: pr.apart(v["pp"], v["ls"], "fisher", k) for k in ("r", "norm_ratio")}
    licensed = all(d and p for d, p in tests.values())
    licensed_head = all(d and p for d, p in (pr.apart(v["pp"], v["ls"], "head_euclid", k) for k in ("r", "norm_ratio")))
    licensed_full = all(d and p for d, p in (pr.apart(v["pp"], v["ls"], "full_euclid", k) for k in ("r", "norm_ratio")))
    fv = v["pp"]["full_euclid"]
    tk = [float(x.mean()) for x in v["tied"].values()]
    tr_lo, tr_hi = min(t["ratio"] for t in v["tied_ratio"].values()), max(t["ratio"] for t in v["tied_ratio"].values())
    closer = pp["r"] < ls["r"] and abs(pp["norm_ratio"] - 1) < abs(ls["norm_ratio"] - 1)
    rung = {k: pr.apart(v["pp"], v["b3"], "fisher", k) for k in ("r", "norm_ratio")}
    rung_moved = all(d and p for d, p in rung.values())
    kl = v["ppkl"]
    kr = v["kl_ratio"]
    ls_sum = v["kl2"]["ls67"] + v["kl2"]["ls89"] + v["kl2"]["ls1011"]
    b3_sum = v["kl2"]["3b-p1"] + v["kl2"]["3b-p2"] + v["kl2"]["3b-p3"]
    all_kl = float(kl[("all", "pp-base")])
    kl_drop = all_kl / v["kl2"]["3b-all"]
    lo, hi = v["noise"]["ci95"]
    b2a, b2b = v["b2_layers"]

    # ---------------- note ----------------
    N = [f"# Counting passes in a looped MoE: pilot measurements on Inkling 2 Mini", "",
         "*Exploratory pilot. Draft, not published. Every number here was rendered by `scripts/card.py` from the "
         "pilot's result files.*", "",
         "## Summary", "",
         f"Inkling 2 Mini runs its looped core (layers {v['blocks'][1]['Layers'].split(',')[0]} and "
         f"{v['blocks'][2]['Layers'].split(',')[0]}) P times on the same expert weights. We asked two things. Can a "
         "block-wise bit allocation, chosen from measurements, beat a public 3-bit GGUF at no more bytes? And how "
         "does quantization error on one looped block behave when it is applied on one pass, or on all three?", "",
         f"- **The release gate {'passed' if passed else 'failed'} on quality.** Our candidate "
         f"({n(cand_bytes)} bytes) has higher KL to the reference than UD-IQ3_XXS ({n(ud_bytes)} bytes) on both "
         f"held-out sets: {ud[0]['d']:+.5f} [{ud[0]['lo']:+.5f}, {ud[0]['hi']:+.5f}] on T1 and {ud[1]['d']:+.5f} "
         f"[{ud[1]['lo']:+.5f}, {ud[1]['hi']:+.5f}] on T2. It is worse on {ud[0]['worse']} of {ud[0]['n']} T1 chunks "
         f"and {ud[1]['worse']} of {ud[1]['n']} T2 chunks.",
         f"- It beats {stock}, {n(stock_bytes)} bytes, on both sets, by {st[0]['d']:+.5f} on T1 and "
         f"{st[1]['d']:+.5f} on T2 (both CIs below zero).",
         "- Under our pre-registered rule a failed gate means no model is released on that gate. The FAIL stands.",
         *([f"- After the FAIL we registered a narrower comparison (Addendum G, post hoc): files under the limit with all "
            f"expert tensors in Metal-supported types. Against all {len(v['counted'])} of them the candidate has lower KL "
            f"on T1 and T2, CI below zero. That rule gives {v['mq']['verdict']}. A card for this narrower claim is "
            "drafted for review, not published."] if v["mac"] else []),
         f"- Per-pass test on block B2 at the 2-bit level: the change from all three passes is not the sum of the "
         f"single-pass changes (Fisher r = {pp['r']:.3f}). A layer-split control with no weight sharing gives "
         f"r = {ls['r']:.3f}. " + ("The CIs for r and for the norm ratio do not overlap, and on these two statistics the pass split is closer to "
                                   "additive than the layer split." if licensed and closer else
                                   "The rule for a claim about the loop is not met."), "",
         "## Setup", ""]
    N += setup(v) + ["", "S is the selection set. T1 and T2 are the held-out test sets, scored only after the candidate was fixed.", ""]
    N += data_table(v) + ["", "Engine and files:", ""] + provenance(v)
    N += ["", f"Our llama.cpp patches make every loop pass read the expert scale tensors of its physical layer, let a "
          f"pass carry its own expert tensors (used for the per-pass test), and speed up the BF16 conversion. The "
          f"build runs {v['loop_tests']} loop checks, and all pass (`results/env.md`). {v['loop_zero']} of them match "
          f"with nmse 0; the largest nmse among the rest is {v['loop_max_nmse']:.1e}.", "",
          "## Floors", "",
          f"- Q8_0 reference against itself: KL {v['kl']['floor-q8-S-P3']:.5f}, top-1 {v['top']['floor-q8-S-P3']:.2f}%.",
          f"- Our IQ4_NL build against the reference: KL {v['kl']['floor-iq4nl-S-P3']:.5f} on S. The stop band was "
          f"{v['floor_band'][0]} to {v['floor_band'][1]}.", "",
          f"## Block map (S, P = {v['P']})", "",
          "One block's experts changed, everything else as in the reference. KL on S.", "",
          "| Block | Layers | 2-bit (IQ2_XXS / Q2_0) | IQ2_XXS / IQ4_NL | IQ2_S / IQ4_NL | IQ3_XXS / Q2_0 (ours) | "
          "3-bit (IQ3_XXS / IQ4_NL) |", "|---|---|---|---|---|---|---|"]
    for b in v["blocks"]:
        k = b["Block"]
        N.append(f"| {k} | {b['Layers']} | " + " | ".join(f"{v['kl'][f'map-{k}-{lv}-S-P3']:.5f}"
                                                          for lv in ("2bit", "gu2xxs", "gu2s", "dnq2o", "3bit")) + " |")
    N += ["", f"Non-expert tensors at Q6_K, experts as in the reference: KL {v['kl']['map-NE-Q6K-S-P3']:.5f}.",
          "Additivity of the map: " + "; ".join(f"{a}-bit sum of blocks {s_} against the uniform build {u} "
                                                f"(uniform / sum {r_})"
                                                for a, s_, u, r_ in v["additivity"]) + ".", "",
          "## Pass sweep (S)", "",
          "Uniform builds, each scored against the reference at the same P. P = 2 and P = 4 are outside what the "
          "model was tuned for. We draw no causal claim about the loop from this sweep.", "",
          "| Level | " + " | ".join(k for k in v["sweep"][0] if k != "Level") + " |",
          "|---|" + "---|" * (len(v["sweep"][0]) - 1)]
    N += [f"| {r['Level']} | " + " | ".join(r[k] for k in r if k != "Level") + " |" for r in v["sweep"]]
    N += ["", "Run time against layer visits:", ""]
    N += [f"- P = {p}: {sec} s, {rt} of the P = 3 time, visits ratio {vr}." for p, sec, rt, vr in v["timing"]]
    N += ["", "## Selection on S", "",
          f"The allocator added per-projection changes from the map under the byte limit and proposed {len(v['sel'])} "
          f"allocations. Each was built by splicing tensors from pre-quantized pools, then scored on S. The lowest "
          "measured KL became the candidate before T1 or T2 were scored.", "",
          "| Pick | Allocation (gate_up / down) | Non-expert | Predicted KL | Measured KL ± SE | Measured / predicted | "
          "File bytes |", "|---|---|---|---|---|---|---|"]
    for x, rt in zip(v["sel"], ratios):
        N.append(f"| sel-{x['i']}{' (chosen)' if x['i'] == v['chosen'] else ''} | {choice(x['choice'])} | {x['ne']} | "
                 f"{x['pred']:.5f} | {x['meas']:.5f} ± {x['se']:.5f} | {rt:.2f} | {n(x['fbytes'])} |")
    N += ["", f"- The allocator overpredicted {sum(r < 1 for r in ratios)} of {len(ratios)} picks "
          f"(measured / predicted {min(ratios):.2f} to {max(ratios):.2f}).",
          f"- Its top pick, sel-{by_pred}, measured {'worst' if by_meas[-1]['i'] == by_pred else 'not lowest'}. "
          f"The order by measured KL is {', '.join(f'sel-{x['i']}' for x in by_meas)}.",
          f"- Each file is {min(over):,} to {max(over):,} bytes larger than the allocator's count. "
          f"{sum(x['fbytes'] <= ud_bytes for x in v['sel'])} of {len(v['sel'])} stayed under the limit of {n(ud_bytes)} bytes.",
          f"- UD-IQ3_XXS scored {v['kl']['UD-IQ3_XXS-S-P3']:.5f} on S, "
          f"{'below every pick' if all(v['kl']['UD-IQ3_XXS-S-P3'] < x['meas'] for x in v['sel']) else 'not below every pick'}. Its S score was not used "
          "for selection.", "",
          f"## Release gate (T1, T2, P = {v['P']})", "",
          "The gate needs the candidate's paired KL to beat both baselines on both sets, with the 95% CI excluding "
          "zero, and top-1 agreement no worse. Negative differences favour the candidate.", ""]
    N += gate_table(v, label)
    N += ["", f"**Verdict: {v['verdict']}.**", "",
          f"- The plain IQ3_XXS build has {n(stock_bytes)} bytes, {'fewer' if stock_bytes < ud_bytes else 'more'} than "
          f"UD-IQ3_XXS, so it is labelled {stock}.",
          (f"- Teacher check: not run. It applies when a gap to UD-IQ3_XXS is below {v['teacher_margin']} KL. Both gaps "
           f"go against the candidate and are larger ({ud[0]['d']:+.5f} on T1, {ud[1]['d']:+.5f} on T2)."
           if all(g["d"] > 0 and g["d"] >= float(v["teacher_margin"]) for g in ud) else
           "- Teacher check: required by the rule but not run. The gate cannot be read until it is."),
          "- Stability battery and speed (`llama-bench`): " + (
              "skipped by the original protocol after a quality FAIL, then run by script as registered steps under "
              "Addendum G (section below)." if v["mac"] and v["stability_run"] and v["bench_run"] else
              "run." if v["stability_run"] and v["bench_run"] else
              "not run under the protocol, which skips them after a quality FAIL.")
          + (f" {len(v['explore_bench'])} off-protocol `llama-bench` runs were made by hand after the FAIL; their "
             "figures are not reported." if v["explore_bench"] else ""),
          "", "Expert tensors in types the pinned Metal backend cannot multiply, read from each file's tensor types:", "",
          "| Build | Bytes | Expert tensor types | Off-Metal expert tensors |", "|---|---|---|---|"]
    for k, r in v["files"].items():
        name = stock if k == stock_key else k
        N.append(f"| {name} | {r['bytes']} | {r['expert tensor types']} | {v['metal'][k]} |")
    N += ["", "UD-IQ3_XXS keeps the release's NVFP4 down projections in "
          f"{v['metal']['UD-IQ3_XXS']} expert tensors. The tensor count says where those multiplies cannot run on stock "
          "llama.cpp. Unsloth's own prebuild (" + REL + ") includes Metal NVFP4 kernels (" + PR + "); with it, "
          "the NVFP4 tensors of Unsloth's root GGUF load on the Mac GPU (checked with UD-IQ2_XXS, 22 Nov). It is "
          "not a speed measurement. " + ("Speed is measured and reported under Addendum G below." if v["mac"] else
                                         "We report no speed figures."), ""]
    if v["mac"]:
        N += [f"## Addendum G: Mac track (T1, T2, P = {v['P']})", "",
              f"Registered on {v['g_date']} at about {v['g_time']}, after the gate above had failed and before any of "
              f"these numbers existed. Two of the files were added by an amendment at about {v['g_amend']}, before any "
              "download or score. This comparison is post hoc. It does not change the FAIL.", "",
              mac_rule(v), "",
              "Bytes are read from each file. The off-Metal count is read from each file's tensor types. Differences are "
              "candidate minus file, so negative values favour the candidate. Speed is llama-bench mean ± standard "
              "deviation.", ""]
        N += mac_table(v) + [""] + speed_par(v) + ["", "Stability battery of the candidate against the Q8_0 reference:", ""]
        N += stability_lines(v) + ["", "Outside files:", ""] + ext_revisions(v) + [""]
    N += ["## Our Q2_0 encoder", "",
          "llama.cpp's reference Q2_0 encoder sets the block scale from the largest weight, so one level goes unused, "
          "and it ignores the importance matrix. We wrote an imatrix-weighted Q2_0 encoding (our encoder). It "
          "searches signed scales over all four levels and refits each block's scale by weighted least squares. The "
          "format, kernels and bytes do not change.", ""]
    for k, (a, b, c) in v["q2"].items():
        N.append(f"- {k}: mean relative l2 weight error {a:.4f} (stock) against {b:.4f} (ours), over {c} layers.")
    N += [f"- Picks {' and '.join(f'sel-{i}' for i in uses_ours)} used it for block B1's down projections. "
          + ("The chosen pick uses it." if cand_ours else "The chosen pick does not use it."), "",
          "## Per-pass test (B2, 2-bit level, S, P = 3)", "",
          f"On these documents, at P = {v['P']}, changing B2's experts on pass k produced the measured logit changes "
          f"below. B2 is layers {b2a}-{b2b}. The base is our IQ4_NL build. The 2-bit level is IQ2_XXS gate_up "
          "with stock Q2_0 down. Its mean relative weight error against the IQ4_NL base is "
          + " and ".join(f"{t:.4f} on {k}" for k, t in v["t2bit"].items())
          + ". Every number in this section holds for that amplitude only.", "",
          ("A no-op splice reproduced the base logits byte for byte, and switching all three passes reproduced an "
           "ordinary build with the same tensors." if v["parity"] else "The parity checks did not pass."), "",
          "| Condition | KL vs base | KL vs Q8_0 reference |", "|---|---|---|"]
    for c_ in ("p1", "p2", "p3", "all"):
        N.append(f"| pass {c_[1]} only | {float(kl[(c_, 'pp-base')]):.5f} | {float(kl[(c_, 'ref-S-P3')]):.5f} |"
                 if c_ != "all" else f"| all three passes | {all_kl:.5f} | {float(kl[(c_, 'ref-S-P3')]):.5f} |")
    N += ["", "We compare d_all, the change in log-probabilities from all three passes, with the sum of the three "
          "single-pass changes. Norms are Fisher-weighted except in the last column, which uses centred Euclidean over "
          "the full vocabulary with the floored entries kept as written (the sensitivity check of Addendum G section 2). "
          "r = ||d_all - sum_k d_k|| / ||d_all||.", "",
          "| Statistic | Pass split, 2-bit | Layer split, 2-bit (control) | Pass split, 3-bit | Pass split, 2-bit, full "
          "vocabulary |", "|---|---|---|---|---|"]
    for k, name in (("r", "r"), ("norm_ratio", "||d_all|| / ||sum_k d_k||"), ("sq_over_sum", "||d_all||^2 / sum_k ||d_k||^2"),
                    ("cos_12", "cos(d1, d2)"), ("cos_23", "cos(d2, d3)"), ("cos_13", "cos(d1, d3)")):
        N.append(f"| {name} | {ci(pp, k)} | {ci(ls, k)} | {ci(b3, k)} | {ci(fv, k)} |")
    N.append(f"| KL all / sum of parts | {kr['ratio']:.3f} [{kr['ci95'][0]:.3f}, {kr['ci95'][1]:.3f}] | "
             f"{v['kl2']['all'] / ls_sum if 'all' in v['kl2'] else all_kl / ls_sum:.3f} (point only) | "
             f"{v['kl2']['3b-all'] / b3_sum:.3f} (point only) | n/a (KL has no norm) |")
    N += ["", f"- The layer split puts the 2-bit block on layers {', '.join(v['ls_pairs'][:-1])} or {v['ls_pairs'][-1]} only, on all passes. Its three "
          "parts share no weights, so it shows how far from additive any three-way split of B2 is.",
          f"- The 3-bit rung changes gate_up only (IQ3_XXS). In `t-blocks.tsv`, which uses a different base from the "
          f"figures above, its mean relative error on B2 is {v['t3bit']:.4f}, against {v['t2gu_tb']:.4f} for the 2-bit "
          "gate_up. It has no layer-split control of its own.",
          f"- The additivity comparators disagree. KL all / sum is {kr['ratio']:.3f}, above 1. The Fisher norm ratio "
          f"is {pp['norm_ratio']:.3f}, below 1. We do not choose among them here.", "",
          "What the controls allow, by the rule written in the change log before the controls ran (r and the norm "
          "ratio must both differ beyond both 95% CIs). We apply it under the Fisher norm, which was fixed in code "
          "before the per-pass logits existed. The head Euclidean substitute "
          + ("gives the same verdict" if licensed_head == licensed else "gives a different verdict")
          + (", and so does the full-vocabulary form against the layer split's full-vocabulary values."
             if licensed_full == licensed else ". The full-vocabulary form gives a different verdict."), ""]
    if licensed:
        N += ["- At the 2-bit level on B2, splitting the change across passes on shared weights gives a different r "
              "and norm ratio than splitting it across layer pairs. "
              + ("On these two statistics the pass split is closer to additive." if closer else
                 "On these two statistics the pass split is farther from additive.")]
    else:
        N += ["- The controls license no claim about the loop. The pass split is not shown to differ from a layer "
              "split with no weight sharing."]
    N += ["- They do not license a classification of the error as counting once, per pass or growing across passes. "
          "They say nothing about B3 or other levels.",
          "- " + ("The 3-bit r and norm ratio differ from the 2-bit values beyond both CIs, so these statistics depend "
                  "on the level." if rung_moved else "The 3-bit values are not shown to differ from the 2-bit ones."),
          f"- At the 3-bit rung the all-pass KL against the base is {v['kl2']['3b-all']:.5f}, about {kl_drop:.1f} times "
          f"lower than the {all_kl:.5f} at 2 bits. Fisher r there is {b3['r']:.3f}, against {pp['r']:.3f} at 2 bits. So "
          "the residual does not shrink in proportion to the amplitude.",
          "", "Gaussian noise of the same relative error as the 2-bit level, on B2. A tied condition puts one draw on "
          f"all three passes; there are {len(v['tied'])} tied draws. The independent condition puts draws 1, 2 and 3 "
          "on passes 1, 2 and 3, and it still has one draw. KL against the base, and each tied draw over the independent one (paired bootstrap over chunks):", "",
          "| Condition | KL vs base | Tied / independent [95% CI] |", "|---|---|---|"]
    N += [f"| tied, draw {d} | {float(x.mean()):.5f} | {t['ratio']:.3f} [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}] |"
          for (d, x), t in zip(sorted(v["tied"].items()), (v["tied_ratio"][d] for d in sorted(v["tied"])))]
    N += [f"| independent | {v['noise']['kl_indep']:.5f} | n/a |", "",
          f"- Tied over independent runs from {tr_lo:.3f} to {tr_hi:.3f} across the {len(v['tied'])} tied draws. The tied "
          f"KLs span {min(tk):.5f} to {max(tk):.5f}.",
          "- All ratios share the one independent draw, so the draw effect on the independent side is not measured. "
          "Each CI covers chunk sampling only.", "",
          f"Three analysis choices differ from the registered plan: the Fisher norm as primary, a {v['head_nats']}-nat head for the "
          "Euclidean substitute, and a pooled ratio of sums instead of a per-token ratio. They were fixed in code "
          f"in commit `{v['dev_commit']}` at {v['dev_time']}. The per-pass logits came at {v['dev_logits']}. They "
          "now have a dated entry in the change log, Addendum G section 2"
          + (", awaiting approval." if v["dev_open"] else "."), "",
          "The per-pass test is a scientific result, not the reason for the release allocation. The block map measures "
          "looped blocks with all passes tied.", ""]
    N += prior_section(v)
    N += ["## Limits", "",
          "- This is an exploratory pilot. None of it is confirmatory evidence.",
          "- One reference, one importance matrix, one machine, one engine commit.",
          f"- T2 has {v['ctx']['T2'][0]} chunks, so its bootstrap is coarse.",
          "- KL to a reference is not task quality. We report no task scores.",
          "- The per-pass test covers block B2 at two levels. B3 was not perturbed per pass.",
          *(["- The Mac-track comparison was registered after the gate failed. It is post hoc. Its speed figures hold "
             "for one machine and one engine commit."] if v["mac"] else []), ""]

    # ---------------- card ----------------
    cand_types = v["files"]["cand"]["expert tensor types"]
    C = ["---", f"base_model: {v['model']}", "tags:", "- gguf", "- llama.cpp", "- exploratory", "---", ""]
    if not passed:
        C += ["> **NOT FOR UPLOAD.** Draft stub only. The release gate failed on quality, so under the pre-registered "
              "rule no model is published. The measurements are in the research note.", ""]
    C += [f"# Inkling 2 Mini GGUF, block-wise allocation (exploratory)", "",
          f"Block-wise allocation chosen from a measured {len(v['blocks'])}-block table at P = {v['P']}. Exploratory.", "",
          "## File", "",
          f"- {n(cand_bytes)} bytes. Expert tensor types: {cand_types}. Non-expert tensors: "
          f"{s['ne']}.",
          f"- Allocation (gate_up / down): {choice(s['choice'])}.",
          f"- {'Uses' if cand_ours else 'Uses no tensors from'} our imatrix-weighted Q2_0 encoding (our encoder).",
          f"- Expert tensors in types the pinned Metal backend cannot multiply: {v['metal']['cand']}. No speed figures "
          "are reported.", "",
          f"## Gate result: {v['verdict']}", "",
          f"KL against a Q8_0 reference converted from the release, at P = {v['P']}, paired bootstrap over chunks.", ""]
    C += gate_table(v, label)
    C += ["", "Stability and speed were not run under the protocol, which skips them after a quality FAIL.", "",
          "## Setup", ""] + setup(v) + ["", "Engine and files:", ""] + provenance(v) + [""]
    if v["mac"]:
        C = mac_card(v, label, cand_bytes, ud_bytes, ud, s, cand_ours)
    return "\n".join(N) + "\n", "\n".join(C) + "\n"


def base_license() -> str:
    """The release's license id, read from its model card front matter."""
    return one(r"^license: (\S+)$", text(ROOT / "models/Inkling-2-Mini/README.md"), re.M)[1]


def upstream_check() -> dict | None:
    """Stock upstream llama.cpp rescore of the candidate (results/pilot/upstream-check.txt)."""
    p = R / "upstream-check.txt"
    if not p.exists():
        return None
    t = text(p)
    kl = re.findall(r"Mean KLD: ([\d.]+) ± ([\d.]+) \| Same top p: ([\d.]+)", t)
    assert len(kl) == 2, "upstream-check.txt: expected two runs"
    return dict(commit=one(r"stock upstream ([0-9a-f]+)", t)[1], same=kl[0] == kl[1], kl=kl[1][0], top=kl[1][2])


DEPTH_FILES = ("cand", "UD-IQ3_XXS", "atomic-AD-IQ2_S")


def depth_rows() -> dict:
    """llama-bench at context depth (results/pilot/depth-NAME.txt): {name: {test: (t/s, ±)}}; files missing are skipped."""
    out = {}
    for name in DEPTH_FILES:
        p = R / f"depth-{name}.txt"
        if p.exists():
            out[name] = {m[1].strip(): (m[2], m[3]) for m in re.finditer(r"\|\s*((?:pp512|tg128)(?: @ d\d+)?)\s*\|\s*([\d.]+) ± ([\d.]+)\s*\|", text(p))}
    return out


def memory_rows() -> dict:
    """Peak memory from /usr/bin/time -l (results/pilot/memory-NAME.txt), in bytes."""
    out = {}
    for name in DEPTH_FILES:
        p = R / f"memory-{name}.txt"
        if p.exists():
            t = text(p)
            out[name] = dict(rss=int(one(r"(\d+)\s+maximum resident set size", t)[1]),
                             foot=int(one(r"(\d+)\s+peak memory footprint", t)[1]))
    return out


def depth_section(v: dict) -> list[str]:
    d, mem = depth_rows(), memory_rows()
    if not d and not mem:
        return []
    out = ["", "## Speed at context depth and peak memory", ""]
    if d:
        tests = sorted({t for r in d.values() for t in r}, key=lambda t: (t[:5], int(t.split("d")[-1]) if "@" in t else 0))
        names = [k for k in DEPTH_FILES if k in d]
        out += ["Same binary and flags as above, `-d 0,4096,16384 -r 2`: each test runs after that many tokens of context.", "",
                "| Test | " + " | ".join(mac_label(v, k, True) for k in names) + " |", "|---|" + "---|" * len(names)]
        out += [f"| {t} | " + " | ".join(f"{d[k][t][0]} ± {d[k][t][1]}" if t in d[k] else "n/a" for k in names) + " |" for t in tests]
    if mem:
        out += ["", "Peak memory of one llama-bench run at depth 16,384 (macOS `/usr/bin/time -l`). The model file is "
                "memory-mapped, so it counts in the resident set size, not in the footprint:", "",
                "| File | Peak memory footprint | Maximum resident set size |", "|---|---|---|"]
        out += [f"| {mac_label(v, k, True)} | {mem[k]['foot'] / 2**30:.2f} GiB | {mem[k]['rss'] / 2**30:.2f} GiB |" for k in DEPTH_FILES if k in mem]
    return out


NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
LARGER = {"uniform-IQ3_XXS": "a stock IQ3_XXS build"}


def join_and(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def day(iso: str) -> str:
    """2026-11-21 -> 21 Nov 2026"""
    import datetime
    d = datetime.date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%b')} {d.year}"


def gap_lines() -> list[str]:
    """Addendum H, H0: the 2 x 2 on S that locates the gap to UD-IQ3_XXS (layout or down format), if it has run."""
    kl = {r["run"]: (float(r["mean_kl"]), float(r["se"])) for r in csv.DictReader(open(R / "kl.tsv"), delimiter="\t")}
    need = ("sel-2-S-P3", "UD-IQ3_XXS-S-P3", "hybrid-S-P3", "mirror-S-P3")
    if not all(k in kl for k in need):
        return []
    c, u, h, m = (kl[k][0] for k in need)
    return [f"Where the gap comes from (Addendum H, on S). Swapping only the down projections moves little: this file "
            f"with UD's NVFP4 down scores {h:.5f} against {c:.5f}, and UD's file with our IQ4_NL down scores {m:.5f} "
            f"against UD's {u:.5f}. Most of the gap ({c - m:.5f} of {c - u:.5f}) comes from UD's layout, not from the "
            "down format.", ""]


def lean_gap() -> str:
    g = gap_lines()
    return (" A swap test on S shows that most of the gap comes from Unsloth's layout choices, not from their NVFP4 "
            "down tensors (see [DETAILS.md](DETAILS.md)).") if g else ""


def depth_gap() -> str:
    """One sentence on how the speed gap to UD-IQ3_XXS changes with a 16K context, if the depth bench exists."""
    d = depth_rows()
    if "cand" not in d or "UD-IQ3_XXS" not in d:
        return ""
    r = {t: float(d["cand"][t][0]) / float(d["UD-IQ3_XXS"][t][0]) for t in ("pp512 @ d16384", "tg128 @ d16384")}
    return (f" With 16K tokens already in context the gap narrows to {r['pp512 @ d16384']:.1f}× and "
            f"{r['tg128 @ d16384']:.2f}×.")


def claim_par(v: dict) -> list[str]:
    f, mf = mac_facts(v), v["mf"]
    limit = mf["UD-IQ3_XXS"]["bytes"]
    under = [k for k in f["counted_ok"] if mf[k]["bytes"] < limit]
    over = [k for k in v["mnames"] if k != "cand" and mf[k]["off"] == 0 and mf[k]["bytes"] >= limit]
    won = lambda k: all(v["mrows"][(t, k)]["win"] for t in ("T1", "T2"))  # noqa: E731
    assert all(won(k) for k in under), "claim needs a win over every under-limit comparator"
    big = [k for k in over if won(k)]
    (pc, _, _), (tc, _, _) = f["vs_ud"]["pp512"], f["vs_ud"]["tg128"]
    return ["## In short", "",
            f"- Of the GGUFs under {limit / 1e9:.1f} GB that we tested and that run entirely on the Mac GPU, this one stays "
            f"closest to the original model. It has lower KL on held-out text than {join_and([mac_label(v, k, True) for k in under])}, "
            f"and also than {NUM.get(len(big), len(big))} larger file{'s' if len(big) != 1 else ''} "
            f"({join_and([LARGER.get(k, mac_label(v, k, True)) for k in big])}). Tested on {day(v['g_date'])}, so later "
            "uploads aren't covered.",
            f"- Unsloth's UD-IQ3_XXS stays closer to the original. With stock llama.cpp on a Mac it's {pc:.1f}× slower at "
            f"reading prompts and {tc:.2f}× slower at generating, because 20 of its expert tensors run on the CPU there.",
            "- All-Metal files of this size run at about the same speed as each other (table below).", ""]


def usage_section(v: dict) -> list[str]:
    u = upstream_check()
    out = ["## Use", "",
           "Needs a llama.cpp build with Inkling 2 support (upstream after 20 Nov 2026). Homebrew's llama.cpp releases "
           "of that date do not load it yet.", "",
           "```sh",
           "llama-cli -m Inkling-2-Mini-Metal.gguf -ngl 99 --jinja -c 32768 --temp 1.0 --reasoning-budget 16384",
           "llama-server -m Inkling-2-Mini-Metal.gguf -ngl 99 --jinja -c 32768 --temp 1.0 --reasoning-budget 16384 --port 8080",
           "```", "",
           "Temperature 1.0 is the release's `generation_config.json` value. The base model card warns about long "
           "reasoning loops, so the commands cap thinking at 16K tokens. The loop count stays at its default of 3."]
    if u:
        out += ["", f"Stock upstream llama.cpp (`{u['commit']}`, no patches) scores this file "
                + ("identically" if u["same"] else "differently") + f" to the build used for the tables (T1: KL {u['kl']}, "
                f"top-1 {u['top']}%)."]
    return out + [""]


def lean_card(v: dict, cand_bytes: int) -> list[str]:
    """Short release README: what the file is, what is different, one table, use, research, limits, provenance.
    The full numbers go to DETAILS.md (the Addendum G card)."""
    f, mf = mac_facts(v), v["mf"]
    limit = mf["UD-IQ3_XXS"]["bytes"]
    under = [k for k in f["counted_ok"] if mf[k]["bytes"] < limit]
    nxt = min(under, key=lambda k: float(v["mrows"][("T1", k)]["ko"]))  # next-lowest-KL all-Metal file
    ud = mf["UD-IQ3_XXS"]
    c1 = next(r for (t, _), r in v["mrows"].items() if t == "T1")
    u = upstream_check()

    def row(k: str, kl: str) -> str:
        name = "**This file**" if k == "cand" else mac_label(v, k, True)
        return (f"| {name} | {mf[k]['bytes'] / 1e9:.2f} GB | {kl} | {'yes' if mf[k]['off'] == 0 else 'no'} | "
                f"{float(mf[k]['sp']['pp512'][1]):.0f} | {float(mf[k]['sp']['tg128'][1]):.1f} |")

    nb = len(v["blocks"])
    C = ["---", f"base_model: {v['model']}", "base_model_relation: quantized", f"license: {base_license()}", "tags:",
         "- gguf", "- llama.cpp", "- inkling2_loop", "- apple-silicon", "---", "",
         f"# Inkling 2 Mini, Metal GGUF ({cand_bytes / 1e9:.1f} GB)", "",
         "![Inkling 2 Mini runs layers 6 to 17 three times on the same expert weights](figures/hero.svg)", "",
         "Unsloth's UD-IQ3_XXS stays closer to the original model (KL "
         f"{v['mrows'][('T1', 'UD-IQ3_XXS')]['d']} on T1 and {v['mrows'][('T2', 'UD-IQ3_XXS')]['d']} on T2, our file "
         "minus theirs), so the quality test we registered against it beforehand did not pass. With stock llama.cpp, this "
         f"GGUF keeps the whole model on the Mac GPU instead, and reads prompts {f['vs_ud']['pp512'][0]:.1f}× faster there at "
         "the moment.", "",
         "## How we made and tested it", "",
         "- Fully on Metal: every expert tensor uses a type stock llama.cpp can run on the Mac GPU. Unsloth's UD-IQ3_XXS "
         f"has {ud['off']} NVFP4 expert tensors that fall back to the CPU on stock llama.cpp.",
         f"- Layout from measurement: we measured the cost of shrinking each of {nb} layer groups, including the core "
         f"that runs three times, and chose the layout from {len(v['sel'])} tested candidates.",
         "- Tested in the open: before release we registered a quality test against Unsloth's file. Our file failed "
         "it. We then compared it with the other all-Metal GGUFs of similar size on held-out text.", "",
         "## Results", "",
         "Lower KL on held-out text means the file is closer to the original model. We measured prompt and generation "
         f"speeds on an {v['machine'].split(',')[0]}.", "",
         "| File | Size | KL | Fully on Metal | Prompt t/s | Generation t/s |", "|---|---|---|---|---|---|",
         row("cand", c1["kc"]), row(nxt, v["mrows"][("T1", nxt)]["ko"]), row("UD-IQ3_XXS", v["mrows"][("T1", "UD-IQ3_XXS")]["ko"]), "",
         "![Held-out KL of every file we compared](figures/mac-comparison.png)", "",
         "![Speed against KL on the Mac](figures/speed-vs-kl.png)", "",
         "![Speed as the context fills up](figures/depth-speed.png)", "",
         f"Of the {len(under) + 1} all-Metal GGUFs under {limit / 1e9:.1f} GB we tested on {day(v['g_date'])}, this file "
         f"had the lowest KL. Unsloth's file was closer to the original, but on our Mac it was "
         f"{f['vs_ud']['pp512'][0]:.1f}× slower at reading prompts and {f['vs_ud']['tg128'][0]:.2f}× slower at "
         "generating text." + depth_gap() + lean_gap() + " For all files, both test sets, confidence intervals and stability checks, "
         "see [DETAILS.md](DETAILS.md).", "",
         "## Use", "",
         "Use a llama.cpp build with Inkling 2 support (upstream after 20 Nov 2026).", "",
         "```sh",
         "llama-cli -m Inkling-2-Mini-Metal.gguf -ngl 99 --jinja -c 32768 --temp 1.0 --reasoning-budget 16384",
         "```", "",
         "Temperature 1.0 is the release default. The base model card warns about long reasoning loops, so the "
         "command caps thinking at 16K tokens."
         + (f" Stock upstream llama.cpp (`{u['commit']}`) gives identical scores." if u and u["same"] else ""), "",
         "## Which Mac", "",
         "- 24 GB or more: fits comfortably, with room for a 16K context and other apps.",
         "- 16 GB: possible after raising the GPU memory limit, with a short context and little RAM left for other "
         "apps. macOS lets the GPU use only about two thirds of RAM by default, which is less than the model needs:", "",
         "```sh",
         "sudo sysctl iogpu.wired_limit_mb=13000   # resets on reboot",
         "```", ""]
    mem = memory_rows().get("cand")
    if mem:
        C += [f"Measured on our Mac, the whole process peaked at {mem['rss'] / 1e9:.1f} GB with the context filled to "
              "16K tokens.", ""]
    C += ["## Research background", "",
         "We made this file as part of a master's thesis pilot on looped models. Inkling 2 Mini runs layers 6 to 17 "
         "three times with the same weights. We enabled quantization error on one pass at a time to measure how the "
         "errors add up across passes. We chose the file's layout independently of it.", "",
         "![How the per-pass test is built](figures/loop-diagram.svg)", "",
         "![The pass split against a layer-split control](figures/pass-vs-layer-split.png)", "",
         "- Each pass costs about the same, and all three together cost more than the sum of the single passes.",
         "- Splitting the same block by layer instead of by pass gives a measurably different residual (the intervals "
         "don't overlap). Which way that points depends on the statistic, so we don't classify the error yet.",
         "- Full write-up with every figure and table: [REPORT.md](REPORT.md). Code, data rules and logs: "
         "[GitHub](https://github.com/me-daniel/inkling-2-mini-quant).", "",
         "## Limits", "",
         "- KL measures how closely a file matches the reference model's outputs. Task quality needs its own test.",
         "- One machine, one llama.cpp build and one importance matrix.",
         "- The comparison covers the files we tested that day.",
         "- Speed was measured with stock llama.cpp on one Mac. Builds that run NVFP4 on Metal don't have the CPU fallback.", "",
         "## Provenance", "",
         f"- Base model: `{v['model']}` at revision `{v['rev'][v['model']][:7]}`, importance matrix from "
         "`ggml-org/Inkling-2-Mini-GGUF`.",
         f"- Measurement build: llama.cpp `{v['base'][:7]}` ({v['tag']}) with {len(v['patches'])} patches. The file runs "
         "without them.",
         "- Numbers: computed by scripts from the run logs. Full tables, settings and file revisions are in "
         "[DETAILS.md](DETAILS.md). Code, the pre-registration and its dated changes are on "
         "[GitHub](https://github.com/me-daniel/inkling-2-mini-quant).", ""]
    return C


DRAFT_LINE = "> DRAFT for Daniel's review. Not uploaded. Delete this line before any upload."


def mac_card(v: dict, label: dict, cand_bytes: int, ud_bytes: int, ud: list, s: dict, cand_ours: bool) -> list[str]:
    """Full card for the Addendum G Mac track (gate FAIL against UD-IQ3_XXS, Mac rule PASS). The draft line is added
    in main() after the wording scan."""
    f = mac_facts(v)
    udf = v["mf"]["UD-IQ3_XXS"]
    nv = one(r"ffn_down_exps NVFP4 x(\d+)", v["files"]["UD-IQ3_XXS"]["expert tensor types"])[1]
    cand_down = one(r"ffn_down_exps (\w+) x(\d+)", v["files"]["cand"]["expert tensor types"])
    assert int(nv) == udf["off"], "NVFP4 count in gate.md differs from the bench file"
    C = ["---", f"base_model: {v['model']}", "base_model_relation: quantized", f"license: {base_license()}", "tags:",
         "- gguf", "- llama.cpp", "- inkling2_loop", "- apple-silicon", "- exploratory", "---", "",
         "# Inkling 2 Mini GGUF, block-wise allocation, all expert tensors in Metal-supported types (exploratory)", ""]
    C += claim_par(v) + usage_section(v)
    C += [f"## Pre-registered gate against UD-IQ3_XXS: {v['verdict']}", "",
         f"This file ({n(cand_bytes)} bytes) has higher KL to the reference than Unsloth UD-IQ3_XXS ({n(ud_bytes)} bytes) "
         f"on both held-out sets. On T1 the difference is {ud[0]['d']:+.5f} [{ud[0]['lo']:+.5f}, {ud[0]['hi']:+.5f}], "
         f"and this file is worse on {ud[0]['worse']} of {ud[0]['n']} chunks. On T2 it is {ud[1]['d']:+.5f} "
         f"[{ud[1]['lo']:+.5f}, {ud[1]['hi']:+.5f}], worse on {ud[1]['worse']} of {ud[1]['n']} chunks. Under the "
         "pre-registered rule this is a FAIL, and it stands. The rule and every later change are in our public "
         "[change log](https://github.com/me-daniel/inkling-2-mini-quant/blob/main/CHANGELOG.md).", "",
         f"UD-IQ3_XXS keeps the release's native NVFP4 down projections in {nv} expert tensors, which the pinned Metal "
         f"backend cannot multiply. This file stores every expert down projection as {cand_down[1]} ({cand_down[2]} "
         "tensors), a type Metal can multiply.", ""] + gap_lines()
    C += gate_table(v, label)
    C += ["", "## What this card claims instead (post hoc)", "",
          f"After the FAIL we registered a narrower comparison in the change log (Addendum G, {v['g_date']}, about "
          f"{v['g_time']}), before any of its numbers existed. Two of the files were added by an amendment at about "
          f"{v['g_amend']}, before any download or score. The comparison is post hoc and is disclosed as such. It does "
          "not change the FAIL above.", "",
          mac_rule(v), "",
          "## Comparison", "",
          f"KL against the Q8_0 reference at P = {v['P']}. T1 is {v['ctx']['T1'][0]} chunks of {n(v['ctx']['T1'][1])} "
          f"tokens, T2 is {v['ctx']['T2'][0]} chunks of {n(v['ctx']['T2'][1])} tokens. Differences are this file minus "
          "the other file, so negative values favour this file. Bytes are read from each file. The size limit is "
          f"UD-IQ3_XXS's {n(ud_bytes)} bytes. bartowski IQ2_M is over it and is shown for context only.", ""]
    C += mac_table(v, card=True)
    for k in v["context"]:
        r1, r2 = v["mrows"][("T1", k)], v["mrows"][("T2", k)]
        won = all(r["win"] for r in (r1, r2))
        C += ["", f"{mac_label(v, k)} is over the limit and shown for context. It has "
                  + ("higher" if won else "not shown to have higher") + f" KL than this file on both sets ({r1['d']} "
                  f"[{r1['lo']}, {r1['hi']}] on T1, {r2['d']} [{r2['lo']}, {r2['hi']}] on T2)."]
    C += ["",
          "## Speed", ""] + speed_par(v) + depth_section(v)
    C += ["", "## Stability", "",
          "The registered stability battery, run on this file and on the Q8_0 reference with the same server settings.", ""]
    C += stability_lines(v)
    C += ["", "## File", "",
          f"- {n(cand_bytes)} bytes. Expert tensor types: {v['files']['cand']['expert tensor types']}. Non-expert "
          f"tensors: {s['ne']}.",
          f"- Block-wise allocation chosen from a measured {len(v['blocks'])}-block table at P = {v['P']}. Allocation "
          f"(gate_up / down): {choice(s['choice'])}.",
          f"- {'Uses' if cand_ours else 'Uses no tensors from'} our imatrix-weighted Q2_0 encoding (our encoder).",
          f"- Expert tensors in types the pinned Metal backend cannot multiply: {v['mf']['cand']['off']}.",
          "- The per-pass result did not shape the allocation. The block map behind it measures looped blocks with all "
          "passes tied.", "",
          "## Setup", ""] + setup(v) + ["", "## Provenance", ""] + provenance(v)
    C += ["", "Outside files in the comparison:", ""] + ext_revisions(v) + [""]
    return C


def check(name: str, s: str) -> list[str]:
    return [f"{name}: '{m.group(0)}' near: {s[max(0, m.start() - 40):m.end() + 40]!r}"
            for p in FORBIDDEN for m in re.finditer(p, s, re.I)]


def main() -> None:
    v = load()
    if v["verdict"] == "PASS":
        raise SystemExit("gate PASS: the full card needs stability-*.json and bench-*.txt; not drafted by this script")
    note, card = render(v)
    bad = check("note.md", note) + check("MODEL_CARD.md", card)
    if bad:
        print("\n".join(bad), file=sys.stderr)
        raise SystemExit(1)
    OUT.mkdir(exist_ok=True)
    if v["mac"]:
        (OUT / "release").mkdir(exist_ok=True)
        (OUT / "release" / "DETAILS.md").write_text(card.split("---\n", 2)[2].lstrip(), encoding="utf-8")  # full card, no front matter
        lean = "\n".join(lean_card(v, v["mf"]["cand"]["bytes"]))
        bad = check("README.md", lean)
        if bad:
            print("\n".join(bad), file=sys.stderr)
            raise SystemExit(1)
        (OUT / "release" / "README.md").write_text(lean, encoding="utf-8")
        card = DRAFT_LINE + "\n\n" + card  # the only line outside the scan; it names the reviewer
    (OUT / "note.md").write_text(note, encoding="utf-8")
    (OUT / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    kind = (f"Mac-track card, Addendum G rule {v['mq']['verdict']}, stability {'PASS' if v['judge']['pass'] else 'FAIL'}"
            if v["mac"] else "stub, not for upload")
    print(f"wrote drafts/note.md ({len(note.split())} words), drafts/MODEL_CARD.md ({kind})")


if __name__ == "__main__":
    main()
