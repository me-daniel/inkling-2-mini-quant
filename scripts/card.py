"""Renders drafts/MODEL_CARD.md and drafts/note.md (pilot write-up drafts, never uploaded by this script) from files.

    .venv/bin/python -I scripts/card.py

Every number comes from results/ (pilot tables, env.md, data.md, t2.md, convert-check.md), logs/revisions.txt, the
llama-perplexity logs (context and chunk counts) and the imatrix download metadata in models/. Gate differences are
recomputed with analysis/compare.py (same paired bootstrap), per-pass ratios and the Addendum F test with
analysis/perpass_report.py. Card wording follows CHANGELOG.md (pilot entry, Addenda C, E, F). If results/pilot/gate.md
does not say PASS, the card is a stub marked NOT FOR UPLOAD and the note carries the measurements. A PASS card needs
the stability and bench files and is not drafted here. The output is checked for wording the CHANGELOG forbids,
private paths and dashes; any hit exits 1 without writing.
"""

from __future__ import annotations

import ast
import csv
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
             r"/Users/", r"private/", r"scratch/", r"Daniel", r"—", r"–", r"\bSOTA\b", r"state of the art"]


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
    return v


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
    closer = pp["r"] < ls["r"] and abs(pp["norm_ratio"] - 1) < abs(ls["norm_ratio"] - 1)
    rung = {k: pr.apart(v["pp"], v["b3"], "fisher", k) for k in ("r", "norm_ratio")}
    rung_moved = all(d and p for d, p in rung.values())
    kl = v["ppkl"]
    kr = v["kl_ratio"]
    ls_sum = v["kl2"]["ls67"] + v["kl2"]["ls89"] + v["kl2"]["ls1011"]
    b3_sum = v["kl2"]["3b-p1"] + v["kl2"]["3b-p2"] + v["kl2"]["3b-p3"]
    all_kl = float(kl[("all", "pp-base")])
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
         "- Under our pre-registered rule a failed gate means no model is released. This note is the result.",
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
          "- Stability battery and speed (`llama-bench`): " + ("run." if v["stability_run"] and v["bench_run"] else
                                                                 "not run under the protocol, which skips them after a quality FAIL."
                                                                 + (f" {len(v['explore_bench'])} off-protocol `llama-bench` runs were made "
                                                                    "after the FAIL; their figures are not reported here."
                                                                    if v["explore_bench"] else "")),
          "", "Expert tensors in types the pinned Metal backend cannot multiply, read from each file's tensor types:", "",
          "| Build | Bytes | Expert tensor types | Off-Metal expert tensors |", "|---|---|---|---|"]
    for k, r in v["files"].items():
        name = stock if k == stock_key else k
        N.append(f"| {name} | {r['bytes']} | {r['expert tensor types']} | {v['metal'][k]} |")
    N += ["", "UD-IQ3_XXS keeps the release's NVFP4 down projections in "
          f"{v['metal']['UD-IQ3_XXS']} expert tensors. The tensor count says where those multiplies cannot run. It is "
          "not a speed measurement, and we report no speed figures.", "",
          "## Our Q2_0 encoder", "",
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
          "single-pass changes. Norms are Fisher-weighted. r = ||d_all - sum_k d_k|| / ||d_all||.", "",
          "| Statistic | Pass split, 2-bit | Layer split, 2-bit (control) | Pass split, 3-bit |", "|---|---|---|---|"]
    for k, name in (("r", "r"), ("norm_ratio", "||d_all|| / ||sum_k d_k||"), ("sq_over_sum", "||d_all||^2 / sum_k ||d_k||^2"),
                    ("cos_12", "cos(d1, d2)"), ("cos_23", "cos(d2, d3)"), ("cos_13", "cos(d1, d3)")):
        N.append(f"| {name} | {ci(pp, k)} | {ci(ls, k)} | {ci(b3, k)} |")
    N.append(f"| KL all / sum of parts | {kr['ratio']:.3f} [{kr['ci95'][0]:.3f}, {kr['ci95'][1]:.3f}] | "
             f"{v['kl2']['all'] / ls_sum if 'all' in v['kl2'] else all_kl / ls_sum:.3f} (point only) | "
             f"{v['kl2']['3b-all'] / b3_sum:.3f} (point only) |")
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
          + ("gives the same verdict." if licensed_head == licensed else "gives a different verdict."), ""]
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
          "", "Gaussian noise of the same relative error as the 2-bit level, on B2:", "",
          f"- The same draw on all three passes gave KL {v['noise']['kl_tied']:.5f}. Different draws per pass gave "
          f"{v['noise']['kl_indep']:.5f}. Ratio {v['noise']['ratio']:.3f} [{lo:.3f}, {hi:.3f}].",
          "- Each condition is one draw per pass, so tying is confounded with which draws sit on passes 2 and 3. The "
          "CI covers chunk sampling only.", "",
          f"Three analysis choices differ from the registered plan: the Fisher norm as primary, a {v['head_nats']}-nat head for the "
          "Euclidean substitute, and a pooled ratio of sums instead of a per-token ratio. They were fixed in code "
          "before the per-pass logits existed. They still need a dated entry in the change log.", "",
          "The per-pass test is a scientific result, not the reason for the release allocation. The block map measures "
          "looped blocks with all passes tied.", "",
          "## Limits", "",
          "- This is an exploratory pilot. None of it is confirmatory evidence.",
          "- One reference, one importance matrix, one machine, one engine commit.",
          f"- T2 has {v['ctx']['T2'][0]} chunks, so its bootstrap is coarse.",
          "- KL to a reference is not task quality. We report no task scores.",
          "- The per-pass test covers block B2 at two levels. B3 was not perturbed per pass.", ""]

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
    return "\n".join(N) + "\n", "\n".join(C) + "\n"


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
    (OUT / "note.md").write_text(note, encoding="utf-8")
    (OUT / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    print(f"wrote drafts/note.md ({len(note.split())} words), drafts/MODEL_CARD.md (stub, not for upload)")


if __name__ == "__main__":
    main()
