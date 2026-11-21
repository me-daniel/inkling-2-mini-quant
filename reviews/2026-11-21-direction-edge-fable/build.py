"""Writes reviews/2026-11-21-direction-edge-fable.md. CPU only: reads result files, logs, GGUF headers and the
saved GitHub API record of PR #20456 in this folder. Starts no model run.

    .venv/bin/python -I reviews/2026-11-21-direction-edge-fable/build.py

Hours, probabilities and words marked estimate, guess or inference are the reviewer's and are not computed.
"""

import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
R = ROOT / "results" / "pilot"
L = ROOT / "logs"
M = ROOT / "models"


# ---------- readers ----------
def read_kl():
    rows = {}
    lines = (R / "kl.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    for ln in lines[1:]:
        d = dict(zip(head, ln.split("\t")))
        rows[d["run"]] = dict(kl=float(d["mean_kl"]), se=float(d["se"]), top=float(d["same_top_p"]),
                              chunks=[float(x) for x in d["per_chunk"].split(",")])
    return rows


CMP = re.compile(r"^(T\d) cand vs (\S+): KL ([\d.]+) vs ([\d.]+), diff ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\], "
                 r"top-1 ([\d.]+)% vs ([\d.]+)% -> (.+)$")


def read_cmp(*names):
    out = {}
    for n in names:
        for ln in (R / n).read_text().splitlines():
            m = CMP.match(ln)
            if m:
                out[(m.group(1), m.group(2))] = dict(diff=m.group(5), lo=m.group(6), hi=m.group(7), verdict=m.group(10),
                                                     top_c=float(m.group(8)), top_b=float(m.group(9)))
    return out


def read_bench(name):
    out = {}
    for ln in (R / name).read_text().splitlines():
        m = re.search(r"\|\s*((?:pp|tg)\d+(?: @ d\d+)?)\s*\|\s*([\d.]+) ± ([\d.]+)", ln)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def eta(log):
    m = re.search(r"([\d.]+) seconds per pass - ETA ([\d.]+) minutes", (L / log).read_text())
    return float(m.group(1)), float(m.group(2))


def layout():
    cache = HERE / "layout.json"
    paths = {"cand": M / "cand.gguf", "UD": M / "UD-IQ3_XXS.gguf"}
    if cache.exists():
        d = json.loads(cache.read_text())
        if all(d[k]["file"] == os.path.getsize(p) for k, p in paths.items()):
            return d
    import gguf
    out = {}
    for label, path in paths.items():
        r = gguf.GGUFReader(str(path))
        cat = Counter(); gu = {}; dn = {}; ne = Counter(); extra = Counter(); down_elems = 0
        for t in r.tensors:
            m = re.match(r"blk\.(\d+)\.(ffn_gate_up_exps|ffn_down_exps)\.(.*)$", t.name)
            ty = t.tensor_type.name; nb = int(t.n_bytes)
            if m and m.group(3) == "weight":
                (gu if m.group(2) == "ffn_gate_up_exps" else dn)[int(m.group(1))] = ty
                cat[m.group(2)] += nb
                if m.group(2) == "ffn_down_exps" and int(m.group(1)) == 0:
                    down_elems = int(t.n_elements)
            elif m:
                cat["expert scale tensors"] += nb; extra[m.group(2) + "." + m.group(3)] += 1
            else:
                cat["non-expert"] += nb; ne[ty] += nb
        out[label] = dict(file=os.path.getsize(path), cat=dict(cat), gu={str(k): gu[k] for k in sorted(gu)},
                          dn={str(k): dn[k] for k in sorted(dn)}, ne=dict(ne), extra=dict(extra), down_elems=down_elems)
    cache.write_text(json.dumps(out, indent=1))
    return out


def tsv(path):
    lines = Path(path).read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def f5(x): return f"{x:.5f}"
def f4(x): return f"{x:.4f}"
def s5(x): return f"{x:+.5f}"
def mb(x): return f"{x / 1e6:,.0f} MB"
def gb(x): return f"{x / 1e9:.2f} GB"
def by(x): return f"{x:,}"


# ---------- numbers ----------
kl = read_kl()
cmp_ = read_cmp("gate-quality.txt", "mac-quality.txt", "mac-context-IQ2_M.txt")
lay = layout()

FILES = {  # label -> (kl.tsv key, path, bench file)
    "cand": ("cand", M / "cand.gguf", "bench-cand.txt"),
    "Unsloth UD-IQ3_XXS": ("UD-IQ3_XXS", M / "UD-IQ3_XXS.gguf", "bench-UD-IQ3_XXS.txt"),
    "AtomicChat AD-IQ2_S": ("atomic-AD-IQ2_S", M / "ext" / "atomic-AD-IQ2_S.gguf", "bench-atomic-AD-IQ2_S.txt"),
    "bartowski IQ2_S": ("bart-IQ2_S", M / "ext" / "bart-IQ2_S.gguf", "bench-bart-IQ2_S.txt"),
    "mudler APEX-I-Nano": ("apex-I-Nano", M / "ext" / "apex-I-Nano.gguf", "bench-apex-I-Nano.txt"),
    "bartowski IQ2_M": ("bart-IQ2_M", M / "ext" / "bart-IQ2_M.gguf", "bench-bart-IQ2_M.txt"),
    "stock IQ3_XXS (ours)": ("uniform-IQ3_XXS", M / "uniform-IQ3_XXS.gguf", "bench-uniform-IQ3_XXS.txt"),
}
size = {k: os.path.getsize(v[1]) for k, v in FILES.items()}
bench = {k: read_bench(v[2]) for k, v in FILES.items()}
LIMIT = size["Unsloth UD-IQ3_XXS"]
T = {k: {s: kl[f"final-{v[0]}-{s}-P3"] for s in ("T1", "T2")} for k, v in FILES.items()}
off_metal = {k: int(re.search(r"cannot multiply: (\d+)", (R / v[2]).read_text()).group(1)) for k, v in FILES.items()}

c, u = T["cand"], T["Unsloth UD-IQ3_XXS"]
under = [k for k in FILES if k != "cand" and size[k] < LIMIT and off_metal[k] == 0]
allmetal = [k for k in FILES if off_metal[k] == 0]
nxt = {s: min(under, key=lambda k: T[k][s]["kl"]) for s in ("T1", "T2")}
assert nxt["T1"] == nxt["T2"]
NX = nxt["T1"]
n = T[NX]
closure = {s: (n[s]["kl"] - c[s]["kl"]) / (n[s]["kl"] - u[s]["kl"]) for s in ("T1", "T2")}
wins_nx = {s: sum(a < b for a, b in zip(c[s]["chunks"], n[s]["chunks"])) for s in ("T1", "T2")}
loss_ud = {s: sum(a > b for a, b in zip(c[s]["chunks"], u[s]["chunks"])) for s in ("T1", "T2")}
nch = {s: len(c[s]["chunks"]) for s in ("T1", "T2")}
pp = {k: bench[k]["pp512"][0] for k in FILES}
tg = {k: bench[k]["tg128"][0] for k in FILES}
pp_am = [pp[k] for k in allmetal]; tg_am = [tg[k] for k in allmetal]

depth = read_bench("depth-cand.txt") if (R / "depth-cand.txt").exists() else {}
mem = (R / "memory-cand.txt").read_text() if (R / "memory-cand.txt").exists() else ""
rss = int(re.search(r"(\d+)\s+maximum resident set size", mem).group(1)) if mem else None

sec_ud, eta_ud = eta("ppl-UD-IQ3_XXS-S-P3.log")
sec_c, eta_c = eta("ppl-sel-2-S-P3.log")

# S scores
S = {k: kl[k]["kl"] for k in kl if k.endswith("-S-P3")}
cand_S, ud_S, uni3_S, iq4_S = S["sel-2-S-P3"], S["UD-IQ3_XXS-S-P3"], S["sweep-3bit-S-P3"], S["floor-iq4nl-S-P3"]
sel_S = [S[f"sel-{i}-S-P3"] for i in range(4)]
pool3_bytes = os.path.getsize(M / "pool-3bit.gguf")
iq4_bytes = os.path.getsize(M / "Inkling-2-Mini-IQ4_NL.gguf")

# byte breakdown
cc, uc = lay["cand"]["cat"], lay["UD"]["cat"]
gu_diff = uc["ffn_gate_up_exps"] - cc["ffn_gate_up_exps"]
dn_diff = uc["ffn_down_exps"] - cc["ffn_down_exps"]
ne_diff = uc["non-expert"] - cc["non-expert"]
scale_bytes = uc.get("expert scale tensors", 0)
n_scale = lay["UD"]["extra"]
ud_ne = sorted(lay["UD"]["ne"].items(), key=lambda x: -x[1])
cand_ne = sorted(lay["cand"]["ne"].items(), key=lambda x: -x[1])
gu_types = {k: Counter(lay[k]["gu"].values()) for k in ("cand", "UD")}
ud_2bit_layers = [int(l) for l, t in lay["UD"]["gu"].items() if t != "IQ3_XXS"]
cand_2bit_layers = [int(l) for l, t in lay["cand"]["gu"].items() if t != "IQ3_XXS"]
ud_nv = [int(l) for l, t in lay["UD"]["dn"].items() if t == "NVFP4"]

# additive map estimate of the gate_up cost above the all-IQ3_XXS level
BLOCKS = {"B1": range(0, 6), "B2": range(6, 12), "B3": range(12, 18), "B4": range(18, 20), "B5": range(20, 24)}
blk_of = {l: b for b, rng in BLOCKS.items() for l in rng}
inc = {b: {lv: (S[f"map-{b}-{lv}-S-P3"] - S[f"map-{b}-3bit-S-P3"]) / len(BLOCKS[b]) for lv in ("gu2s", "gu2xxs")} for b in BLOCKS}


def gu_cost(types, xs_as):
    lv = {"IQ3_XXS": None, "IQ2_S": "gu2s", "IQ2_XXS": "gu2xxs", "IQ2_XS": xs_as}
    return sum(inc[blk_of[int(l)]][lv[t]] for l, t in types.items() if lv[t])


dgu_c = gu_cost(lay["cand"]["gu"], "gu2s")
dgu_u_lo, dgu_u_hi = gu_cost(lay["UD"]["gu"], "gu2s"), gu_cost(lay["UD"]["gu"], "gu2xxs")
ne_c = S["map-NE-Q6K-S-P3"]
add_c = dgu_c + ne_c
meas_c = cand_S - uni3_S
shrink = meas_c / add_c
gap_S = cand_S - ud_S
d_floor = uni3_S - ud_S
d_raw = (d_floor + dgu_u_lo, d_floor + dgu_u_hi)
d_shr = (d_floor + shrink * dgu_u_lo, d_floor + shrink * dgu_u_hi)
be = (ne_c + dgu_c - dgu_u_hi, ne_c + dgu_c - dgu_u_lo)  # NE cost UD would need, additive map units

# linear-in-weight-error extrapolation (Addendum A style) for IQ4_NL down in layers 0-19
tb = tsv(R / "t-blocks.tsv"); q2 = tsv(R / "q2opt-errors.tsv")


def mean_t(rows, kind, layers, col, **filt):
    return statistics.mean(float(r[col]) for r in rows if r["kind"] == kind and int(r["layer"]) in layers
                           and all(r[k] == v for k, v in filt.items()))


lin_d = 0.0; lin3 = 0.0; meas3 = 0.0
for b, rng in BLOCKS.items():
    t_dn4 = mean_t(tb, "ffn_down_exps", rng, "t", pool="3bit")
    t_dno = mean_t(q2, "ffn_down_exps", rng, "t_opt")
    t_gu3 = mean_t(tb, "ffn_gate_up_exps", rng, "t", pool="3bit")
    t_gu2 = mean_t(tb, "ffn_gate_up_exps", rng, "t", pool="2bit")
    a_dn = (S[f"map-{b}-dnq2o-S-P3"] - S[f"map-{b}-3bit-S-P3"]) / (t_dno - t_dn4)
    a_gu = (S[f"map-{b}-gu2xxs-S-P3"] - S[f"map-{b}-3bit-S-P3"]) / (t_gu2 - t_gu3)
    lin3 += a_dn * t_dn4 + a_gu * t_gu3
    meas3 += S[f"map-{b}-3bit-S-P3"]
    if b != "B5":
        lin_d += a_dn * t_dn4
t_dn4_all = mean_t(tb, "ffn_down_exps", range(0, 20), "t", pool="3bit")
t_gu3_all = mean_t(tb, "ffn_gate_up_exps", range(0, 24), "t", pool="3bit")

# format constants
common = (ROOT / "vendor/llama.cpp/ggml/src/ggml-common.h").read_text()
nv_sub = int(re.search(r"#define QK_NVFP4_SUB (\d+)", common).group(1))
iq4_blk = int(re.search(r"#define QK4_NL (\d+)", common).group(1))
kdir = ROOT / "vendor/llama.cpp/ggml/src/ggml-metal/kernels"
n_kernel_files = len([p for p in kdir.iterdir() if p.suffix == ".metal"])
has_mono = (ROOT / "vendor/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal").exists()

# PR record (GitHub API, saved in this folder)
pr = json.loads((HERE / "pr-20456.json").read_text())
prf = json.loads((HERE / "pr-20456-files.json").read_text())
pr_kernels = sorted({k for f in prf for k in re.findall(r"kernel_\w*nvfp4\w*", f.get("patch") or "")})
pr_id = [k for k in pr_kernels if "_id_" in k]
pr_metal = next(f for f in prf if f["filename"].endswith(".metal"))
pr_speed = {k: (a, b) for k, a, b in re.findall(r"\| (pp512|tg128) \| ([\d.]+) t/s \| ([\d.]+) t/s", pr["body"])}

# changelog and scoop check facts
chg = (ROOT / "CHANGELOG.md").read_text()
pend = re.search(r"`APEX-I-NanoPlus-V2\.1` \(([\d,]+) bytes.*?`Q2_K` \(([\d,]+) bytes", chg, re.S).groups()
scoop = (ROOT / "reviews/2026-11-21-scoop-check.md").read_text()
lamq = re.search(r"\| (8\.87 GB) \(.*?this file free_KL ([\d.]+), Unsloth UD-IQ3_XXS \(11\.66 GB\) ([\d.]+)", scoop).groups()
bayt = re.search(r"repacked for MLX, ([\d.]+ GB)", scoop).group(1)

pending_done = [x for x in ("isvalorum-NanoPlus", "prithiv-Q2_K") if f"final-{x}-T1-P3" in kl]
upload_log = L / "hf-upload-v1.log"
has_upload = upload_log.exists() and "huggingface.co" in upload_log.read_text()
hybrid_sh = (ROOT / "scripts/hybrid.sh").exists()
hybrid_scored = "hybrid-S-P3" in kl or ((L / "ppl-hybrid-S-P3.log").exists() and "Mean" in (L / "ppl-hybrid-S-P3.log").read_text())

pp_res = json.loads((R / "perpass-residual.json").read_text())["fisher"]
ls_res = json.loads((R / "layersplit-residual.json").read_text())["fisher"]
stab = json.loads((R / "stability-cand.json").read_text())["gsm8k"]
stab_ref = json.loads((R / "stability-ref.json").read_text())["gsm8k"]

def card_facts(path):
    lines = (ROOT / path).read_text().splitlines()
    h2 = [ln[3:] for ln in lines if ln.startswith("## ")]
    fail_i = next((i for i, ln in enumerate(lines) if re.search(r"fail", ln, re.I)), None)
    head = next((ln.lstrip("# ") for ln in reversed(lines[:fail_i or 0]) if ln.startswith("#")), "none")
    text = "\n".join(lines)
    three = re.search(r"[^.\n]*three times[^.\n]*\.", text)
    return dict(h2=h2, n=len(lines), fail_line=None if fail_i is None else fail_i + 1, fail_head=head,
                fail_h2=next((i + 1 for i, h in enumerate(h2) if "FAIL" in h or "fail" in h), None),
                m1=cmp_[("T1", "UD-IQ3_XXS")]["diff"].lstrip("+") in text, m2=cmp_[("T2", "UD-IQ3_XXS")]["diff"].lstrip("+") in text,
                three=three.group(0).strip() if three else None, t1="T1" in text)


rd = card_facts("drafts/release/README.md")
mc = card_facts("drafts/MODEL_CARD.md")
yn = lambda b: "yes" if b else "no"
sib = (ROOT / "reviews/2026-11-21-metal-nvfp4-feasibility.md").exists()
lm = (ROOT / "vendor/llama.cpp/src/llama-model.cpp").read_text()
scale_opt = bool(re.search(r'ffn_down_exps_s = create_tensor\(tn\(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i\), \{n_expert\}, TENSOR_NOT_REQUIRED\)', lm)) and \
    bool(re.search(r'ffn_up_exps_s = create_tensor\(tn\(LLM_TENSOR_FFN_GATE_UP_EXPS, "scale", i\), \{n_expert\}, TENSOR_NOT_REQUIRED\)', lm))


def row(k):
    key = FILES[k][0]
    d1 = cmp_.get(("T1", key)); d2 = cmp_.get(("T2", key))
    return (f"| {k} | {by(size[k])} | {'under' if size[k] < LIMIT else ('limit' if size[k] == LIMIT else 'over')} | {off_metal[k]} | "
            f"{f5(T[k]['T1']['kl'])} | {f5(T[k]['T2']['kl'])} | "
            f"{(d1['diff'] + ' [' + d1['lo'] + ', ' + d1['hi'] + ']') if d1 else 'n/a'} | "
            f"{(d2['diff'] + ' [' + d2['lo'] + ', ' + d2['hi'] + ']') if d2 else 'n/a'} | "
            f"{T[k]['T1']['top']:.2f}% | {pp[k]:.1f} | {tg[k]:.1f} |")


order = ["cand", "Unsloth UD-IQ3_XXS", "AtomicChat AD-IQ2_S", "bartowski IQ2_S", "mudler APEX-I-Nano", "bartowski IQ2_M", "stock IQ3_XXS (ours)"]
table = "\n".join(row(k) for k in order)
gU = cmp_[("T1", "UD-IQ3_XXS")]; gU2 = cmp_[("T2", "UD-IQ3_XXS")]
gN = cmp_[("T1", FILES[NX][0])]; gN2 = cmp_[("T2", FILES[NX][0])]


def types_line(k):
    return ", ".join(f"{t} x{n_}" for t, n_ in sorted(gu_types[k].items()))


def ne_line(items):
    return ", ".join(f"{t} {mb(v)}" for t, v in items)


md = f"""# Direction review: is there an edge, and what is the file for?

- Date: 2026-11-21. Reviewer: Fable 5.2 (workflow subagent). Independent of the session that built the candidate.
- Question from Daniel: "Do we have any edge, or should we redo more? What is the point of our model at the moment?"
- Read: `results/pilot/` (gate.md, mac-quality.txt, bench-*.txt, types-*.txt, selection.md, phase2.md, perpass.md, kl.tsv, t-blocks.tsv, q2opt-errors.tsv, depth and memory files), `drafts/MODEL_CARD.md`, `drafts/note.md`, `drafts/release/README.md`, `reviews/2026-11-21-scoop-check.md`, `CHANGELOG.md` with Addenda A to G, the two code reviews, `scripts/hybrid.sh`, and the headers of `cand.gguf` and `UD-IQ3_XXS.gguf`. Not read: `private/`.
- Numbers: this file is written by `reviews/2026-11-21-direction-edge-fable/build.py`. Every measured number is read by that script from the result files, logs, GGUF headers, the llama.cpp tree, or the GitHub API record of PR #20456 saved in that folder (fetched tonight). Rerun with `.venv/bin/python -I reviews/2026-11-21-direction-edge-fable/build.py`.
- Labels: "estimate" and "guess" mark my own hours and odds. They are typed, not computed. "Inference" marks a conclusion that follows from the numbers but was not measured.
- No model run was started for this review.

## Short answer

1. **There is an edge. It is narrow, and it will not last.** Among the GGUFs under {by(LIMIT)} bytes that run fully on Metal, the candidate has the lowest held-out KL. The next file is {NX}, behind by {gN['diff'].lstrip('-')} KL on T1 and {gN2['diff'].lstrip('-')} on T2. Speed is a tie with every other all-Metal file.
2. **It is not the best file at its size.** UD-IQ3_XXS is closer to the reference by {gU['diff'].lstrip('+')} KL on T1 and {gU2['diff'].lstrip('+')} on T2. The candidate's only lead over it is Mac speed. That lead comes from a missing kernel in llama.cpp, not from anything the file does well.
3. **Do not redo the candidate.** It is frozen (Addendum G, point 1) and T1 and T2 have been seen. Inference: no reallocation inside Metal types will pass the original gate either. Our own build with IQ3_XXS gate_up in all 24 layers, IQ4_NL down and Q8_0 non-experts is {gb(pool3_bytes)} and scores {f5(uni3_S)} on S. UD-IQ3_XXS scores {f5(ud_S)} at {gb(LIMIT)}. A file that is at least as fine as the candidate in every tensor, and a gigabyte larger, still does not reach it. Only a change of the down format can move the comparison with UD.
4. **The point of the model right now:** a stopgap for people who run llama.cpp on a Mac and want a file near {size['cand'] / 1e9:.1f} GB. The pilot is worth more than the file. It found the NVFP4 fact on Metal, the `.scale` bug on passes 2 and 3, the Q2_0 encoder defect, and it produced the per-pass method that the scoop check calls the novel part.
5. **Recommendation:** close the release track with one bounded step. Run a two-build diagnostic on S (about {eta_ud + eta_c:.0f} minutes of scoring). Then put the hours into the per-pass work. Section 4 has the order.

## 1. The edge

### 1.1 What is measured

KL against the Q8_0 reference at P = 3. Differences are cand minus file with the 95% CI from `gate-quality.txt`, `mac-quality.txt` and `mac-context-IQ2_M.txt`. Speed is the `llama-bench` mean from `bench-*.txt` on the M5 Pro.

| File | Bytes | vs limit | Off-Metal expert tensors | KL T1 | KL T2 | cand minus file, T1 | cand minus file, T2 | Top-1 T1 | pp512 t/s | tg128 t/s |
|---|---|---|---|---|---|---|---|---|---|---|
{table}

- Against UD-IQ3_XXS the candidate loses on {loss_ud['T1']} of {nch['T1']} T1 chunks and {loss_ud['T2']} of {nch['T2']} T2 chunks. Its KL is {c['T1']['kl'] / u['T1']['kl']:.2f} times UD's on T1 and {c['T2']['kl'] / u['T2']['kl']:.2f} times on T2. Top-1 agreement is {u['T1']['top'] - c['T1']['top']:.2f} points lower on T1.
- Against {NX}, the best other all-Metal file under the limit, it wins on {wins_nx['T1']} of {nch['T1']} T1 chunks and {wins_nx['T2']} of {nch['T2']} T2 chunks. Top-1 is {c['T1']['top'] - n['T1']['top']:.2f} points higher on T1.
- The candidate sits about halfway between the two. It closes {100 * closure['T1']:.0f}% of the T1 gap between {NX} and UD-IQ3_XXS, and {100 * closure['T2']:.0f}% of the T2 gap.
- It also beats two larger all-Metal files, bartowski IQ2_M ({gb(size['bartowski IQ2_M'])}) and our stock IQ3_XXS ({gb(size['stock IQ3_XXS (ours)'])}).
- Speed among the {len(allmetal)} all-Metal files: pp512 from {min(pp_am):.1f} to {max(pp_am):.1f} t/s, tg128 from {min(tg_am):.1f} to {max(tg_am):.1f} t/s. The candidate has no speed edge inside that group. Against UD-IQ3_XXS it is {pp['cand'] / pp['Unsloth UD-IQ3_XXS']:.2f} times as fast on pp512 and {tg['cand'] / tg['Unsloth UD-IQ3_XXS']:.2f} times on tg128.
- The file is {by(LIMIT - size['cand'])} bytes under the limit. The selection left that budget unused.

So the measured claim holds: lowest KL among the all-Metal files under the limit that were tested, at equal speed. The gate FAIL holds too.

### 1.2 Who would use it

- A person on Apple Silicon, on llama.cpp or a front end built on it, who wants a file of about this size.
- The maximum resident set size in the 16K-depth bench run was {rss / 1e9:.1f} GB (`memory-cand.txt`). Inference: a 16 GB Mac is tight, an 18 or 24 GB Mac fits. A Mac with more memory can load a larger file and has no reason to pick this size class. None of this was measured on another machine.
- That person gets the same speed as from {NX} and {gN['diff'].lstrip('-')} lower KL on T1. Compared with UD-IQ3_XXS they get {pp['cand'] / pp['Unsloth UD-IQ3_XXS']:.1f} times the prompt speed and pay {gU['diff'].lstrip('+')} KL.
- The two margins are about the same size. The lead over {NX} is {gN['diff'].lstrip('-')} on T1. The loss to UD-IQ3_XXS is {gU['diff'].lstrip('+')}. If the lead is worth a release, the loss is worth the same weight in the text. The card cannot call one meaningful and the other small.
- We have no evidence that either margin is visible in use. KL is not task quality. The stability battery is at ceiling for both files ({stab['correct']} of {stab['n']} GSM8K for the candidate, {stab_ref['correct']} of {stab_ref['n']} for the reference), so it cannot separate them.
- Not served: anyone on CUDA or CPU, where UD-IQ3_XXS runs every tensor and is closer. Anyone on MLX. The scoop check lists an MLX file of {lamq[0]} whose own card claims KL {lamq[1]} against {lamq[2]} for UD-IQ3_XXS (its numbers, its teacher, not comparable with ours), and an MLX repack that runs the release's native NVFP4 on the Apple GPU at {bayt}.

Guess: the audience is small. It is real today.

### 1.3 What happens if Metal NVFP4 lands

- PR #{pr['number']} ("{pr['title']}") is {pr['state']}, created {pr['created_at'][:10]}, last updated {pr['updated_at'][:10]}, merge state "{pr['mergeable_state']}", {pr['comments']} comments, {pr['review_comments']} review comments. It is {pr['additions']} added lines in {pr['changed_files']} files. {pr_metal['additions']} of them are in `{pr_metal['filename'].split('/')[-1]}`.
- The patch already holds the MoE kernels: {', '.join('`' + k + '`' for k in pr_id)}. Its body reports pp512 {pr_speed['pp512'][0]} t/s on the CPU against {pr_speed['pp512'][1]} on Metal, and tg128 {pr_speed['tg128'][0]} against {pr_speed['tg128'][1]} (the author's machine and model, not ours).{" The sibling review `reviews/2026-11-21-metal-nvfp4-feasibility.md` reads the patch in detail and reports that a code owner approved it in March. I did not verify that." if sib else ""}
- Our pinned tree no longer has the single kernel file the PR edits (`ggml-metal.metal` present: {'yes' if has_mono else 'no'}; `kernels/` now holds {n_kernel_files} `.metal` files). So the PR needs a port, not a rebase.
- If a kernel lands, UD-IQ3_XXS runs fully on Metal. Inference: its speed then joins the all-Metal group. At that point UD-IQ3_XXS is the better file on every axis we measure except {by(LIMIT - size['cand'])} bytes, and the candidate has no use case left. The card says this already.
- How long: guess. The PR sat for months because nothing popular used NVFP4. Inkling 2 shipped on 20 Nov with NVFP4 experts, and Unsloth's files carry them. Demand is new. I would plan for weeks, not for a year.

The edge lasts until somebody ports about {pr['additions']} lines. That is the honest size of the moat.

### 1.4 Does "lowest KL among all-Metal files at this size" matter when speed is equal?

Yes, for the narrow audience. With size and speed equal, closeness to the reference is the only thing left to choose on, and the candidate wins it on every chunk. Three limits:

- **The field is two days old and nobody else aimed at this niche.** The comparators are general-purpose uploads. The obvious rival does not exist yet: UD-IQ3_XXS's own layout with its 20 NVFP4 down tensors re-encoded to IQ4_NL. Anyone can build that in minutes. Section 2 shows that our data cannot say whether it would beat the candidate.
- **The size limit is UD-IQ3_XXS's byte count.** It was the right limit for the gate. A user does not have that limit.
- **Two comparators are still open.** IsValorum APEX-I-NanoPlus-V2.1 ({pend[0]} bytes) and prithivMLmods Q2_K ({pend[1]} bytes) joined under Amendment 2. Scored so far in `kl.tsv`: {len(pending_done)} of 2. Both are smaller than the candidate. Guess: neither beats it. The rule needs both.

### 1.5 Problems in the current release text

- `logs/hf-upload-v1.log` holds a Hugging Face commit URL ({'present' if has_upload else 'absent'}). I did not open it and I do not know whether that repository is private or public. If it is public, option (a) has already happened. Daniel should confirm that this upload had his go at the time, as `CLAUDE.md` requires.
- Addendum G requires that "the FAIL against UD-IQ3_XXS is stated first, with both margins". `drafts/release/README.md`, as it stood when this file was built: first section "{rd['h2'][0]}"; the failure is first mentioned on line {rd['fail_line']} of {rd['n']}, under "{rd['fail_head']}"; T1 margin ({gU['diff'].lstrip('+')}) in the text: {yn(rd['m1'])}; T2 margin ({gU2['diff'].lstrip('+')}) in the text: {yn(rd['m2'])}. {"The release copy meets the margin part of the rule." if rd['m1'] and rd['m2'] else "The release copy does not meet the rule. It mentions the failure and leaves both margins to DETAILS.md."}
- `drafts/MODEL_CARD.md` gives both margins (T1: {yn(mc['m1'])}, T2: {yn(mc['m2'])}), under section {mc['fail_h2']} of {len(mc['h2'])}. Its first section is "{mc['h2'][0]}", which leads with the all-Metal claim. So it does not state the FAIL first either.
- {('The README says: "' + rd['three'] + '" The card rules forbid "loop-aware" and "pass-aware". This sentence is not those words, but a reader will take it that way. The map measures the looped blocks with all passes tied.') if rd['three'] else 'The README no longer has a sentence about the core running three times.'}
- The README names T1 for its KL column: {yn(rd['t1'])}.

## 2. What the data already says about the gap

"Should we redo more" depends on one thing we have not measured: is our gate_up and non-expert allocation any better than Unsloth's, once the down projections are equal?

### 2.1 Where the bytes go (GGUF headers)

| Tensor group | cand | UD-IQ3_XXS | UD minus cand |
|---|---|---|---|
| ffn_gate_up_exps | {by(cc['ffn_gate_up_exps'])} | {by(uc['ffn_gate_up_exps'])} | {gu_diff:+,} |
| ffn_down_exps | {by(cc['ffn_down_exps'])} | {by(uc['ffn_down_exps'])} | {dn_diff:+,} |
| non-expert | {by(cc['non-expert'])} | {by(uc['non-expert'])} | {ne_diff:+,} |
| per-expert scale tensors | 0 | {by(scale_bytes)} | {scale_bytes:+,} |

- **The down projections cost the same bytes in both files.** Native NVFP4 in layers {min(ud_nv)}-{max(ud_nv)} is exact with respect to the release. IQ4_NL at the same bytes is lossy. So wherever NVFP4 can run, IQ4_NL down in those layers is a dominated choice. This matters for the thesis as well: an allocator that does not offer native NVFP4 as a level works on a dominated frontier.
- UD-IQ3_XXS spends {mb(gu_diff)} more than we do on gate_up and {mb(-ne_diff)} less on non-experts.
- gate_up types. cand: {types_line('cand')}, 2-bit in layers {min(cand_2bit_layers)}-{max(cand_2bit_layers)}. UD: {types_line('UD')}, 2-bit in layers {', '.join(map(str, ud_2bit_layers))}.
- non-expert types. cand: {ne_line(cand_ne)}. UD: {ne_line(ud_ne)}.
- UD-IQ3_XXS also carries scale tensors for gate_up ({n_scale.get('ffn_gate_up_exps.scale', 0)} of them), so its gate_up codes are stored in a different scale domain from ours. A "same type" layer in the two files is not the same encoding.

### 2.2 How much does IQ4_NL down cost? Two estimates that disagree

Both are inference from S scores. Neither is a measurement of the down format.

- **Cross-file estimate.** Our all-IQ3_XXS build scores {f5(uni3_S)} on S. UD-IQ3_XXS scores {f5(ud_S)}, lower by {f5(d_floor)}, although {len(ud_2bit_layers)} of its gate_up layers are 2-bit and its non-experts are smaller than Q8_0. Pricing those {len(ud_2bit_layers)} layers from our block map (per-layer share of each block's gate_up step) gives {f5(dgu_u_lo)} to {f5(dgu_u_hi)} KL. Under additivity, native down in layers 0-19 is then worth {f5(d_raw[0])} to {f5(d_raw[1])} on S, plus whatever UD's non-expert types cost. Our own picks came in below the additive sum (measured over additive for the candidate's gate_up and non-expert steps: {shrink:.2f}). With that factor the range is {f5(d_shr[0])} to {f5(d_shr[1])}. The whole cand minus UD gap on S is {f5(gap_S)}.
- **Weight-error estimate (Addendum A style).** The mean relative weight error of IQ4_NL down in layers 0-19 is {t_dn4_all:.4f}, against {t_gu3_all:.4f} for IQ3_XXS gate_up (`t-blocks.tsv`). A slope fitted per block from the optimized-Q2_0 down level prices IQ4_NL down in layers 0-19 at {f5(lin_d)} KL. The same linear model prices the five 3-bit map blocks at {f5(lin3)} in total, where the map measured {f5(meas3)}. So it underprices small errors by a factor of {meas3 / lin3:.1f}, and its figure for the down format is a floor at best.

Reading: the cross-file estimate says the down format explains most or all of the gap ({100 * d_shr[0] / gap_S:.0f}% to {100 * d_raw[1] / gap_S:.0f}% of it before non-experts). The weight-error estimate says it explains {100 * lin_d / gap_S:.0f}%, or {100 * lin_d * meas3 / lin3 / gap_S:.0f}% after correcting by its own underpricing factor. The first rests on file scores and I trust it more. But it has a hole: it assumes Unsloth's gate_up encoding is no better than ours at the same type. If theirs is better, the down format is worth less and our allocation is worse than it looks.

### 2.3 Whose allocation is better at equal down? The map cannot say

Under additivity the hybrid beats UD-IQ3_XXS on S only if our gate_up and non-expert steps cost less than theirs.

- Our gate_up step costs {f5(dgu_c)} by the map. Theirs costs {f5(dgu_u_lo)} to {f5(dgu_u_hi)}. Theirs is cheaper, which fits the {mb(gu_diff)} more they spend there.
- Our non-experts (all Q6_K) cost {f5(ne_c)} by the map. Theirs have no map entry.
- Break-even: their non-expert types would have to cost {f5(be[0])} to {f5(be[1])}, which is {be[0] / ne_c:.1f} to {be[1] / ne_c:.1f} times our Q6_K figure. They use Q4_K and Q5_K on part of those tensors, so that is possible. It is not shown.

Guess: the hybrid lands within about 0.002 KL of UD-IQ3_XXS on S, either side. I would not bet on the sign.

### 2.4 The diagnostic that settles it: a 2 x 2 on S

| | down IQ4_NL (runs on Metal) | down native NVFP4 in layers 0-19 |
|---|---|---|
| our gate_up and non-experts | cand: {f5(cand_S)} (have) | **hybrid** (`scripts/hybrid.sh`: {'drafted' if hybrid_sh else 'not written'}, {'scored' if hybrid_scored else 'not run'}) |
| UD's gate_up and non-experts | **mirror** (new) | UD-IQ3_XXS: {f5(ud_S)} (have) |

- Rows compare the allocations. Columns price the down format. Two new builds fill it.
- **Mirror** is the build nobody has made: UD-IQ3_XXS's gate_up and non-experts, with our IQ4_NL down in layers 0-19. It runs fully on Metal. It is also the rival from section 1.4. If mirror scores below {f5(cand_S)} on S, then Unsloth's layout beats ours on our own ground, and "lowest KL among all-Metal files" is true only until someone re-encodes 20 tensors.
- Build note for mirror: use `cand.gguf` as base and take every UD tensor except the down weights of layers 0-19, plus UD's gate_up scale tensors with `--add`. Using UD as base would keep its down scale tensors and apply them to weights that are already scaled.
- Cost from the logs: a UD-type run on S took {sec_ud:.1f} s per chunk (ETA {eta_ud:.2f} min) because NVFP4 runs on the CPU. An all-Metal run took {sec_c:.1f} s per chunk (ETA {eta_c:.2f} min). So hybrid needs about {eta_ud:.0f} minutes and mirror about {eta_c:.0f}.
- Caveat: the candidate was selected on S from 4 picks (S scores {f5(min(sel_S))} to {f5(max(sel_S))}), UD's layout was not. That bias favours us and is small next to the gap. S is still the right set. T stays reserved for release comparisons.
- Addendum C already allows a native-down build "as a quality reference only". Mirror is new and needs its own dated entry before it is scored.

## 3. Options

Hours are my estimates.

| Option | Cost (estimate) | Risk | What it would show | Thesis fit | Verdict |
|---|---|---|---|---|---|
| (a) Upload as is | 1 to 2 h | Low. A file that loses to UD, with a short shelf life | Nothing new | None | Do once, bounded, after fixes. No upkeep |
| (b) Hybrid, plus mirror | 1 to 2 h, about {eta_ud + eta_c:.0f} min of scoring | Low. Loader must accept a down scale without a gate_up scale | Whether the gap is the down format, and whose allocation is better | Medium. Tests the map-to-allocator chain. Tells the thesis whether NVFP4 must be a level | **Do first** |
| (c) Port the Metal NVFP4 kernel | 8 to 16 h plus Daniel's review. Upstream merge time unknown | Medium. Removes our own file's edge. Review latency upstream | UD-type files at Metal speed on this Mac | Indirect. Speeds every native-down run here by about {sec_ud / sec_c:.1f}x | Decide after (b). Worth a 4 h spike |
| (d) Better NVFP4 to IQ4_NL re-encode | Unknown. Daniel's task | High. No exact mapping exists | How much of the down cost an encoder can recover | High for the thesis tool. Low for this release | Park until (b) gives the ceiling |
| (e) Stop the release track, write up per-pass | 20 to 40 h | Medium. Result is narrow today (one block, two levels) | The part nobody else has | Direct | **Main line after (b)** |
| (f1) Publish the measurements | 2 to 3 h | Low | An independent KL table of the public files under one reference | Low | Fold into (a) |

### (a) Upload as is

The work is done and the claim is true as scoped. The cost is small. The return is small too: a narrow audience, a lead that depends on an upstream gap, and no task evidence. The original pilot rule said a failed gate means "publish the measurements without a model". Addendum G changed that after the fact, with approval and in the open. A strict reader will still see moved goalposts. The defence is the rule Addendum G itself set: the FAIL first, with both margins. The release README does not do that yet (section 1.5).

Do it once if Daniel wants a public artifact. Conditions: both pending comparators scored and beaten, README fixed, the llama.cpp commit next to the Metal claim, and a line that the file is retired when Metal gains NVFP4. Then stop. No second version, no tuning.

### (b) Hybrid, and mirror

This is the cheapest experiment with the most to say. One of three things comes out.

1. **Hybrid clearly below UD-IQ3_XXS, and cand below mirror.** Our allocation is better at equal down in both columns. The whole gate loss was the down format. The map-to-allocator chain works. Then (c) has a prize: with a Metal kernel, a native-down file with our allocation could be the best file under the limit on every backend. That needs a new registered gate and a T score for the hybrid. Its allocation was fixed before T was seen, so T is still a fair test for it, but the decision to try is post hoc and must be said so.
2. **A tie.** The allocation is a wash against Unsloth's heuristic. The measured map bought nothing at this granularity. That is a useful negative for H3 planning: five blocks and tied passes are too coarse to beat a good prior.
3. **Hybrid above UD-IQ3_XXS, and mirror below cand.** Unsloth's allocation is better than ours. The Mac claim then rests only on the fact that nobody has re-encoded their file. I would not upload in that case, or I would upload with that sentence in the card.
4. **The two columns disagree.** Then the allocation interacts with the down format. Report it as that. For the Mac claim only the IQ4_NL column counts: mirror below cand is still bad news for it.

The same run prices the down format (cand minus hybrid, mirror minus UD). That number is the ceiling for (d) and the quality prize for (c).

Check before the run: the hybrid has a down scale and no gate_up scale on 20 layers. The generic scale pass in `llama-model.cpp` creates each projection's scale on its own and marks each optional (found in the source by this script: {yn(scale_opt)}). So the file should load and scale correctly. This was read, not run.

### (c) Metal NVFP4 kernel

- What exists: {pr['additions']} lines against a file layout that is gone.{" The sibling feasibility review puts the port at about a day plus testing and names one bug to fix on the way. My estimate agrees with it." if sib else ""} The MoE variants are in the patch. `CLAUDE.md` assigns llama.cpp patches to Claude, each with a test that fails without it, and Daniel reviews.
- What it would show here: UD-IQ3_XXS and the hybrid at Metal speed. Today a native-down scoring run costs {sec_ud:.1f} s per chunk against {sec_c:.1f} s. If thesis candidates use native NVFP4 as their 4.5-bit level, which section 2.1 says they should, every such run on this Mac pays that factor until the kernel exists.
- What it does to the release: it ends the candidate's reason to exist. That is the right outcome for users. A file whose value depends on a bug staying open is a weak thing to stand behind. A kernel helps every Mac user of every NVFP4 GGUF.
- Risk: correctness on 1152-wide rows and on the expert-indexed path, numerical parity with the CPU path, and upstream review time we do not control. A local patch is enough for our own runs.
- It is not thesis science. Guess: one to two working days to a tested local patch.

Verdict: not ahead of (b) and not ahead of G0 work. If (b) gives outcome 1, or if native NVFP4 becomes a thesis level, do a 4-hour spike: port, bit-compare against the CPU path, bench UD-IQ3_XXS.

### (d) A better NVFP4 to IQ4_NL re-encode

- Reserved for Daniel under `CLAUDE.md`. I give no design.
- Structure, from `ggml-common.h`: NVFP4 has one scale per {nv_sub} weights, IQ4_NL one per {iq4_blk} with a fixed non-linear codebook. Inference: an exact re-encode is not possible in general, so some loss stays.
- The current IQ4_NL down already has a low weight error ({t_dn4_all:.4f}). Guess: an encoder that knows the source grid recovers a part of the down cost, not all of it. If the down cost is near the gap ({f5(gap_S)} on S), a part of it does not flip the gate.
- It fits the thesis tool, so it will be written anyway. It should not be written for this release.

Verdict: park. Let (b) give the ceiling first.

### (e) Stop the release track and write up the per-pass result

- This is what the scoop check found nowhere else: the same quantized block on one pass at a time, an additivity residual, and a layer-split control without weight sharing.
- State today: Fisher r is {pp_res['r']:.3f} [{pp_res['ci95']['r'][0]:.3f}, {pp_res['ci95']['r'][1]:.3f}] for the pass split and {ls_res['r']:.3f} [{ls_res['ci95']['r'][0]:.3f}, {ls_res['ci95']['r'][1]:.3f}] for the layer split. The controls license one narrow claim on B2 at the 2-bit level. The three additivity comparators disagree. The 3-bit rung has no layer-split control. B3 is untouched. The Gaussian single-pass conditions were not run, so one Addendum A prediction is untested.
- That is a method and a first result, not yet a finding that carries a note. Each missing piece is a handful of all-Metal runs at about {eta_c:.0f} minutes each.
- Open items that block publication: Daniel's approval of the deviations in Addendum G section 2, his review of the delegated analysis code, and the major findings of `reviews/2026-11-21-code-review-analysis-opus.md` (the chunk parser and the missing tests on the gate path).
- The novelty has a clock too. One Mini repo already claims a visit-level sensitivity measurement, and another states the accumulation hypothesis as untested.

Verdict: this is where the hours should go.

### (f) Other things I see

- **f1. Publish the comparison table itself.** {len(order)} files ({len(order) - 2} public, 2 ours) scored on held-out text under one reference, with types and bytes read from the files. That has value without our model, and it is what the original rule promised after a FAIL.
- **f2. Mirror build.** Part of (b) above. It is the test of whether the Mac claim survives the obvious rival.
- **f3. Offer native NVFP4 as a level in the thesis allocator.** Same bytes as IQ4_NL down, no loss (section 2.1). This is a design note for Daniel, not a pilot run.
- **f4. Do not chase the unused {mb(LIMIT - size['cand'])}.** Spending it needs a reselection after T was seen.
- **f5. No task evaluation in the pilot.** It would be the right way to learn whether {gN['diff'].lstrip('-')} KL matters to a user. GSM8K test and all task sets are reserved for the thesis, so leave it.

## 4. Recommendation and order for the next 48 hours

The file has a real, small, short-lived edge. It is not worth more engineering. The pilot's lasting value is the per-pass method and what it taught about the down format.

1. **Tonight, no action.** Let the running chain finish the depth bench and the two pending comparators. If either beats the candidate on T1 and T2, the Mac-track rule fails and no model card goes out.
2. **Daniel, about 30 minutes.** Check what `logs/hf-upload-v1.log` points to and whether it is public. Approve or reject the Addendum G section 2 deviations. Write the dated entry for the mirror build and the 2 x 2 reading rule before any score exists.
3. **Run the 2 x 2 on S** (hybrid, then mirror; about {eta_ud + eta_c:.0f} minutes of scoring, 1 to 2 h in all). One heavy run at a time, after the chain is idle.
4. **Close the release track, at most 2 h.** Fix the README order and the "looped core" sentence. Then one of:
   - outcome 1 or 2 of (b) and comparators beaten: publish once, with the 2 x 2 in the details, and the retirement line.
   - mirror below cand (outcome 3 or 4): publish the measurements, and say that a re-encode of UD's layout does better. No model, or the model with that sentence.
5. **Hours 6 to 40: per-pass work (e), each step registered before its data.** In this order: the layer-split control at the 3-bit rung, the Gaussian single-pass conditions, two more independent-noise draws, then the pass split on B3. Fix the chunk parser finding first, since every CI runs through it.
6. **Hours 40 to 48: turn the per-pass section of `drafts/note.md` into a standalone draft** for Daniel's review. The release section shrinks to the table and the FAIL.
7. **Only if (b) gives outcome 1, or Daniel decides native NVFP4 is a thesis level:** the 4-hour kernel spike (c). Otherwise leave a note that PR #{pr['number']} exists and is small.
8. **(d) stays with Daniel** and waits for the ceiling from step 3.

What I would not do: reselect the candidate, build a second release, or spend the G0 run-up on a file that one upstream merge retires.

## 5. What I did not check

- Whether the Hugging Face repository in the upload log is public, and what it contains.
- Whether the hybrid loads and scores correctly. I read the scale pass in the loader and ran nothing.
- The body of PR #{pr['number']} beyond its metadata, file list and kernel names. I did not read the review comments.
- Any speed or quality figure on a machine other than this M5 Pro.
- `private/`, including the handoff, so thesis-fit judgments rest on `CLAUDE.md` and `CHANGELOG.md` only.
- The pending comparators. They were not scored when this was written ({len(pending_done)} of 2 in `kl.tsv`).
"""

assert "—" not in md
out = HERE.parent / "2026-11-21-direction-edge-fable.md"
out.write_text(md)
print(f"wrote {out} ({len(md.splitlines())} lines)")
