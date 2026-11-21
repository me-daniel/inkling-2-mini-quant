"""Numbers for reviews/2026-11-21-metal-nvfp4-feasibility.md. Usage: python -I facts.py REPO PRDIR"""
import json, re, sys, math
from pathlib import Path

repo, prdir = Path(sys.argv[1]), Path(sys.argv[2])
up = repo / "vendor/llama.cpp-upstream"

# block sizes from ggml-common.h (struct layouts read by hand from the header, sizes computed here)
hdr = (up / "ggml/src/ggml-common.h").read_text()
def define(name):
    return int(re.search(rf"#define {name} (\d+)", hdr).group(1))
qk_nv, qk_nv_sub, qk_mx, qk_nl = define("QK_NVFP4"), define("QK_NVFP4_SUB"), define("QK_MXFP4"), define("QK4_NL")
size = {
    "nvfp4": qk_nv // qk_nv_sub + qk_nv // 2,   # uint8 d[QK/SUB] + qs[QK/2]
    "mxfp4": 1 + qk_mx // 2,                     # uint8 e + qs[QK/2]
    "iq4_nl": 2 + qk_nl // 2,                    # ggml_half d + qs[QK/2]
}
qk = {"nvfp4": qk_nv, "mxfp4": qk_mx, "iq4_nl": qk_nl}
for t in size:
    print(f"{t}: block {qk[t]} elems, {size[t]} bytes, {8*size[t]/qk[t]:.4f} bpw, block 4-byte aligned: {size[t] % 4 == 0}")

# mul_mv lane use on the 1152-wide down rows: 16 lane pairs, one block per pair per step
ne00 = 1152
for t in ("nvfp4", "iq4_nl"):
    nb = ne00 // qk[t]
    steps = math.ceil(nb / 16)
    print(f"{t}: ne00 {ne00} -> {nb} blocks/row, {steps} steps of 16 lane pairs, lane use {nb/(16*steps):.3f}")
print(f"ne00 % 128 == 0 (mul_mv_ext gate): {ne00 % 128 == 0}; ne00 % 64 == 0: {ne00 % 64 == 0}")

# PR #20456 diff stats per file
diff = (prdir / "20456.diff").read_text().splitlines()
cur, stats = None, {}
for ln in diff:
    if ln.startswith("diff --git"):
        cur = ln.split(" b/")[-1]; stats[cur] = [0, 0, 0]
    elif ln.startswith("@@"):
        stats[cur][2] += 1
    elif ln.startswith("+") and not ln.startswith("+++"):
        stats[cur][0] += 1
    elif ln.startswith("-") and not ln.startswith("---"):
        stats[cur][1] += 1
for f, (a, d, h) in stats.items():
    print(f"PR20456 {f}: +{a} -{d}, {h} hunks")
print("PR20456 total: +%d -%d" % (sum(v[0] for v in stats.values()), sum(v[1] for v in stats.values())))
for n in (20456, 20060):
    p = json.loads((prdir / f"pr{n}.json").read_text())
    print(f"PR{n}: state {p['state']}, merged {p['merged']}, draft {p['draft']}, +{p['additions']} -{p['deletions']}, "
          f"{p['changed_files']} files, mergeable_state {p['mergeable_state']}, created {p['created_at']}, updated {p['updated_at']}, base {p['base']['sha'][:9]}")

# bench rows (llama-bench, M5 Pro, build 9f5b9d6)
def bench(name):
    txt = (repo / "results/pilot" / name).read_text()
    rows = {m.group(1): float(m.group(2)) for m in re.finditer(r"\|\s+(pp512|tg128)\s+\|\s+([\d.]+) ±", txt)}
    off = re.search(r"cannot multiply: (\d+)", txt).group(1)
    return rows, off
b = {k: bench(f"bench-{k}.txt") for k in ("cand", "UD-IQ3_XXS", "uniform-IQ3_XXS")}
for k, (r, off) in b.items():
    print(f"bench {k}: pp512 {r['pp512']}, tg128 {r['tg128']}, expert tensors off Metal {off}")
for t in ("pp512", "tg128"):
    print(f"UD-IQ3_XXS / cand {t}: {b['UD-IQ3_XXS'][0][t]/b['cand'][0][t]:.3f}")

# merge times of merged "Apple Metal" PRs created since 2026-03-01 (GitHub search, saved in PRDIR)
import datetime as dt, statistics as st
items = []
for f in ("metal_merged.json", "metal_merged2.json"):
    items += json.loads((prdir / f).read_text())["items"]
def d(s): return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
for name, sel in (("ggerganov", lambda a: a == "ggerganov"), ("other authors", lambda a: a != "ggerganov")):
    xs = sorted((d(i["pull_request"]["merged_at"]) - d(i["created_at"])).total_seconds() / 86400 for i in items if sel(i["user"]["login"]))
    q = st.quantiles(xs, n=4)
    print(f"merge days {name}: n {len(xs)}, median {st.median(xs):.1f}, q1 {q[0]:.1f}, q3 {q[2]:.1f}, p90 {st.quantiles(xs, n=10)[8]:.1f}")
p = json.loads((prdir / "pr20456.json").read_text())
now = dt.datetime(2026, 11, 21, tzinfo=dt.timezone.utc)
print(f"PR20456 open {(now - d(p['created_at'])).days} days, last update {(now - d(p['updated_at'])).days} days ago")
