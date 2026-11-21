"""Prints every number cited in reviews/2026-11-21-audit-perpass-opus.md, from the result files, the run logs and
the two audit outputs in this folder (audit_check.json: chunks 0-1; audit_kl-chunk0.json).

    .venv/bin/python -I reviews/2026-11-21-audit-perpass-opus/audit_numbers.py
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
R = ROOT / "results/pilot"

kl = {(r["cond"], r["against"]): r for r in csv.DictReader(open(R / "perpass-kl.tsv"), delimiter="\t")}
g = lambda c, a="pp-base": np.array([float(x) for x in kl[(c, a)]["per_chunk"].split(",")])  # noqa: E731
p1, p2, p3, al, nt, ni = (g(c) for c in ("p1", "p2", "p3", "all", "noise-tied", "noise-indep"))
n = p1.size
idx = np.random.default_rng(0).integers(0, n, size=(10000, n))


def boot(f):
    b = np.array([f(i) for i in idx])
    return f(np.arange(n)), np.percentile(b, 2.5), np.percentile(b, 97.5)


def loo(f):
    v = [f(np.delete(np.arange(n), j)) for j in range(n)]
    return min(v), max(v)


ratio_sum = lambda i: al[i].mean() / (p1[i] + p2[i] + p3[i]).mean()  # noqa: E731
tied_indep = lambda i: nt[i].mean() / ni[i].mean()  # noqa: E731
print("[KL ratios, perpass-kl.tsv, paired bootstrap over chunks, seed 0]")
print("  all/sum %.3f [%.3f, %.3f]" % boot(ratio_sum))
print("  all/mean %.3f [%.3f, %.3f]" % boot(lambda i: 3 * ratio_sum(i)))
print("  p1/p3 %.3f [%.3f, %.3f]" % boot(lambda i: p1[i].mean() / p3[i].mean()))
print("  p2/p3 %.3f [%.3f, %.3f]" % boot(lambda i: p2[i].mean() / p3[i].mean()))
print("  tied/all-2bit %.3f [%.3f, %.3f]" % boot(lambda i: nt[i].mean() / al[i].mean()))
print("  leave-one-chunk-out all/sum %.3f-%.3f, tied/indep %.3f-%.3f" % (*loo(ratio_sum), *loo(tied_indep)))

res = json.loads((R / "perpass-residual.json").read_text())
print("[perpass-residual.json]")
for nm in ("fisher", "head_euclid"):
    d = res[nm]
    r, nr = d["r"], d["norm_ratio"]
    cos = (1 + 1 / nr**2 - r**2) * nr / 2  # ||all|| = 1, ||sum|| = 1/nr, ||R||^2 = 1 + 1/nr^2 - 2 cos / nr
    print(f"  {nm}: cos(all, sum) {cos:.3f}; r if all were parallel to the sum {abs(1 / nr - 1):.3f}; "
          f"||all||^2 / sum_k ||d_k||^2 {d['visit_multiplier'] / 3:.3f}")
    for k in ("pass1_over_pass3", "pass2_over_pass3"):
        print(f"    {k} {d[k]:.3f} [{d['ci95'][k][0]:.3f}, {d['ci95'][k][1]:.3f}]")
rc = sorted(res["fisher"]["r_chunks"])
print(f"  Fisher r_chunks: report's 'median' sorted[{len(rc) // 2}] = {rc[len(rc) // 2]:.3f}, true median {np.median(rc):.3f}")

ll = {r["run"]: r for r in csv.DictReader(open(R / "kl.tsv"), delimiter="\t")}
f = ll["floor-iq4nl-S-P3"]
print("[estimators] Q8_0 -> IQ4_NL: llama-perplexity (kl.tsv) %s, kldfile (perpass-kl.tsv) %s; chunk 0: %s vs %.6f" % (
    f["mean_kl"], kl[("base", "ref-S-P3")]["mean_kl"], f["per_chunk"].split(",")[0], g("base", "ref-S-P3")[0]))
k0 = json.loads((HERE / "audit_kl-chunk0.json").read_text())
for key, v in k0.items():
    if key != "chunk":
        print(f"  chunk 0 {key}: kldfile {v['kldfile']:.6f}, proxy {v['llama_proxy']:.6f}, shift {100 * (v['llama_proxy'] / v['kldfile'] - 1):+.2f}%")
for est in ("kldfile", "llama_proxy"):
    s = sum(k0[f"pp-base->pp-{c}"][est] for c in ("p1", "p2", "p3"))
    print(f"  chunk 0 {est}: all/sum {k0['pp-base->pp-all'][est] / s:.4f}, tied/indep "
          f"{k0['pp-base->pp-noise-tied'][est] / k0['pp-base->pp-noise-indep'][est]:.4f}")

a = json.loads((HERE / "audit_check.json").read_text())
print("[audit_check.json, chunks 0 and 1]")
for c in ("0", "1"):
    x = a[c]
    print(f"  chunk {c}: n_rows {x['n_rows']}; stored Fisher r {res['fisher']['r_chunks'][int(c)]}, recomputed {x['fisher']['r']:.4f}; "
          f"stored head r {res['head_euclid']['r_chunks'][int(c)]}, recomputed {x['head10']['r']:.4f}")
    print(f"    KL recomputed " + ", ".join(f"{k} {v:.6f}" for k, v in x["kl_kldfile"].items())
          + " | table " + ", ".join(f"{k} {g(k)[int(c)]:.6f}" for k in x["kl_kldfile"]))
    print(f"    floor: base mass on floored entries {x['tail_mass_mean']:.4f}, rows with clipping {x['rows_with_clipping']:.3f}; "
          f"max Fisher share of entries floored in S only {max(x['cross_fisher_share'].values()):.5f}")
    print(f"    Fisher r all entries {x['fisher']['r']:.4f}, clean entries only {x['fisher_clean']['r']:.4f}")
    print("    half Fisher / KL " + ", ".join(f"{k} {v:.3f}" for k, v in x["half_fisher_over_kl"].items()))
    print(f"    per-token Fisher r median {x['per_token_r_fisher']['median']:.3f} (p90 {x['per_token_r_fisher']['p90']:.3f}); "
          f"top 1% tokens hold {x['top1pct_residual_energy_share']:.3f} of residual energy")
    print(f"    head-10 entries per token: min {x['head10_entries']['min']}, median {x['head10_entries']['median']:.0f}, max {x['head10_entries']['max']}")
    for nm in ("fisher", "full_euclid", "head5", "head10", "head14"):
        y = x[nm]
        print(f"    {nm:11s} r {y['r']:.3f}  norm ratio {y['norm_ratio']:.3f}  p1/p3 {y['p1_over_p3']:.3f}  p2/p3 {y['p2_over_p3']:.3f}")

t = list(csv.DictReader(open(R / "noise-t.tsv"), delimiter="\t"))
seeds = [k for k in t[0] if k.startswith("t_seed")]
for kind in ("ffn_gate_up_exps", "ffn_down_exps"):
    d = np.array([[float(r[s]) / float(r["t_2bit"]) - 1 for s in seeds] for r in t if r["kind"] == kind])
    print(f"[noise-t.tsv] {kind}: achieved/target - 1 over {d.shape[0]} experts x {d.shape[1]} seeds: "
          f"mean {100 * d.mean():+.2f}%, min {100 * d.min():+.2f}%, max {100 * d.max():+.2f}%")


def ppl(c):
    s = (ROOT / f"logs/ppl-pp-{c}.log").read_text()
    return re.findall(r"\[\d+\]([\d.]+),", s)


for x, y in (("noop", "base"), ("allslots", "all"), ("p1", "base")):
    print(f"[logs] per-chunk PPL of pp-{x} equals pp-{y} in all chunks: {ppl(x) == ppl(y)} ({len(ppl(x))} chunks)")

# consistency of the pooled statistics: rebuild the norm ratio and the visit multiplier from norms and cosines
for nm in ("fisher", "head_euclid"):
    d = res[nm]
    n3 = d["share_last"]
    nk = {1: d["pass1_over_pass3"] * n3, 2: d["pass2_over_pass3"] * n3, 3: n3}
    cs = {(1, 2): d["cos_12"], (2, 3): d["cos_23"], (1, 3): d["cos_13"]}
    s2 = sum(v**2 for v in nk.values()) + 2 * sum(c * nk[a] * nk[b] for (a, b), c in cs.items())
    print(f"[consistency] {nm}: norm ratio stored {d['norm_ratio']:.4f}, rebuilt {1 / np.sqrt(s2):.4f}; "
          f"visit multiplier stored {d['visit_multiplier']:.4f}, rebuilt {3 / sum(v**2 for v in nk.values()):.4f}")

import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import kldfile  # noqa: E402

h = kldfile.read_header(ROOT / "models/logits/pp-base.kld")
print(f"[header] n_vocab {h.n_vocab}, n_ctx {h.n_ctx}, n_chunk {h.n_chunk}, scored per chunk {h.n_scored}, scored total {h.n_chunk * h.n_scored}")
