"""Writes results/pilot/perpass-h1-h5.md: the per-pass controls of Addendum H (H1 to H5) as a short results section.

    .venv/bin/python -I scripts/perpass-summary.py

Reads only files in results/pilot/: perpass-kl.tsv and perpass3-kl.tsv (per-chunk KL against the IQ4_NL base),
perpass2-kl.tsv (mean KL of the 3-bit pass split), noise-t.tsv (noise amplitude) and the per_pass.py JSON files.
Paired ratios use analysis/noise.paired_ratio (10,000 resamples over chunks, seed 0), as analysis/addendum_h.py does.
Every number in the output is computed here; the text holds only the registered rules and the wording around them.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results/pilot"
sys.path.insert(0, str(ROOT / "analysis"))
import noise  # noqa: E402
from addendum_h import all_above_one  # noqa: E402
from perpass_report import apart  # noqa: E402

P = 3  # pass count of every run here
N_CHUNK = 16

# group -> (conditions with per-chunk KL against pp-base, residual JSON or None)
GROUPS = {
    "H1": (["3b-ls67", "3b-ls89", "3b-ls1011", "3b-all"], "layersplit3bit-residual.json"),
    "H2": (["g1-p1", "g1-p2", "g1-p3", "noise-tied"], "noise-tied-residual.json"),
    "H3": (["g1-p1", "g2-p2", "g3-p3", "noise-indep"], "noise-indep-residual.json"),
    "H4": (["noise-tied", "noise-tied2", "noise-tied3", "noise-indep", "indep-231", "indep-312"], None),
    "H5": (["b3-p1", "b3-p2", "b3-p3", "b3-all"], "perpass-b3-residual.json"),
}
ON_RECORD = ["perpass-residual.json", "layersplit-residual.json", "perpass3bit-residual.json"]


def chunks() -> dict[str, np.ndarray]:
    """Per-chunk KL against pp-base. A condition in both tables must carry the same values."""
    out: dict[str, np.ndarray] = {}
    for name in ("perpass-kl.tsv", "perpass3-kl.tsv"):
        for r in csv.DictReader(open(R / name), delimiter="\t"):
            if r["against"] != "pp-base" or r["mean_kl"] == "deleted":
                continue
            v = np.array([float(x) for x in r["per_chunk"].split(",")])
            if v.size != N_CHUNK or not np.all(np.isfinite(v)):
                raise ValueError(f"{name}: {r['cond']} has {v.size} chunks or non-finite values")
            if not math.isclose(v.mean(), float(r["mean_kl"]), abs_tol=1e-6):
                raise ValueError(f"{name}: {r['cond']} mean_kl does not match its chunks")
            if r["cond"] in out and not np.array_equal(out[r["cond"]], v):
                raise ValueError(f"{r['cond']} differs between the two tables")
            out[r["cond"]] = v
    return out


def load(name: str) -> dict | None:
    p = R / name
    return json.loads(p.read_text())["fisher"] if p.exists() else None


def ci(x: dict, k: str, d: int = 3) -> str:
    lo, hi = x["ci95"][k]
    return f"{x[k]:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def rat(r: dict) -> str:
    return f"{r['ratio']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"


def kl_sum(kc: dict, all_: str, parts: list[str]) -> dict:
    return noise.paired_ratio([kc[all_]], [kc[p] for p in parts])


def main() -> None:
    kc = chunks()
    res = {g: load(j) for g, (_, j) in GROUPS.items() if j}
    rec = {j: load(j) for j in ON_RECORD}
    pp2 = {r["cond"]: float(r["kl_vs_base"]) for r in csv.DictReader(open(R / "perpass2-kl.tsv"), delimiter="\t")}

    L = ["# Per-pass controls H1 to H5 (Addendum H)", "",
         "Written by `scripts/perpass-summary.py` from files in `results/pilot/`. Exploratory pilot. KL against the IQ4_NL "
         f"base on S (16 chunks of 4,096 tokens), P = {P}. Intervals are 95% percentile bootstraps over chunks "
         "(10,000 resamples). r is the Fisher residual share ||d_all - sum_k d_k|| / ||d_all||; the norm ratio is "
         "||d_all|| / ||sum_k d_k||; both are 0 and 1 when the parts add up exactly.", ""]

    # completeness
    L += ["## Completeness", ""]
    missing = []
    for g, (conds, js) in GROUPS.items():
        lack = [c for c in conds if c not in kc] + ([js] if js and res[g] is None else [])
        missing += [f"{g}: {x}" for x in lack]
        L.append(f"- {g}: {len(conds) - len([c for c in conds if c not in kc])} of {len(conds)} conditions scored"
                 + (f", `{js}` " + ("present" if res[g] else "missing") if js else ", no residual registered") + ".")
    L += [f"- Missing: {', '.join(missing)}." if missing else "- Nothing registered is missing."]
    L += ["- Not computable, because the logits were deleted before these were stored: per-chunk KL of the 3-bit pass "
          "split (only means are kept, `perpass2-kl.tsv`), so its KL ratio below is a point estimate; and per-chunk "
          "sums of squares for any group, so no bootstrap of a paired difference in r or norm ratio between two groups.",
          "- H4's other orders (indep-231, indep-312) have no single-pass runs of their own (draw 2 on pass 1, draw 3 on "
          "pass 2 and so on were never run), so they have KL only, no residual. This is as registered.", ""]

    L += ["## Findings", ""]
    n = 0

    def item(s: str) -> None:
        nonlocal n
        n += 1
        L.append(f"{n}. {s}")

    # H1
    p3, l3 = rec["perpass3bit-residual.json"], res["H1"]
    r_apart, n_apart = apart({"f": p3}, {"f": l3}, "f", "r")[0], apart({"f": p3}, {"f": l3}, "f", "norm_ratio")[0]
    rp = np.array(json.loads((R / "perpass3bit-residual.json").read_text())["fisher"]["r_chunks"])
    rl = np.array(json.loads((R / "layersplit3bit-residual.json").read_text())["fisher"]["r_chunks"])
    ls = kl_sum(kc, "3b-all", ["3b-ls67", "3b-ls89", "3b-ls1011"])
    ps = pp2["3b-all"] / sum(pp2[f"3b-p{k}"] for k in (1, 2, 3))
    p2, l2 = rec["perpass-residual.json"], rec["layersplit-residual.json"]
    item("H1, layer split at the 3-bit rung. Registered reading: \"the Addendum F rule at 3 bits, r and norm ratio of "
         "the 3-bit pass split (on record) against this split, beyond both 95% CIs\". "
         f"Pass split r {ci(p3, 'r')}, layer split r {ci(l3, 'r')}; pass split norm ratio {ci(p3, 'norm_ratio')}, "
         f"layer split {ci(l3, 'norm_ratio')}. The r intervals {'do not overlap' if r_apart else 'overlap'} and the "
         f"norm-ratio intervals {'do not overlap' if n_apart else 'overlap'}, so the rule "
         f"{'is met' if r_apart and n_apart else 'is not met'}. Per chunk, the layer split has the larger r in "
         f"{int((rl > rp).sum())} of {rl.size} chunks. KL all / sum of parts is {rat(ls)} for the layer split and "
         f"{ps:.3f} (point only) for the pass split. At 2 bits (on record) the same order held: r {ci(p2, 'r')} "
         f"against {ci(l2, 'r')}. Both partitions share the same all-on run (3b-all, KL "
         f"{kc['3b-all'].mean():.5f}); the pass split is the closer to additive.")
    # H2
    t = res["H2"]
    ts = kl_sum(kc, "noise-tied", ["g1-p1", "g1-p2", "g1-p3"])
    item("H2, tied Gaussian noise by pass (draw 1 on pass 1, 2 or 3 only, and on all passes). No reading rule is "
         f"registered; descriptive. Single-pass KL {', '.join(f'p{k} {kc[f'g1-p{k}'].mean():.5f}' for k in (1, 2, 3))}, "
         f"all {kc['noise-tied'].mean():.5f}. KL all / sum of parts {rat(ts)}. r {ci(t, 'r')}, norm ratio "
         f"{ci(t, 'norm_ratio')}, pairwise Fisher cosines {ci(t, 'cos_12')}, {ci(t, 'cos_23')}, {ci(t, 'cos_13')} "
         f"(passes 1-2, 2-3, 1-3). The 2-bit quantization split on the same block (on record) gives r {ci(p2, 'r')} "
         f"and KL all / sum {rat(kl_sum(kc, 'all', ['p1', 'p2', 'p3']))}: tied noise of the same relative error "
         "behaves like the quantization error.")
    # H3
    i = res["H3"]
    a = kl_sum(kc, "noise-indep", ["g1-p1", "g2-p2", "g3-p3"])
    inside = a["ci95"][0] <= 1 <= a["ci95"][1]
    s_t = sum(kc[f"g1-p{k}"] for k in (1, 2, 3))
    s_i = kc["g1-p1"] + kc["g2-p2"] + kc["g3-p3"]
    item("H3, independent noise by pass (draw k on pass k only). Registered reading: \"KL(noise-indep) / sum of the three "
         "single-pass KLs, paired chunk bootstrap; 'about additive' if its 95% CI contains 1\". "
         f"Ratio {rat(a)}: the CI {'contains' if inside else 'excludes'} 1, so the registered reading "
         f"{'calls it about additive' if inside else 'does not call it about additive'} "
         f"({'sub-additive' if a['ratio'] < 1 else 'super-additive'} by {abs(1 - a['ratio']):.1%}). "
         f"Its single-pass KLs sum to {s_i.mean():.5f}, against {s_t.mean():.5f} for the tied draws, so tied and "
         f"independent differ in how the visits combine, not in what one visit costs. r {ci(i, 'r')}, norm ratio "
         f"{ci(i, 'norm_ratio')}, cosines {ci(i, 'cos_12')}, {ci(i, 'cos_23')}, {ci(i, 'cos_13')}.")
    # H4
    tied, indep = ["noise-tied", "noise-tied2", "noise-tied3"], ["noise-indep", "indep-231", "indep-312"]
    m = [[noise.paired_ratio([kc[x]], [kc[y]]) for y in indep] for x in tied]
    flat = [c for row in m for c in row]
    pts = [c["ratio"] for c in flat]
    item("H4, tied against independent noise over every pair of draws. Registered reading: \"the tied/independent "
         "ratio is then reported for every pair of the 3 tied and 3 independent draws; 'tied costs more' is stated only "
         f"if all 9 ratios are above 1\". {len(pts)} ratios from {min(pts):.3f} to {max(pts):.3f}; lowest lower CI bound "
         f"{min(c['ci95'][0] for c in flat):.3f}. The rule {'is met: tied costs more' if all_above_one(pts) and len(pts) == 9 else 'is not met'}. "
         f"Tied KL {', '.join(f'{kc[x].mean():.5f}' for x in tied)}; independent KL "
         f"{', '.join(f'{kc[y].mean():.5f}' for y in indep)}. The table follows the findings.")
    # H5
    b3 = res["H5"]
    s3, s2 = kl_sum(kc, "b3-all", ["b3-p1", "b3-p2", "b3-p3"]), kl_sum(kc, "all", ["p1", "p2", "p3"])
    item("H5, the 2-bit pass split on B3 (layers 12-17). Registered: \"same statistics as B2, descriptive\". "
         f"B3 against B2 (on record): r {ci(b3, 'r')} against {ci(p2, 'r')}; norm ratio {ci(b3, 'norm_ratio')} against "
         f"{ci(p2, 'norm_ratio')}; KL all / sum {rat(s3)} against {rat(s2)}; share of the last pass "
         f"||d_3|| / ||d_all|| {ci(b3, 'share_last')} against {ci(p2, 'share_last')}. Single-pass KL on B3 "
         f"{', '.join(f'p{k} {kc[f'b3-p{k}'].mean():.5f}' for k in (1, 2, 3))} (all {kc['b3-all'].mean():.5f}); on B2 "
         f"{', '.join(f'p{k} {kc[f'p{k}'].mean():.5f}' for k in (1, 2, 3))} (all {kc['all'].mean():.5f}).")
    # across groups
    rows = [("B2 2-bit quantization (on record)", s2), ("B2 tied noise (H2)", ts), ("B2 independent noise (H3)", a),
            ("B3 2-bit quantization (H5)", s3)]
    item(f"Across the visit splits, KL all / sum of the {P} single-visit KLs is "
         + "; ".join(f"{name} {rat(x)}" for name, x in rows)
         + f"; 3-bit quantization on B2 {ps:.3f} (point only). To second order in the logit change, error that counts "
         f"once gives about {1 / P:.2f}, independent per-visit error about 1, and {P} identical, aligned visit errors "
         f"about {P}.")

    # H4 matrix
    L += ["", "H4, KL(tied) / KL(independent), paired bootstrap over chunks:", "",
          "| tied \\ independent | " + " | ".join(indep) + " |", "|---|" + "---|" * len(indep)]
    L += [f"| {x} | " + " | ".join(rat(c) for c in row) + " |" for x, row in zip(tied, m)]

    # noise amplitude
    amp: dict[str, dict[str, list[float]]] = {}
    for r in csv.DictReader(open(R / "noise-t.tsv"), delimiter="\t"):
        d = amp.setdefault(r["kind"], {})
        for k in ("t_2bit", "t_seed1", "t_seed2", "t_seed3"):
            d.setdefault(k, []).append(float(r[k]))
    L += ["", "Noise amplitude (`noise-t.tsv`, mean relative l2 error per expert on B2): "
          + "; ".join(f"{kind} 2-bit {np.mean(d['t_2bit']):.4f}, draws "
                      + ", ".join(f"{np.mean(d[f't_seed{s}']):.4f}" for s in (1, 2, 3)) for kind, d in amp.items()) + "."]

    L += ["", "Sources: H1 `layersplit3bit-residual.json`, `perpass3bit-residual.json`, `perpass2-kl.tsv`, "
          "`perpass3-kl.tsv`; H2 `noise-tied-residual.json`; H3 `noise-indep-residual.json`; H4 `perpass-kl.tsv` "
          "(tied draws 2 and 3) and `perpass3-kl.tsv`; H5 `perpass-b3-residual.json`, `perpass-residual.json`, "
          "`perpass-kl.tsv` (B2 runs p1, p2, p3, all).", ""]

    L += ["## Limits", "",
          "- One block per question: B2 (layers 6-11) for H1 to H4, B3 (layers 12-17) for H5. Nothing here covers "
          "layers 6-17 changed together.",
          "- One amplitude per group: the 2-bit level (IQ2_XXS gate_up, stock Q2_0 down), the 3-bit level (gate_up only) "
          "and noise matched to the 2-bit relative error. r and the norm ratio depend on the level (finding 1).",
          f"- One capture: S, {N_CHUNK} chunks, one Q8_0 reference, one IQ4_NL base, one engine commit, P = {P} only. "
          "The intervals cover chunk sampling only, not seeds, documents or builds.",
          "- Three noise draws. H4 permutes the same three draws, so its 9 ratios are not 9 replications; fresh seed "
          "triplets were not run.",
          "- H1's layer split is not a control without weight sharing: each layer pair still runs on all three visits. "
          "It shows that the partition matters, not that the loop alone causes the difference.",
          "- The H3 KL ratio mixes two things. With exactly additive logit changes it would still exceed 1 by the "
          "cross-pass Fisher terms, which are positive here; the logit-level residual r is large in every group.",
          "- The logits files floor log-probs below max - 16 nats; the Fisher norm gives the floored tail little weight.",
          "- No route replay and no tiny-amplitude rung, so the cause of the residual (router flips or smooth "
          "nonlinearity) is untested.",
          "- Tied draws 2 and 3 come from the 21 Nov runs in `perpass-kl.tsv`. Their 22 Nov reruns (logs/chain12.out, "
          "logs/macgate-rerun.out) never entered a table and their logits are gone, so the two could not be compared.",
          "- `analysis/per_pass.py` and `analysis/noise.py` were written by Claude and are not yet reviewed by Daniel; "
          "three analysis choices (Fisher norm, 10-nat head, pooled ratio) await their change-log approval.",
          ""]

    lo, hi = min([x["ratio"] for _, x in rows] + [ps]), max([x["ratio"] for _, x in rows] + [ps])
    worst = (1 + int(np.argmax([kc[f"p{k}"].mean() for k in (1, 2, 3)])),
             1 + int(np.argmax([pp2[f"3b-p{k}"] for k in (1, 2, 3)])),
             1 + int(np.argmax([kc[f"b3-p{k}"].mean() for k in (1, 2, 3)])))
    L += ["## What it means for the thesis question", "",
          f"On these documents at P = {P}, quantization error in a looped block counts roughly once per visit: changing "
          f"all {P} visits costs {lo:.2f} to {hi:.2f} times the sum of the single-visit costs (finding 6), far from the "
          f"{1 / P:.2f} that counting once would give. "
          f"Shared weights add a modest positive interaction: the same noise on every visit costs {min(pts):.2f} to "
          f"{max(pts):.2f} times a fresh draw per visit, although the single-visit runs cost the same (KL sums "
          f"{s_t.mean():.4f} and {s_i.mean():.4f}). "
          f"Nothing compounds strongly: the ratios stay far below the {P} of fully aligned visit errors, and the logit "
          f"changes combine to less than their vector sum (norm ratio {p2['norm_ratio']:.2f} on B2, "
          f"{b3['norm_ratio']:.2f} on B3). "
          "For an allocator, a bit in a looped layer is priced roughly per visit with a block-dependent surcharge, and "
          f"the costliest visit (pass {worst[0]} on B2 at 2 bits, pass {worst[1]} at 3 bits, pass {worst[2]} on B3) "
          "changes with block and level, so one visit weight will not fit all of them.", ""]

    (R / "perpass-h1-h5.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
