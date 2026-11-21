"""Writes results/pilot/addendum-h.md: the Addendum H results with the reading rules registered in CHANGELOG.md.

    python analysis/addendum_h.py

Every section is rendered only when its inputs exist, so the script can run while the chain is still going.
Inputs: results/pilot/kl.tsv (hybrid, mirror, sel-2 and UD-IQ3_XXS on S), perpass-kl.tsv and perpass3-kl.tsv
(per-chunk KL against the IQ4_NL base), and the per_pass.py JSON files of each group.
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
from perpass_report import apart  # noqa: E402


def two_by_two(cand: tuple[float, float], ud: tuple[float, float], hybrid: tuple[float, float] | None,
               mirror: tuple[float, float] | None) -> list[str]:
    """H0 reading rule on (mean KL, SE) pairs. 2 SE of a difference uses both runs' SEs."""
    out = []
    if hybrid:
        d, se = hybrid[0] - ud[0], math.hypot(hybrid[1], ud[1])
        ok = d <= 2 * se
        out.append(f"hybrid minus UD-IQ3_XXS {d:+.5f} (2 SE = {2 * se:.5f}): "
                   + ("our layout is at least as good as UD's at equal down, and the down format explains the gap."
                      if ok else "our layout is worse than UD's even with UD's down projections."))
    if mirror:
        d, se = mirror[0] - cand[0], math.hypot(mirror[1], cand[1])
        better = d < -2 * se
        out.append(f"mirror minus cand {d:+.5f} (2 SE = {2 * se:.5f}): "
                   + ("UD's layout is better than ours at equal down too." if better
                      else "UD's layout is not shown to be better than ours at equal down."))
    return out


def all_above_one(ratios: list[float]) -> bool:
    """H4 rule: 'tied costs more' only if every tied/independent pair ratio is above 1."""
    return bool(ratios) and all(r > 1 for r in ratios)


def kl_rows() -> dict[str, dict]:
    rows = {}
    for r in csv.DictReader(open(R / "kl.tsv"), delimiter="\t"):
        rows[r["run"]] = {"kl": float(r["mean_kl"]), "se": float(r["se"])}
    return rows


def chunks() -> dict[str, np.ndarray]:
    out = {}
    for name in ("perpass-kl.tsv", "perpass3-kl.tsv"):
        p = R / name
        if not p.exists():
            continue
        for r in csv.DictReader(open(p), delimiter="\t"):
            if r["against"] != "pp-base":
                continue
            try:
                out[r["cond"]] = np.array([float(x) for x in r["per_chunk"].split(",")])
            except (ValueError, KeyError):
                pass
    return out


def load(name: str) -> dict | None:
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt_ci(d: dict) -> str:
    return f"{d['ratio']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def residual_table(rows: list[tuple[str, dict]]) -> list[str]:
    out = ["| Split | r (Fisher) | norm ratio (Fisher) |", "|---|---|---|"]
    for name, res in rows:
        f = res["fisher"]
        out.append(f"| {name} | {f['r']:.3f} [{f['ci95']['r'][0]:.3f}, {f['ci95']['r'][1]:.3f}] | "
                   f"{f['norm_ratio']:.3f} [{f['ci95']['norm_ratio'][0]:.3f}, {f['ci95']['norm_ratio'][1]:.3f}] |")
    return out


def main() -> None:
    kl, kc = kl_rows(), chunks()
    L = ["# Addendum H results", "", "Written by `analysis/addendum_h.py`. Reading rules as registered in `CHANGELOG.md` "
         "(Addendum H). KL on S at P = 3. Sections appear once their runs exist.", ""]

    # H0
    if "hybrid-S-P3" in kl or "mirror-S-P3" in kl:
        cand, ud = kl["sel-2-S-P3"], kl["UD-IQ3_XXS-S-P3"]
        h, m = kl.get("hybrid-S-P3"), kl.get("mirror-S-P3")
        L += ["## H0. The 2 x 2 on S", "", "| | our layout | UD's layout |", "|---|---|---|",
              f"| IQ4_NL down (Metal) | cand {cand['kl']:.5f} ± {cand['se']:.5f} | "
              + (f"mirror {m['kl']:.5f} ± {m['se']:.5f}" if m else "mirror (not run)") + " |",
              "| NVFP4 down (CPU on a Mac) | " + (f"hybrid {h['kl']:.5f} ± {h['se']:.5f}" if h else "hybrid (not run)")
              + f" | UD-IQ3_XXS {ud['kl']:.5f} ± {ud['se']:.5f} |", ""]
        L += [f"- {s}" for s in two_by_two((cand["kl"], cand["se"]), (ud["kl"], ud["se"]),
                                           (h["kl"], h["se"]) if h else None, (m["kl"], m["se"]) if m else None)] + [""]

    # amendment 2: which part of UD's layout
    if "ne-ud-S-P3" in kl or "gu-ud-S-P3" in kl:
        cand, mir = kl["sel-2-S-P3"], kl.get("mirror-S-P3")
        L += ["## Amendment 2. Which part of UD's layout", "", "| Build (S, P = 3) | KL | drop from cand | share of the cand-to-mirror drop |", "|---|---|---|---|"]
        for name, label in (("ne-ud-S-P3", "cand + UD's non-expert tensors"), ("gu-ud-S-P3", "cand + UD's gate_up experts")):
            if name in kl and mir:
                d = cand["kl"] - kl[name]["kl"]
                L.append(f"| {label} | {kl[name]['kl']:.5f} ± {kl[name]['se']:.5f} | {d:+.5f} | {d / (cand['kl'] - mir['kl']):.0%} |")
        L += [""]

    # H1
    pass3, split3 = load("perpass3bit-residual.json"), load("layersplit3bit-residual.json")
    if pass3 and split3:
        r_apart, n_apart = apart(pass3, split3, "fisher", "r")[0], apart(pass3, split3, "fisher", "norm_ratio")[0]
        L += ["## H1. Layer split at the 3-bit rung", ""] + residual_table([("pass split, 3-bit", pass3), ("layer split, 3-bit", split3)])
        L += ["", f"- Addendum F rule at 3 bits: r intervals {'do not overlap' if r_apart else 'overlap'}, norm-ratio intervals "
              f"{'do not overlap' if n_apart else 'overlap'}. "
              + ("The pass split differs from the layer split at 3 bits as well." if r_apart and n_apart
                 else "The 3-bit rung does not show the pass vs layer difference under the rule."), ""]

    # H2, H3
    tied_res, indep_res = load("noise-tied-residual.json"), load("noise-indep-residual.json")
    if tied_res or indep_res:
        L += ["## H2, H3. Gaussian noise decomposed by pass", ""]
        L += residual_table([(n, x) for n, x in (("tied (draw 1 on every pass)", tied_res), ("independent (draw k on pass k)", indep_res)) if x])
        L += [""]
        if all(k in kc for k in ("noise-indep", "g1-p1", "g2-p2", "g3-p3")):
            a = noise.paired_ratio([kc["noise-indep"]], [kc["g1-p1"], kc["g2-p2"], kc["g3-p3"]])
            inside = a["ci95"][0] <= 1 <= a["ci95"][1]
            L += [f"- Addendum A point 7: KL(independent) / sum of single-pass KLs = {fmt_ci(a)}. "
                  + ("About additive (the CI contains 1)." if inside else "Not additive (the CI excludes 1).")]
        if all(k in kc for k in ("noise-tied", "g1-p1", "g1-p2", "g1-p3")):
            t = noise.paired_ratio([kc["noise-tied"]], [kc["g1-p1"], kc["g1-p2"], kc["g1-p3"]])
            L += [f"- Tied: KL(tied) / sum of its single-pass KLs = {fmt_ci(t)}."]
        L += [""]

    # H4
    tied = [k for k in ("noise-tied", "noise-tied2", "noise-tied3") if k in kc]
    indep = [k for k in ("noise-indep", "indep-231", "indep-312") if k in kc]
    if len(indep) > 1:
        L += ["## H4. Tied against independent noise, every pair of draws", "",
              "| tied \\ independent | " + " | ".join(indep) + " |", "|---|" + "---|" * len(indep)]
        ratios = []
        for t in tied:
            cells = []
            for i in indep:
                r = noise.paired_ratio([kc[t]], [kc[i]])
                ratios.append(r["ratio"])
                cells.append(fmt_ci(r))
            L.append(f"| {t} | " + " | ".join(cells) + " |")
        L += ["", f"- {len(ratios)} ratios, from {min(ratios):.3f} to {max(ratios):.3f}. "
              + ("Tied noise costs more in every pair." if all_above_one(ratios) and len(ratios) == 9
                 else "The registered rule (all 9 above 1) is " + ("not yet testable." if len(ratios) < 9 else "not met.")), ""]

    # H5
    b3, b2 = load("perpass-b3-residual.json"), load("perpass-residual.json")
    if b3:
        L += ["## H5. The pass split on B3 (layers 12-17), 2-bit", ""] + residual_table([("B2 (on record)", b2), ("B3", b3)])
        if all(k in kc for k in ("b3-p1", "b3-p2", "b3-p3", "b3-all")):
            s = noise.paired_ratio([kc["b3-all"]], [kc["b3-p1"], kc["b3-p2"], kc["b3-p3"]])
            L += ["", f"- KL all / sum of single passes on B3 = {fmt_ci(s)}; per pass "
                  + ", ".join(f"p{k} {kc[f'b3-p{k}'].mean():.5f}" for k in (1, 2, 3)) + f", all {kc['b3-all'].mean():.5f}."]
        L += [""]

    (R / "addendum-h.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
