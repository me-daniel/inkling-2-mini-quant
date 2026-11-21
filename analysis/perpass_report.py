"""Writes results/pilot/perpass.md: the per-pass test (CHANGELOG run 4, Addenda A, B, F) from its result files.

Inputs (results/pilot/): perpass-parity.txt, perpass-kl.tsv, perpass-residual.json (analysis/per_pass.py),
noise-t.tsv, noise-ratio.txt (analysis/noise.py), the registered prediction in reviews/*prediction-perpass*.md, and,
once scripts/perpass2.sh has made them, layersplit-residual.json, perpass3bit-residual.json and perpass2-kl.tsv.
If perpass-residual.json predates the full-vocabulary norm, the chunk 0-1 sensitivity table comes from the audit's
reread of the raw logits (reviews/*audit-perpass*/audit_check.json).

Wording rule (CHANGELOG card wording, Addendum F): the report states measurements and the card sentence only. It makes
no loop-specific claim and no H1 classification until the layer-split control has run. Once it has, verdict() applies
the Addendum F rule on the Fisher norm (disjoint 95% CIs for both r and the norm ratio) and picks the claim sentences by
its outcome. FORBIDDEN lists phrases a test keeps out of the output.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise  # noqa: E402

R = ROOT / "results/pilot"
CONDS = ("base", "noop", "p1", "p2", "p3", "all", "allslots", "noise-tied", "noise-indep")
STATS = (("r", "r = ||R|| / ||all||"), ("norm_ratio", "norm ratio ||all|| / ||sum_k d_k||"),
         ("pass1_over_pass3", "||d1|| / ||d3||"), ("pass2_over_pass3", "||d2|| / ||d3||"),
         ("share_last", "||d3|| / ||all||"), ("cos_12", "cos(d1, d2)"), ("cos_23", "cos(d2, d3)"),
         ("cos_13", "cos(d1, d3)"), ("visit_multiplier", "visit multiplier ||all||^2 / mean_k ||d_k||^2"),
         ("sq_over_sum", "||all||^2 / sum_k ||d_k||^2"))
NORMS = (("fisher", "Fisher"), ("head_euclid", "head Euclidean (substitute)"), ("full_euclid", "full vocabulary (floor included)"))
FORBIDDEN = ("once per visit", "per visit", "dominat", "compounds", "loop-aware", "pass-aware", "loop-specific effect")
CARD = ("On these documents, at P = 3, changing B2's experts on pass k produced the measured logit changes below.")
RULE = ("Addendum F: a loop-specific claim needs the pass-split r and norm ratio to differ from the layer-split values "
        "beyond both CIs.")


def kl_table(path: Path) -> dict[tuple[str, str], str]:
    return {(r["cond"], r["against"]): r["mean_kl"] for r in csv.DictReader(open(path), delimiter="\t")}


def kl_chunks(path: Path, against: str = "pp-base") -> dict[str, np.ndarray]:
    """Per-chunk KL of every condition against `against`; rows whose logits were deleted are left out."""
    out = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        if r["against"] == against:
            try:
                out[r["cond"]] = np.array([float(x) for x in r["per_chunk"].split(",")])
            except ValueError:
                pass
    return out


def with_derived(res: dict) -> dict:
    """Adds sq_over_sum = ||all||^2 / sum_k ||d_k||^2 = visit_multiplier / 3 (a positive rescaling, so its bootstrap
    percentiles are the multiplier's divided by 3)."""
    for d in res.values():
        if isinstance(d, dict) and "visit_multiplier" in d:
            d["sq_over_sum"] = d["visit_multiplier"] / 3
            d["ci95"]["sq_over_sum"] = [v / 3 for v in d["ci95"]["visit_multiplier"]]
    return res


def norms_in(*results: dict) -> list[tuple[str, str]]:
    return [(n, label) for n, label in NORMS if all(n in r for r in results)]


def direction(lo: float, hi: float) -> str:
    return "CI above 1" if lo > 1 else ("CI below 1" if hi < 1 else "CI contains 1")


def fmt_kl(v: str | None) -> str:
    if v is None:
        return "not run"
    try:
        return f"{float(v):.5f}"
    except ValueError:
        return v  # "deleted" rows: logits removed after the parity check


def ci(d: dict, k: str) -> str:
    lo, hi = d["ci95"][k]
    return f"{d[k]:.3f} [{lo:.3f}, {hi:.3f}]"


def stats_table(res: dict) -> list[str]:
    norms = norms_in(res)
    out = ["| Statistic | " + " | ".join(n for _, n in norms) + " |", "|---|" + "---|" * len(norms)]
    for k, label in STATS:
        out.append(f"| {label} | " + " | ".join(ci(res[n], k) for n, _ in norms) + " |")
    return out


def audit_sensitivity(path: Path) -> list[str]:
    """Chunk 0-1 table of the norm forms from the audit's reread of the raw logits (audit_check.json)."""
    a = json.loads(path.read_text())
    cs = sorted(a, key=int)
    forms = (("fisher", "Fisher"), ("full_euclid", "full vocabulary (floor included)"), ("head5", "head 5 nats"),
             ("head10", "head 10 nats (reported)"), ("head14", "head 14 nats"))
    keys = (("r", "r"), ("norm_ratio", "norm ratio"), ("p1_over_p3", "p1/p3"), ("p2_over_p3", "p2/p3"))
    out = [f"Norm sensitivity on chunks {' and '.join(cs)} only, from `{path.relative_to(ROOT)}` (the audit's CPU reread "
           "of the raw logits; no CI). A 16-chunk full-vocabulary column needs a CPU rerun of `analysis/per_pass.py` "
           "on the stored logits.", "",
           "| Norm | " + " | ".join(f"{lab} (c{', c'.join(cs)})" for _, lab in keys) + " |", "|---|" + "---|" * len(keys)]
    for f, lab in forms:
        out.append(f"| {lab} | " + " | ".join(", ".join(f"{a[c][f][k]:.3f}" for c in cs) for k, _ in keys) + " |")
    h = [a[c]["head10_entries"] for c in cs]
    out += ["", f"The head is unweighted. On these chunks a token's 10-nat head holds {min(x['min'] for x in h)} to "
            f"{max(x['max'] for x in h)} entries (median " + ", ".join(f"{x['median']:.0f}" for x in h) + "), so tokens "
            "with flat distributions weigh more in the head norm than in the Fisher norm.", ""]
    return out


def predictions(path: Path) -> list[list[tuple[float, float]]]:
    """Ranges a-b of each numbered item in section 6 ("Falsifiable registrations"), in order of appearance."""
    text = path.read_text(encoding="utf-8")
    sec = text[text.index("**Falsifiable registrations.**"):]
    sec = sec[: sec.index("\n7. ")] if "\n7. " in sec else sec
    items = re.split(r"\n\s+\d+\. ", sec)[1:]
    return [[(float(a), float(b)) for a, b in re.findall(r"(\d+\.\d+)\s*[–-]\s*(\d+\.\d+)", it)] for it in items]


def inside(v: float, rng: tuple[float, float]) -> str:
    return "inside" if rng[0] <= v <= rng[1] else "outside"


def apart(main: dict, other: dict, norm: str, k: str) -> tuple[bool, bool]:
    """Addendum F test for one statistic: (CIs do not overlap, each point lies outside the other's CI)."""
    (alo, ahi), (blo, bhi) = main[norm]["ci95"][k], other[norm]["ci95"][k]
    a, b = main[norm][k], other[norm][k]
    return ahi < blo or bhi < alo, not blo <= a <= bhi and not alo <= b <= ahi


def kl_sum_ratio(kl_all: str | None, parts: list[str | None]) -> str:
    """Mean-KL all / sum of the parts, point only (the per-chunk KL of the controls is not stored)."""
    try:
        return f"{float(kl_all) / sum(float(x) for x in parts):.3f}"
    except (TypeError, ValueError):
        return "not available"


def control_section(title: str, path: Path, main: dict, kl2: dict[str, str], conds: tuple[str, ...], note: str,
                    pending: str, labels: tuple[str, str] = ("pass split", "this split"),
                    kl_all: str | None = None) -> list[str]:
    out = [f"## {title}", ""]
    if not path.exists():
        return out + [f"Placeholder: filled when `{path.relative_to(ROOT)}` exists (scripts/perpass2.sh). {pending}", ""]
    res = with_derived(json.loads(path.read_text()))
    out += [note, "", f"Residual statistics from `{path.relative_to(ROOT)}` (95% bootstrap CI over chunks):", ""]
    out += stats_table(res) + [""]
    for n, name in norms_in(main, res):
        for k in ("r", "norm_ratio"):
            disjoint, points = apart(main, res, n, k)
            out.append(f"- {name} {k}: {labels[0]} {main[n][k]:.3f}, {labels[1]} {res[n][k]:.3f}; CIs "
                       + ("do not overlap." if disjoint else "overlap.")
                       + ("" if disjoint == points else " Each point lies outside the other's CI."))
    out += ["", "| Condition | KL vs base |", "|---|---|"]
    out += [f"| {c} | {fmt_kl(kl2.get(c))} |" for c in conds]
    parts = [kl2.get(c) for c in conds if c.endswith(("67", "89", "1011", "p1", "p2", "p3"))]
    out += ["", f"Mean KL all / sum of the three parts: {kl_sum_ratio(kl_all if kl_all is not None else kl2.get(conds[-1]), parts)} "
            "(point only; the per-chunk KL of the controls is not stored).", ""]
    return out


def verdict(main: dict, split_path: Path, rung_path: Path) -> list[str]:
    """What the controls license, by the Addendum F rule on the primary (Fisher) norm. Sentences are chosen by the
    outcome of the test, never typed per result."""
    out = ["## What the controls allow", ""]
    if not split_path.exists():
        return out + ["Not yet decidable: the layer-split control has not run. " + RULE, ""]
    split = with_derived(json.loads(split_path.read_text()))
    tests = {k: apart(main, split, "fisher", k) for k in ("r", "norm_ratio")}
    licensed = all(d and p for d, p in tests.values())
    mixed = any(d != p for d, p in tests.values())
    f, g = main["fisher"], split["fisher"]
    nums = (f"r {f['r']:.3f} against {g['r']:.3f}, norm ratio {f['norm_ratio']:.3f} against {g['norm_ratio']:.3f}")
    lines = [f"Rule as applied: under the Fisher norm, for both r and the norm ratio, the pass-split and layer-split 95% "
             f"CIs must not overlap ({nums}; pass split first)."
             + (" The two readings of \"beyond both CIs\" (disjoint CIs; each point outside the other's CI) disagree "
                "here, and the stricter one is applied." if mixed else ""), ""]
    which = [k.replace("_", " ") for k, (d, p) in tests.items() if d and p]
    if licensed:
        closer = f["r"] < g["r"] and abs(f["norm_ratio"] - 1) < abs(g["norm_ratio"] - 1)
        farther = f["r"] > g["r"] and abs(f["norm_ratio"] - 1) > abs(g["norm_ratio"] - 1)
        how = ("on these two statistics the pass split is nearer additive than the layer split (smaller r, norm ratio "
               "nearer 1)" if closer else
               "on these two statistics the pass split is farther from additive than the layer split (larger r, norm "
               "ratio farther from 1)" if farther else "the two statistics move in opposite directions relative to "
               "additivity")
        lines.append("The controls license one loop-specific claim at the 2-bit amplitude on B2: splitting the change "
                     "across passes on shared weights gives a different Fisher r and norm ratio than splitting it across "
                     f"layer pairs, which share no weights, and {how}. They do not license an H1 classification, any "
                     "statement about B3 or other amplitudes, or a choice among the additivity comparators.")
    else:
        lines.append("The controls license no loop-specific claim: "
                     + (f"only the {which[0]} clears the rule, and the rule needs both. " if which else
                        "neither r nor the norm ratio clears the rule. ")
                     + "The pass-split non-additivity is not shown to differ from that of a layer split with no weight "
                       "sharing, so it may be a property of perturbing B2 in parts at all rather than of the loop. "
                       "What remains is the measured statistics at the 2-bit amplitude on B2, with no H1 classification.")
    if rung_path.exists():
        rung = with_derived(json.loads(rung_path.read_text()))
        amp = {k: apart(main, rung, "fisher", k) for k in ("r", "norm_ratio")}
        moved = [k.replace("_", " ") for k, (d, p) in amp.items() if d and p]
        lines.append("The 3-bit rung (gate_up only, no layer-split control) licenses no loop-specific claim, and its "
                     "Fisher " + (" and ".join(moved) + (" differ" if len(moved) > 1 else " differs")
                     + " from the 2-bit pass split beyond both CIs, so the per-pass statistics depend on the level."
                     if moved else "r and norm ratio do not differ from the 2-bit pass split beyond both CIs, so the "
                     "per-pass statistics are not shown to depend on the level between these two rungs."))
    lines = lines[:2] + [" ".join(lines[2:])]
    return out + lines + [""]


def main() -> None:
    kl = kl_table(R / "perpass-kl.tsv")
    kc = kl_chunks(R / "perpass-kl.tsv")
    res = with_derived(json.loads((R / "perpass-residual.json").read_text()))
    noise_r = ast.literal_eval((R / "noise-ratio.txt").read_text().strip())
    pred_file = sorted((ROOT / "reviews").glob("*prediction-perpass*.md"))[0]
    pred = predictions(pred_file)
    kl2_path = R / "perpass2-kl.tsv"
    kl2 = {r["cond"]: r["kl_vs_base"] for r in csv.DictReader(open(kl2_path), delimiter="\t")} if kl2_path.exists() else {}
    split_path = R / "layersplit-residual.json"
    audit = sorted((ROOT / "reviews").glob("*audit-perpass*/audit_check.json"))

    t_rows = list(csv.DictReader(open(R / "noise-t.tsv"), delimiter="\t"))
    seeds = [k for k in t_rows[0] if k.startswith("t_seed")]
    t_mean = {kind: {k: sum(float(r[k]) for r in t_rows if r["kind"] == kind) / sum(r["kind"] == kind for r in t_rows)
                     for k in ("t_2bit", *seeds)} for kind in sorted({r["kind"] for r in t_rows})}
    t2 = ", ".join(f"{t_mean[k]['t_2bit']:.4f} on {k}" for k in t_mean)

    lines = ["# Per-pass test (pilot run 4, Addenda A, B, F)", "",
             "Written by `analysis/perpass_report.py`. Exploratory. Block B2 (layers 6-11) experts at the 2-bit level "
             "(IQ2_XXS gate_up, stock Q2_0 down) on chosen passes, on the IQ4_NL base, at P = 3, on S (16 x 4,096 tokens). "
             + CARD, "",
             "## Scope", "",
             f"- Amplitude-specific (Addendum F). The 2-bit level's mean relative l2 weight error is {t2} "
             "(`results/pilot/noise-t.tsv`, column t_2bit). Every 2-bit number below holds for this amplitude only.",
             "- B2 only. B2 is layers 6-11, half of the loop core (layers 6-17). B3 (layers 12-17) was not perturbed.",
             "- One noise draw per pass. noise-tied puts draw 1 on all three passes. noise-indep puts draws 1, 2 and 3 on "
             "passes 1, 2 and 3. The tied/independent CI covers chunk sampling only, not the draw.",
             (f"- Loop-specific reading only by the Addendum F rule. {RULE} The layer-split control has run; the "
              "outcome is under \"What the controls allow\"." if split_path.exists() else
              f"- No loop-specific reading. {RULE} ")
             + ("" if split_path.exists() else
                "The layer-split control has not run, so this report makes no loop-specific claim and no H1 "
                "classification."), "",
             "## Parity", "", (R / "perpass-parity.txt").read_text().strip(), "",
             "## KL per condition", "",
             "Mean KL over the 16 chunks from `results/pilot/perpass-kl.tsv` (`scripts/kldfile.py`).", "",
             "| Condition | KL vs IQ4_NL base | KL vs Q8_0 reference |", "|---|---|---|"]
    for c in CONDS:
        lines.append(f"| {c} | {fmt_kl(kl.get((c, 'pp-base')))} | {fmt_kl(kl.get((c, 'ref-S-P3')))} |")
    singles = [kc[c] for c in ("p1", "p2", "p3")]
    kr = {"all / sum of single passes": noise.paired_ratio([kc["all"]], singles),
          "all / mean single pass": noise.paired_ratio([kc["all"]], singles, scale=3.0),
          "p1 / p3": noise.paired_ratio([kc["p1"]], [kc["p3"]]),
          "p2 / p3": noise.paired_ratio([kc["p2"]], [kc["p3"]])}
    lines += ["", "KL ratios against the base, paired bootstrap over chunks (seed 0):", "",
              "| Ratio | Point [95% CI] | Per-chunk range |", "|---|---|---|"]
    for name, d in kr.items():
        lines.append(f"| {name} | {d['ratio']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] | "
                     f"{d['chunk_range'][0]:.3f}-{d['chunk_range'][1]:.3f} |")
    f, a = res["fisher"], kr["all / sum of single passes"]
    lines += ["", "Three additivity comparators, all against the sum of the single passes: KL all / sum "
              f"{a['ratio']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]; squared Fisher ||all||^2 / sum_k ||d_k||^2 "
              f"{ci(f, 'sq_over_sum')}; Fisher vector norm ratio ||all|| / ||sum_k d_k|| {ci(f, 'norm_ratio')}. "
              "Values above 1 read super-additive and values below 1 sub-additive, so the comparators disagree. "
              "This report does not choose among them. That choice belongs to the H1 analysis.", "",
              "## Residual statistics", "",
              "From `results/pilot/perpass-residual.json` (`analysis/per_pass.py`). d_k = logp(pass k only) - logp(base), "
              "R = d_all - sum_k d_k, pooled over all scored tokens as a ratio of sums. Point estimate [95% percentile "
              "bootstrap over chunks].", "",
              "Fisher is the primary norm. The registered prediction asks for centred Euclidean over the full vocabulary. "
              "That form cannot be computed faithfully from these files: the writer floors every value more than 16 nats "
              "below the top, and the floored entries hold the floor, not the model's log-probs. The head Euclidean "
              "column (entries within 10 nats of the base's top) is a head-restricted substitute for it. The "
              "full-vocabulary column, where present, keeps the floored entries as written. Three choices deviate "
              "from CHANGELOG.md: the Fisher primary norm, the 10-nat head, and a pooled ratio of sums instead of run 4's "
              "per-token ratio. They were fixed in code before the first per-pass logits existed but are not yet "
              "recorded in CHANGELOG.md (`reviews/2026-11-21-audit-perpass-opus.md`, finding 3).", ""]
    lines += stats_table(res)
    rc = res["fisher"]["r_chunks"]
    lines += ["", f"Fisher r per chunk: min {min(rc):.3f}, median {float(np.median(rc)):.3f}, max {max(rc):.3f} "
              f"({len(rc)} chunks).", ""]
    if "full_euclid" not in res and audit:
        lines += audit_sensitivity(audit[0])

    lines += ["## Tied against independent noise (Addendum A)", "",
              "Relative l2 error per expert tensor, mean over experts, from `results/pilot/noise-t.tsv`:", "",
              "| Tensor | 2-bit | " + " | ".join(s.replace("t_", "") for s in seeds) + " |", "|---|---|" + "---|" * len(seeds)]
    for kind, m in t_mean.items():
        lines.append(f"| {kind} | {m['t_2bit']:.4f} | " + " | ".join(f"{m[s]:.4f}" for s in seeds) + " |")
    lo, hi = noise_r["ci95"]
    lines += ["", f"From `results/pilot/noise-ratio.txt` (`analysis/noise.py`): KL tied {noise_r['kl_tied']:.5f}, "
              f"independent {noise_r['kl_indep']:.5f}, ratio {noise_r['ratio']:.3f} [{lo:.3f}, {hi:.3f}] (paired bootstrap "
              "over chunks)."]
    if "noise-tied" in kc:
        ta = noise.paired_ratio([kc["noise-tied"]], [kc["all"]])
        lines.append(f"KL tied / KL all (2-bit): {ta['ratio']:.3f} [{ta['ci95'][0]:.3f}, {ta['ci95'][1]:.3f}].")
    lines += ["", "Each condition is one draw per pass, so tying is confounded with which draws sit on passes 2 and 3. "
              "The CI covers chunk sampling only. Not yet run: noise-tied with draws 2 and 3 (two splices and two "
              "llama-perplexity runs); the spread of the three tied KLs bounds the draw effect.", "",
              "Addendum A also predicts independent ~ sum of single-pass effects. This prediction is not tested: no "
              "Gaussian single-pass condition was run, and the 2-bit single passes cannot stand in for one (registered "
              "prediction, point 7).", ""]

    e = res["head_euclid"]
    (r13, r23), (rr,), (rn,), (rq,) = pred[0][:2], pred[1][:1], pred[2][:1], pred[3][:1]
    rows = [("6.1 pass-1/pass-3 ratio", e["pass1_over_pass3"], e["ci95"]["pass1_over_pass3"], r13),
            ("6.1 pass-2/pass-3 ratio", e["pass2_over_pass3"], e["ci95"]["pass2_over_pass3"], r23),
            ("6.2 r", e["r"], e["ci95"]["r"], rr),
            ("6.3 norm ratio", e["norm_ratio"], e["ci95"]["norm_ratio"], rn),
            ("6.4 tied / independent KL (one draw per pass)", noise_r["ratio"], noise_r["ci95"], rq)]
    lines += ["## Registered predictions", "",
              f"From `{pred_file.relative_to(ROOT)}`, section 6. Quantization items scored on the head Euclidean "
              "substitute (the registered full-vocabulary form is not computable, see Residual statistics), the noise "
              "item on pooled mean KL. The head form is used only to score the registration.", "",
              "| Prediction | Range | Measured [95% CI] | Point | CI |", "|---|---|---|---|---|"]
    for name, v, (lo, hi), rg in rows:
        ci_s = "inside" if rg[0] <= lo and hi <= rg[1] else ("outside" if hi < rg[0] or lo > rg[1] else "straddles")
        lines.append(f"| {name} | {rg[0]:.2f}-{rg[1]:.2f} | {v:.3f} [{lo:.3f}, {hi:.3f}] | {inside(v, rg)} | {ci_s} |")
    lines += ["", "Pass ordering by measure. The registration predicts ||d3|| > ||d2|| > ||d1||, i.e. both ratios below 1. "
              "The pass ordering depends on the measure. KL ratios compare squared magnitudes (KL ~ 1/2 Fisher^2); the norm "
              "rows compare norms.", "",
              "| Measure | p1 / p3 [95% CI] | p2 / p3 [95% CI] |", "|---|---|---|"]
    order = [("KL (chunk means)", *[(kr[k]["ratio"], *kr[k]["ci95"]) for k in ("p1 / p3", "p2 / p3")])]
    order += [(name, *[(res[n][k], *res[n]["ci95"][k]) for k in ("pass1_over_pass3", "pass2_over_pass3")])
              for n, name in norms_in(res)]
    for name, *vals in order:
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f} [{lo:.3f}, {hi:.3f}], {direction(lo, hi)}" for v, lo, hi in vals)
                     + " |")
    lines += [""]

    tb = list(csv.DictReader(open(R / "t-blocks.tsv"), delimiter="\t")) if (R / "t-blocks.tsv").exists() else []
    t_b2 = {(r["pool"], r["kind"]): [] for r in tb}
    for r in tb:
        if 6 <= int(r["layer"]) <= 11:
            t_b2[(r["pool"], r["kind"])].append(float(r["t"]))
    amp3 = (f"On B2 the mean relative l2 error against the BF16 source is {np.mean(t_b2[('3bit', 'ffn_gate_up_exps')]):.4f} "
            f"for IQ3_XXS gate_up and {np.mean(t_b2[('2bit', 'ffn_gate_up_exps')]):.4f} for IQ2_XXS gate_up "
            "(`results/pilot/t-blocks.tsv`). " if t_b2.get(("3bit", "ffn_gate_up_exps")) else "")
    lines += control_section("Layer-split control", split_path, res, kl2, ("ls67", "ls89", "ls1011"),
                             "The 2-bit block on layers 6-7 / 8-9 / 10-11 only, all passes; the three sum to the `all` condition. "
                             "Here d1, d2, d3 are the three layer pairs, not passes. " + RULE,
                             RULE + " Until this section is filled, no loop-specific reading is licensed.",
                             kl_all=kl.get(("all", "pp-base")))
    lines += control_section("3-bit rung", R / "perpass3bit-residual.json", res, kl2, ("3b-p1", "3b-p2", "3b-p3", "3b-all"),
                             "Pass 1 / 2 / 3 / all at the 3-bit level (IQ3_XXS gate_up, down IQ4_NL as in the base). "
                             + amp3 + "The comparison lines below set it against the 2-bit pass split.",
                             "Until it is filled, every number above is specific to the 2-bit amplitude.",
                             labels=("2-bit", "3-bit"))
    lines += verdict(res, split_path, R / "perpass3bit-residual.json")
    text = "\n".join(lines).rstrip() + "\n"
    bad = [w for w in FORBIDDEN if w in text.lower()]
    if bad:
        raise SystemExit(f"report uses forbidden wording: {bad}")
    (R / "perpass.md").write_text(text, encoding="utf-8")
    print(text, end="")

if __name__ == "__main__":
    main()
