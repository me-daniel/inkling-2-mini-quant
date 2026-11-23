"""Addendum H amendment 16 reading: v2c (code added to the calibration) against v2m on the fresh set A2, with guards.

    python -I analysis/a16.py   (reads results/pilot/kl.tsv rows a2-<domain>-<model>-P3 and final-<model>-T1/T2-P3,
                                 and results/battery/B-scores.json; prints Markdown)

Ratios are geometric means of per-chunk KL, 95% CIs from analysis/battery_A.py's resampler (A2: whole documents of 4
chunks; T1 and T2: single chunks), 10,000 resamples, seed 0.
- H-C1 (primary): R over A2 python and rust of v2c / v2m. Pass: upper bound below 1.
- H-C2 (secondary): the same over A2 go and typescript, languages in no calibration text. Transfer: upper bound below 1.
- Guards: v2c / v2m on German (battery A's protocols, already used there), T1 and T2 (seen) each with upper
  bound at most 1.05.
- Tasks (H3 rule of amendment 15, v2c against v1 on the same 200 B1, B3 and B4 items): one-sided 95% lower bound above -5.
- Decision: H-C1 pass, every guard held and the task rule passed: v2c is proposed to replace v2m as the alternative
  file (publishing needs Daniel's go). Otherwise v2m stays.
- Amendment 17, default: the alternative file (v2c if it replaced v2m, else v2m) against v1, R over A2 python, rust,
  go and typescript; switch if the upper bound is below 1, no domain's upper bound is above 1.05 and its task rule
  passed. File names stay; publishing needs Daniel's go.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery_A import ratio_ci  # noqa: E402
from compare import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CODE, NEW, GUARD_A2 = ("python", "rust"), ("go", "typescript"), ("german",)
DOMAINS = CODE + NEW + GUARD_A2


def fmt(r) -> str:
    return f"{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]"


def tasks() -> tuple[str, bool | None]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("bb", ROOT / "analysis/battery_B.py")
    bb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bb)
    sc_p = ROOT / "results/battery/B-scores.json"
    sc = json.loads(sc_p.read_text()) if sc_p.exists() else {}
    if "v2c" not in sc:
        return "tasks: v2c not scored yet", None
    ids = sorted(i for i in sc["cand"] if i.split("-")[0] in ("B1", "B4") or i.startswith("Mbpp"))
    if not all(i in sc["v2c"] for i in ids) or len(ids) != 200:
        return f"tasks: v2c incomplete ({sum(i in sc['v2c'] for i in ids)} of {len(ids)})", None
    d, lo, hi, w, l = bb.h3([sc["v2c"][i] for i in ids], [sc["cand"][i] for i in ids])
    ok = lo > -5
    return (f"tasks (B1+B3+B4, n={len(ids)}): v2c - v1 = {d:+.1f} points, one-sided 95% bounds [{lo:+.1f}, {hi:+.1f}], "
            f"discordant {w} / {l} -> {'PASS' if ok else 'FAIL'}"), ok


def main() -> None:
    runs = load()
    a2 = lambda d, n: runs[f"a2-{d}-{n}-P3"]["chunks"]  # noqa: E731
    print("| set | v2c / v2m [95% CI] | v2c / v1 | v2m / v1 | v2c / UD | v2m / UD |\n|---|---|---|---|---|---|")
    for d in DOMAINS:
        tag = {"go": " (no calibration)", "typescript": " (no calibration)", "german": " (battery A's text, guard)"}.get(d, "")
        r = {k: ratio_ci({d: (a2(d, a), a2(d, b))}) for k, (a, b) in
             {"cm": ("v2c", "v2m"), "c1": ("v2c", "cand"), "m1": ("v2m", "cand"), "cu": ("v2c", "UD-IQ3_XXS"),
              "mu": ("v2m", "UD-IQ3_XXS")}.items()}
        print(f"| A2 {d}{tag} | {fmt(r['cm'])} | {r['c1'][0]:.3f} | {r['m1'][0]:.3f} | {r['cu'][0]:.3f} | {r['mu'][0]:.3f} |")
    guards = {}
    for t in ("T1", "T2"):
        c, m = runs[f"final-v2c-{t}-P3"]["chunks"], runs[f"final-v2m-{t}-P3"]["chunks"]
        guards[f"{t} (seen)"] = ratio_ci({t: (c, m)}, per_doc=1)
    for d in GUARD_A2:
        guards[f"A2 {d}"] = ratio_ci({d: (a2(d, "v2c"), a2(d, "v2m"))})
    print()
    c1 = ratio_ci({d: (a2(d, "v2c"), a2(d, "v2m")) for d in CODE})
    c2 = ratio_ci({d: (a2(d, "v2c"), a2(d, "v2m")) for d in NEW})
    h1 = c1[2] < 1
    print(f"H-C1 (A2 python, rust): v2c / v2m = {fmt(c1)} -> {'PASS' if h1 else 'FAIL'}")
    print(f"H-C2 (A2 go, typescript): v2c / v2m = {fmt(c2)} -> {'transfer' if c2[2] < 1 else 'no evidence of transfer'}")
    gok = all(g[2] <= 1.05 for g in guards.values())
    for k, g in guards.items():
        print(f"guard {k}: v2c / v2m = {fmt(g)} -> {'held' if g[2] <= 1.05 else 'BROKEN'}")
    line, tok = tasks()
    print(line)
    if tok is None:
        verdict = "pending (tasks)"
    else:
        verdict = "v2c replaces v2m as the alternative file (needs Daniel's go to publish)" if h1 and gok and tok else "v2m stays"
    print(f"\nDecision: {verdict}")
    alt = "v2c" if tok and h1 and gok else "v2m"
    four = CODE + NEW
    R = ratio_ci({d: (a2(d, alt), a2(d, "cand")) for d in four})
    per = {d: ratio_ci({d: (a2(d, alt), a2(d, "cand"))}) for d in four}
    task_ok = True if alt == "v2m" else tok  # v2m passed amendment 15's H3
    switch = R[2] < 1 and all(r[2] <= 1.05 for r in per.values()) and task_ok
    print(f"Amendment 17 ({alt} against v1, A2 code): R = {fmt(R)}; per domain "
          + ", ".join(f"{d} {fmt(r)}" for d, r in per.items())
          + f" -> {'switch the default to ' + alt + ' (needs Daniel' + chr(39) + 's go)' if switch else 'v1 stays the default'}")


if __name__ == "__main__":
    main()
