"""Amendment 18 reading: our released files against ISTA-DASLab's GSQ-RCO files and Unsloth UD-IQ3_XXS on fresh set A3.

    python -I analysis/a18.py   (reads results/pilot/kl.tsv rows a3-<domain>-<name>-P3, the p99 lines of the matching
                                 logs/ppl-*.log, and file sizes; prints Markdown)

G = exp(mean over domains of the per-domain mean of log(KL_a / KL_b) per chunk), symmetric (G(b, a) = 1 / G(a, b)).
95% CI from a paired, domain-stratified cluster bootstrap: per domain, resample its documents (4 chunks each) with
replacement, carrying both files, and recompute G; 10,000 resamples, seed 0. Outcomes (fixed in amendment 18):
upper bound < 1 and G <= 0.97: a closer; lower bound > 1 and G >= 1/0.97: b closer; CI inside [0.95, 1/0.95]:
equivalent at +-5%; otherwise inconclusive.
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ("python", "javascript", "german", "english", "math")
PER_DOC = 4
FILES = {  # name in logs -> (label, path for bytes)
    "cand": ("v1", "models/cand.gguf"),
    "v2m": ("v2m (refined)", "models/v2m.gguf"),
    "v2c": ("v2c (refined2)", "models/v2c.gguf"),
    "c1": ("c1 (8.9G)", "models/c1.gguf"),
    "ista-IQ3_S": ("ISTA IQ3_S", "models/ista-IQ3_S.gguf"),
    "ista-IQ3_XXS": ("ISTA IQ3_XXS", "models/ista-IQ3_XXS.gguf"),
    "ista-Q2_K_XL": ("ISTA Q2_K_XL", "models/ista-Q2_K_XL.gguf"),
    "ista-IQ2_S": ("ISTA IQ2_S", "models/ista-IQ2_S.gguf"),
    "UD-IQ3_XXS": ("Unsloth UD-IQ3_XXS", "models/UD-IQ3_XXS.gguf"),
}
PAIRS = [("P1", "v2c", "ista-IQ3_S"), ("P2", "c1", "ista-IQ2_S"), ("P3", "v2c", "UD-IQ3_XXS")]


def g_stat(pairs: dict[str, tuple[np.ndarray, np.ndarray]], n_boot: int = 10000, seed: int = 0, per_doc: int = PER_DOC):
    """pairs: domain -> (KL of a, KL of b) per chunk. Returns (G, lo, hi)."""
    rng = np.random.default_rng(seed)
    logs = {d: np.log(a / b).reshape(-1, per_doc) for d, (a, b) in pairs.items()}
    point = float(np.exp(np.mean([x.mean() for x in logs.values()])))
    acc = np.zeros(n_boot)
    for x in logs.values():
        idx = rng.integers(0, x.shape[0], size=(n_boot, x.shape[0]))
        acc += x[idx].mean(axis=(1, 2))
    boots = np.exp(acc / len(logs))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def outcome(g: float, lo: float, hi: float, a: str, b: str) -> str:
    if hi < 1 and g <= 0.97:
        return f"{a} closer"
    if lo > 1 and g >= 1 / 0.97:
        return f"{b} closer"
    if lo >= 0.95 and hi <= 1 / 0.95:
        return "equivalent at +-5%"
    return "inconclusive"


def p99(name: str, d: str) -> float:
    m = re.search(r"99\.0%\s+KLD:\s+([\d.]+)", (ROOT / f"logs/ppl-a3-{d}-{name}-P3.log").read_text(errors="replace"))
    return float(m.group(1)) if m else float("nan")


def main() -> None:
    runs = load()
    kl = lambda n, d: runs[f"a3-{d}-{n}-P3"]  # noqa: E731
    have = [n for n in FILES if all(f"a3-{d}-{n}-P3" in runs for d in DOMAINS)]
    print("## Absolute KL on A3 (mean per chunk; top-1 agreement %; p99 per-token KL)\n")
    print("| file | GB | " + " | ".join(DOMAINS) + " | token-weighted mean |\n|---|---|" + "---|" * (len(DOMAINS) + 1))
    for n in sorted(have, key=lambda n: (ROOT / FILES[n][1]).stat().st_size):
        gb = (ROOT / FILES[n][1]).stat().st_size / 1e9
        cells = [f"{kl(n, d)['chunks'].mean():.5f} ({kl(n, d)['top']:.1f}; {p99(n, d):.3f})" for d in DOMAINS]
        tw = np.concatenate([kl(n, d)["chunks"] for d in DOMAINS]).mean()
        print(f"| {FILES[n][0]} | {gb:.3f} | " + " | ".join(cells) + f" | {tw:.5f} |")
    print()
    for tag, a, b in PAIRS:
        if a not in have or b not in have:
            print(f"{tag}: {a} or {b} not scored yet\n")
            continue
        A, B = FILES[a][0], FILES[b][0]
        ga = (ROOT / FILES[a][1]).stat().st_size - (ROOT / FILES[b][1]).stat().st_size
        full = g_stat({d: (kl(a, d)["chunks"], kl(b, d)["chunks"]) for d in DOMAINS})
        noen = g_stat({d: (kl(a, d)["chunks"], kl(b, d)["chunks"]) for d in DOMAINS if d != "english"})
        per = {d: g_stat({d: (kl(a, d)["chunks"], kl(b, d)["chunks"])}) for d in DOMAINS}
        loo = []
        for d in DOMAINS:
            n_doc = kl(a, d)["chunks"].size // PER_DOC
            for k in range(n_doc):
                keep = np.ones(n_doc * PER_DOC, bool)
                keep[k * PER_DOC:(k + 1) * PER_DOC] = False
                sub = {e: (kl(a, e)["chunks"][keep] if e == d else kl(a, e)["chunks"],
                           kl(b, e)["chunks"][keep] if e == d else kl(b, e)["chunks"]) for e in DOMAINS}
                loo.append(float(np.exp(np.mean([np.log(x / y).mean() for x, y in sub.values()]))))
        fav = sum(v[0] < 1 for v in per.values())
        print(f"### {tag}: {A} / {B} (bytes {A} minus {B}: {ga / 1e6:+.0f} MB)\n")
        print(f"- G = {full[0]:.3f} [{full[1]:.3f}, {full[2]:.3f}] -> **{outcome(*full, A, B)}**")
        print(f"- leave-English-out G = {noen[0]:.3f} [{noen[1]:.3f}, {noen[2]:.3f}]; leave-one-document-out range "
              f"{min(loo):.3f} to {max(loo):.3f}; domains favouring {A}: {fav} of {len(DOMAINS)}")
        print("- per domain: " + ", ".join(f"{d} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for d, v in per.items()) + "\n")


if __name__ == "__main__":
    main()
