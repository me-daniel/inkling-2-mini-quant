"""Gate path (analysis/compare.py on results/pilot/kl.tsv): reproduce every comparison, add the sign count, a Student-t
interval and the effect of the 5-decimal running means in the logs; count T2 tokens. CPU only, reads small files.

    .venv/bin/python -I reviews/2026-11-21-code-review-analysis-opus/check_gate.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
import compare  # noqa: E402
import maptable  # noqa: E402

T975 = {15: 2.131449546, 4: 2.776445105}  # Student t 0.975 quantiles, df 15 and df 4 (no scipy in .venv)
runs = compare.load()
rng = np.random.default_rng(123)
print("== compare.py reproduced, with t-interval, sign count and re-rounding of the running means ==")
for t in ("T1", "T2"):
    for b in ("UD-IQ3_XXS", "uniform-IQ3_XXS", "bart-IQ2_S", "atomic-AD-IQ2_S", "apex-I-Nano", "bart-IQ2_M"):
        a, o = runs[f"final-cand-{t}-P3"]["chunks"], runs[f"final-{b}-{t}-P3"]["chunks"]
        d = a - o
        m, lo, hi = compare.paired(a, o)
        n = d.size
        se = d.std(ddof=1) / np.sqrt(n)
        tl, th = m - T975[n - 1] * se, m + T975[n - 1] * se
        run_m = {}
        for k in ("cand", b):
            text = (ROOT / f"logs/ppl-final-{k}-{t}-P3.log").read_text()
            run_m[k] = np.array([float(x.group(2)) for x in map(maptable.CHUNK.match, text.splitlines()) if x])
        widths = []
        for _ in range(200):  # each printed running mean is uniform within +-5e-6 of the truth
            rec = {}
            for k, mv in run_m.items():
                pm = mv + rng.uniform(-5e-6, 5e-6, mv.size)
                kk = np.arange(1, mv.size + 1)
                rec[k] = kk * pm - (kk - 1) * np.concatenate([[0.0], pm[:-1]])
            _, l2, h2 = compare.paired(rec["cand"], rec[b])
            widths.append(h2 - l2)
        print(f"{t} cand vs {b}: n={n} diff {m:+.5f} boot [{lo:+.5f}, {hi:+.5f}] (width {hi - lo:.5f}); "
              f"t [{tl:+.5f}, {th:+.5f}] (width {th - tl:.5f}, boot/t {(hi - lo) / (th - tl):.2f}); "
              f"cand worse on {int((d > 0).sum())}/{n}; boot width after re-rounding {np.mean(widths):.5f} "
              f"(sd {np.std(widths):.1e})")

print("\n== T2 tokens (HF tokenizer of the release; llama-perplexity drops the tail past whole chunks) ==")
from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(str(ROOT / "models/Inkling-2-Mini/tokenizer.json"))
nt = len(tok.encode((ROOT / "data/T2.txt").read_text(encoding="utf-8"), add_special_tokens=False).ids)
ch = nt // 2048
print(f"T2 tokens {nt}, chunks of 2048: {ch}, tokens past the last chunk: {nt - ch * 2048}, "
      f"scored positions (second half of each chunk): {ch * (2048 - 1 - 2048 // 2)}")

print("\n== compare.paired with unequal lengths ==")
print("paired(16 chunks of 0.02, 1 chunk of 0.03) ->", tuple(round(x, 6) for x in compare.paired(np.full(16, 0.02), np.array([0.03]))))

print("\n== stored compare.py lines vs this recomputation ==")
import re  # noqa: E402

n_ok, n_all = 0, 0
for f in ("gate-quality.txt", "mac-quality.txt", "mac-context-IQ2_M.txt"):
    for line in (ROOT / "results/pilot" / f).read_text().splitlines():
        m = re.match(r"(T[12]) cand vs (\S+): .*diff ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", line)
        if not m:
            continue
        t, b = m.group(1), m.group(2)
        got = compare.paired(runs[f"final-cand-{t}-P3"]["chunks"], runs[f"final-{b}-{t}-P3"]["chunks"])
        n_all += 1
        n_ok += tuple(f"{x:+.5f}" for x in got) == m.group(3, 4, 5)
print(f"{n_ok} of {n_all} stored comparison lines reproduced to the printed digit")
