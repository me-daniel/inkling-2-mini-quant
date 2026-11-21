"""KL on one chunk by two estimators: kldfile (both sides renormalised over all entries, floors included) and a proxy
for llama-perplexity's own KL (base side: file probabilities with log p > -16, not renormalised; other side:
normalised over its unfloored entries, standing in for the exact logits). The proxy is validated on Q8_0 -> IQ4_NL
against llama-perplexity's printed per-chunk value in results/pilot/kl.tsv.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
CHUNK = int(sys.argv[2])
sys.path.insert(0, str(ROOT / "scripts"))
import kldfile  # noqa: E402

K = ROOT / "models/logits"
PAIRS = [("ref-S-P3", "pp-base")] + [("pp-base", f"pp-{c}") for c in ("p1", "p2", "p3", "all", "noise-tied", "noise-indep")]
names = sorted({n for pr in PAIRS for n in pr})
h = kldfile.read_header(K / "pp-base.kld")
raws = {n: np.memmap(K / f"{n}.kld", dtype="<u2", mode="r", offset=h.data_offset, shape=(h.n_chunk, h.n_scored, h.nv)) for n in names}
ROWS = 32
acc = {f"{a}->{b}": [0.0, 0.0] for a, b in PAIRS}
for s in range(0, h.n_scored, ROWS):
    dec = {}
    for n in names:
        block = np.asarray(raws[n][CHUNK, s : s + ROWS])
        params = np.ascontiguousarray(block[:, :4]).view("<f4")
        codes = block[:, 4 : 4 + h.n_vocab]
        lp = params[:, 1:2].astype(np.float64) + params[:, 0:1].astype(np.float64) * codes.astype(np.float64)
        dec[n] = (lp, codes == 0)
    for a, b in PAIRS:
        la, _ = dec[a]
        lq, fq = dec[b]
        p = np.exp(la)
        pn = p / p.sum(axis=1, keepdims=True)
        lqn = lq - np.log(np.exp(lq).sum(axis=1, keepdims=True))
        k1 = float(np.sum(pn * (np.log(pn) - lqn)))
        lq_clean = lq - np.log(np.where(fq, 0.0, np.exp(lq)).sum(axis=1, keepdims=True))
        k2 = float(np.sum(np.where(la > -16.0, p * (la - lq_clean), 0.0)))
        acc[f"{a}->{b}"][0] += k1
        acc[f"{a}->{b}"][1] += k2
out = {k: {"kldfile": v[0] / h.n_scored, "llama_proxy": v[1] / h.n_scored} for k, v in acc.items()}
print(json.dumps({"chunk": CHUNK, **out}, indent=1))
