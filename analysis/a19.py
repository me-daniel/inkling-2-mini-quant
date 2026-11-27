"""Addendum H amendment 19: single changes on v2c (s1-s8), their paired KL deltas on S, and the subsets the
registered rule builds next.

    python -I analysis/a19.py               report (needs logs/ppl-a19-sN-S-P3.log and logs/ppl-v2c-S-P3.log)
    python -I analysis/a19.py spec s1       prints "DONOR REGEX" for scripts/a19.sh

Bytes come from the headers: for each change, donor bytes minus v2c bytes over the tensors its regex selects.
Metal bytes are file bytes minus token_embd (llama.cpp keeps the input layer on the CPU).
"""

import itertools
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

SMALL = "attn_sink_proj|attn_k_conv_proj|attn_v_conv_proj"
CHANGES = {
    "s1": ("models/c1.gguf", r"^blk\.[0-2]\.ffn_gate_up_exps\.weight$", "gate_up 0-2 IQ1_S (c1)"),
    "s2": ("models/c1.gguf", r"^blk\.(0|1|3)\.ffn_down_exps\.weight$", "down 0,1,3 Q2_0 (c1)"),
    "s3": ("models/Inkling-2-Mini-Q8_0.gguf", r"^blk\.([6-9]|1[0-7])\.attn_v\.weight$", "attn_v 6-17 Q8_0"),
    "s4": ("models/Inkling-2-Mini-Q8_0.gguf", r"^token_embd\.weight$", "token_embd Q8_0"),
    "s5": ("models/pool-3bit.gguf", r"^blk\.(6|7|8|12|18)\.ffn_gate_up_exps\.weight$", "gate_up 6,7,8,12,18 IQ3_XXS stock"),
    "s6": ("models/Inkling-2-Mini-BF16.gguf", rf"^blk\.[0-9]+\.({SMALL})\.weight$", "sink + conv projections BF16"),
    "s7": ("models/Inkling-2-Mini-BF16.gguf", rf"^blk\.[0-9]+\.({SMALL}|attn_rel)\.weight$", "s6 + attn_rel BF16"),
    "s8": ("models/tmp/a19-iq3s-donor.gguf", r"^blk\.(9|10|15)\.ffn_gate_up_exps\.weight$", "gate_up 9,10,15 IQ3_S stock"),
}
EXCLUSIVE = [{"s6", "s7"}]  # s7 contains s6
V2C_BYTES, UD_BYTES = 11_566_703_648, 11_663_310_784


def paired_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """mean(a) - mean(b) over paired chunks, with a chunk-bootstrap 95% CI."""
    d = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def subsets(singles: dict) -> list[tuple[float, int, int, tuple]]:
    """(predicted delta, file delta, Metal delta, names) for every valid subset; singles: name -> (delta, file, metal)."""
    out = []
    names = sorted(singles)
    for r in range(1, len(names) + 1):
        for c in itertools.combinations(names, r):
            if any(e <= set(c) for e in EXCLUSIVE):
                continue
            out.append((sum(singles[n][0] for n in c), sum(singles[n][1] for n in c), sum(singles[n][2] for n in c), c))
    return out


def pick(subs: list, rule: str, k: int = 2) -> list:
    if rule == "file":
        ok = [s for s in subs if s[1] <= 0]
    else:
        ok = [s for s in subs if s[2] <= 0 and V2C_BYTES + s[1] <= UD_BYTES]
    return sorted(ok)[:k]


def byte_delta(name: str) -> tuple[int, int]:
    from gguf import GGUFReader
    donor, rx, _ = CHANGES[name]
    base = {t.name: int(t.n_bytes) for t in GGUFReader(str(ROOT / "models/v2c.gguf")).tensors}
    don = {t.name: int(t.n_bytes) for t in GGUFReader(str(ROOT / donor)).tensors}
    sel = [n for n in base if re.search(rx, n)]
    assert sel, name
    d = sum(don[n] - base[n] for n in sel)
    return d, d - sum(don[n] - base[n] for n in sel if n == "token_embd.weight")


def main() -> None:
    if sys.argv[1:2] == ["spec"]:
        donor, rx, _ = CHANGES[sys.argv[2]]
        print(donor, rx)
        return
    from maptable import parse
    base = np.array(parse(ROOT / "logs/ppl-v2c-S-P3.log")["per_chunk"])
    singles, lines = {}, []
    lines += ["# Amendment 19: single changes on v2c, KL on S at P = 3", "",
              "Built by `analysis/a19.py` from `logs/ppl-a19-*-S-P3.log`; delta = KL(change) - KL(v2c), paired over S's "
              "16 chunks, chunk-bootstrap 95% CI. MB are decimal, from the headers.", "",
              f"v2c: KL {base.mean():.5f}", "",
              "| change | what | KL | delta [95% CI] | file MB | Metal MB |", "|---|---|---|---|---|---|"]
    for n, (_, _, what) in CHANGES.items():
        log = ROOT / f"logs/ppl-a19-{n}-S-P3.log"
        r = parse(log) if log.exists() else None
        if not r:
            lines.append(f"| {n} | {what} | missing | | | |")
            continue
        d, lo, hi = paired_delta(np.array(r["per_chunk"]), base)
        fb, mb = byte_delta(n)
        singles[n] = (d, fb, mb)
        lines.append(f"| {n} | {what} | {r['mean_kl']:.5f} | {d:+.5f} [{lo:+.5f}, {hi:+.5f}] | {fb / 1e6:+.1f} | {mb / 1e6:+.1f} |")
    lines += ["", "## Subsets the rule builds next (additive prediction)", ""]
    subs = subsets(singles)
    for rule in ("file", "metal"):
        for p, fb, mb, c in pick(subs, rule):
            lines.append(f"- {rule}: {'+'.join(c)}: predicted KL {base.mean() + p:.5f} (delta {p:+.5f}), "
                         f"file {fb / 1e6:+.1f} MB, Metal {mb / 1e6:+.1f} MB")
    picks = list(dict.fromkeys("+".join(c) for rule in ("file", "metal") for *_, c in pick(subs, rule)))
    if sys.argv[1:2] == ["picks"]:
        print(" ".join(picks))
        return
    measured = [(n, ROOT / f"logs/ppl-a19-{n}-S-P3.log") for n in picks]
    if any(log.exists() for _, log in measured):
        lines += ["", "## Subsets, measured on S", "", "| subset | KL | delta [95% CI] | beats v2c (upper < 0) |", "|---|---|---|---|"]
        for n, log in measured:
            r = parse(log) if log.exists() else None
            if r is None:
                lines.append(f"| {n} | not scored | | |")
                continue
            d, lo, hi = paired_delta(np.array(r["per_chunk"]), base)
            lines.append(f"| {n} | {r['mean_kl']:.5f} | {d:+.5f} [{lo:+.5f}, {hi:+.5f}] | {'yes' if hi < 0 else 'no'} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
