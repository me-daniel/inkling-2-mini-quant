"""Allocation for the release candidate (Addenda A, C, D, E). Written by Claude at Daniel's request; Daniel reviews.

Model (the linearity theorem, arXiv 2411.17525): a build's KL is the sum of what each choice adds on its own, measured
in the block map against the Q8_0 reference. Within a block the two projections are also taken as additive
(Addendum E): cost(gate_up, down) = KL(3bit) + dGU(gate_up) + dDN(down), where each change comes from the map level
that differs from 3bit in that projection only. Every combination over 5 blocks, 4 gate_up and 2 down types and 2
non-expert types is enumerated, so the optimum is exact under the model; bytes are exact (scripts/budget.py arithmetic).
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import budget  # noqa: E402

BLOCKS = {"B1": range(0, 6), "B2": range(6, 12), "B3": range(12, 18), "B4": range(18, 20), "B5": range(20, 24)}
# type label -> GGML type for bytes; "Q2_0O" is our optimized encoding, stored as Q2_0
GGML = {"IQ3_XXS": "IQ3_XXS", "IQ2_S": "IQ2_S", "IQ2_XXS": "IQ2_XXS", "Q2_0O": "Q2_0", "IQ4_NL": "IQ4_NL"}
# gate_up option -> (map level with it, map level it is compared with); down likewise
GU = {"IQ3_XXS": None, "IQ2_S": ("gu2s", "3bit"), "IQ2_XXS": ("gu2xxs", "3bit")}
DN = {"IQ4_NL": None, "Q2_0O": ("dnq2o", "3bit")}
BUDGET = 11_663_310_784 - 11_977_760  # Unsloth UD-IQ3_XXS file bytes minus the constant GGUF overhead


def measured() -> dict[str, float]:
    return {r["run"]: float(r["mean_kl"]) for r in csv.DictReader(open(ROOT / "results/pilot/kl.tsv"), delimiter="\t")}


def tensor_table() -> tuple[dict, int, int]:
    """Elements of each layer's gate_up and down experts, plus the bytes of everything else at Q8_0 and at Q6_K."""
    el, rest = {}, {"Q8_0": 0, "Q6_K": 0}
    for t in budget.gguf.GGUFReader(budget.REF).tensors:
        n = int(t.n_elements)
        if t.tensor_type in (budget.Q.F32, budget.Q.F16, budget.Q.BF16):
            rest["Q8_0"] += int(t.n_bytes)
            rest["Q6_K"] += int(t.n_bytes)
        elif ".ffn_gate_up_exps." in t.name or ".ffn_down_exps." in t.name:
            el[(int(t.name.split(".")[1]), "gu" if "gate_up" in t.name else "dn")] = n
        else:
            rest["Q8_0"] += budget.tensor_bytes(n, budget.Q.Q8_0)
            rest["Q6_K"] += budget.tensor_bytes(n, budget.Q.Q6_K)
    return el, rest["Q8_0"], rest["Q6_K"]


def main(top: int = 8, n_pick: int = 4) -> None:
    kl = measured()
    el, rest_q8, rest_q6 = tensor_table()

    def m(b: str, lvl: str) -> float | None:
        return kl.get(f"map-{b}-{lvl}-S-P3")

    options = {}  # block -> list of (gu, dn, predicted KL, bytes)
    for b, layers in BLOCKS.items():
        base = m(b, "3bit")
        opts = []
        for gu, src in GU.items():
            dgu = 0.0 if src is None else (None if m(b, src[0]) is None or m(b, src[1]) is None else m(b, src[0]) - m(b, src[1]))
            for dn, srd in DN.items():
                ddn = 0.0 if srd is None else (None if m(b, srd[0]) is None else m(b, srd[0]) - base)
                if base is None or dgu is None or ddn is None:
                    continue
                nbytes = sum(budget.tensor_bytes(el[(l, "gu")], budget.Q[GGML[gu]]) + budget.tensor_bytes(el[(l, "dn")], budget.Q[GGML[dn]]) for l in layers)
                opts.append((gu, dn, base + dgu + ddn, nbytes))
        options[b] = opts
    if any(not o for o in options.values()):
        sys.exit(f"map incomplete: {[b for b, o in options.items() if not o]}")

    ne = {"Q8_0": (0.0, rest_q8)}
    if m("NE", "Q6K") is not None:
        ne["Q6_K"] = (m("NE", "Q6K"), rest_q6)
    best = []
    for combo in itertools.product(*options.values()):
        kl_exp = sum(o[2] for o in combo)
        by_exp = sum(o[3] for o in combo)
        for ne_t, (ne_kl, ne_bytes) in ne.items():
            if by_exp + ne_bytes <= BUDGET:
                best.append((kl_exp + ne_kl, by_exp + ne_bytes, ne_t, combo))
    best.sort(key=lambda r: r[0])
    for pred, nbytes, ne_t, combo in best[:top]:
        print(f"predicted KL {pred:.5f}  bytes {nbytes:,} ({(BUDGET - nbytes) / 1e6:.0f} MB spare)  NE {ne_t}  {describe(combo)}")
    # the prediction is only additive, so phase 4 builds and measures several structurally different allocations
    # (each differs from every earlier pick in at least 2 of the 11 choices) and keeps the measured best
    picks = []
    for pred, nbytes, ne_t, combo in best:
        key = [ne_t] + [x for o in combo for x in o[:2]]
        if all(sum(u != v for u, v in zip(key, k)) >= 2 for k, _ in picks):
            picks.append((key, {"predicted_kl": pred, "bytes": nbytes, "ne": ne_t, "choice": {b: list(o[:2]) for b, o in zip(BLOCKS, combo)}}))
        if len(picks) == n_pick:
            break
    out = [c for _, c in picks]
    for c in out:
        if budget_price(rules_of(c)) != c["bytes"]:
            sys.exit("byte check failed: the per-block sum and the full pricing disagree")
    (ROOT / "results/pilot/allocation-top.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} candidates to results/pilot/allocation-top.json")


def describe(combo) -> str:
    return " ".join(f"{b}:{o[0]}/{o[1]}" for b, o in zip(BLOCKS, combo))


def rules_of(c: dict) -> dict:
    rules = []
    for b, layers in BLOCKS.items():
        alt = "|".join(str(l) for l in layers)
        gu, dn = c["choice"][b]
        rules += [[rf"^blk\.({alt})\.ffn_gate_up_exps\.weight$", GGML[gu]], [rf"^blk\.({alt})\.ffn_down_exps\.weight$", GGML[dn]]]
    return {"default": c["ne"], "rules": rules}


# where the bytes of each choice come from (identical to what llama-quantize writes with the same imatrix)
DONOR = {("gu", "IQ3_XXS"): "pool-3bit.gguf", ("gu", "IQ2_S"): "pool-gu2s.gguf", ("gu", "IQ2_XXS"): "pool-2bit.gguf",
         ("dn", "IQ4_NL"): "pool-3bit.gguf", ("dn", "Q2_0O"): "q2opt/L{l}.gguf"}


def splice_args(c: dict, models: str) -> str:
    out = []
    for b, layers in BLOCKS.items():
        for (kind, name), t in zip((("gu", "ffn_gate_up_exps"), ("dn", "ffn_down_exps")), c["choice"][b]):
            for l in layers:
                out += ["--take", f"{models}/{DONOR[(kind, t)].format(l=l)}", rf"^blk\.{l}\.{name}\.weight$"]
    return " ".join(out)


def budget_price(alloc: dict) -> int:
    rules = [(budget.re.compile(rx), budget.Q[t.upper()]) for rx, t in alloc["rules"]]
    default = budget.Q[alloc["default"].upper()]
    total = 0
    for t in budget.gguf.GGUFReader(budget.REF).tensors:
        if t.tensor_type in (budget.Q.F32, budget.Q.F16, budget.Q.BF16):
            total += int(t.n_bytes)
        else:
            total += budget.tensor_bytes(int(t.n_elements), next((q for rx, q in rules if rx.search(t.name)), default))
    return total


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--splice-args":  # --splice-args INDEX MODELS_DIR
        print(splice_args(json.loads((ROOT / "results/pilot/allocation-top.json").read_text())[int(sys.argv[2])], sys.argv[3]))
    else:
        main()
