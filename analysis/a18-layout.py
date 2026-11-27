"""Amendment 18, descriptive: where ISTA's files spend their bytes compared with ours, from the GGUF headers only.

    python -I analysis/a18-layout.py > results/a18/layout.md

1. Experts per layer: gate_up and down type and bits per weight for each file.
2. Non-expert tensors by role: the type in each file and the bytes it costs, where the files differ.
"""

import re
from collections import defaultdict

from gguf import GGUFReader

FILES = ["v2c", "ista-IQ3_S", "ista-IQ3_XXS", "UD-IQ3_XXS", "c1", "ista-IQ2_S", "ista-Q2_K_XL"]


def load(name: str) -> dict:
    r = GGUFReader(f"models/{name}.gguf")
    return {t.name: (t.tensor_type.name, int(t.n_bytes), int(t.n_elements)) for t in r.tensors}


def bpw(x: tuple) -> float:
    return 8 * x[1] / x[2]


def main() -> None:
    f = {n: load(n) for n in FILES}
    print("# Amendment 18: byte layout of the compared files (headers only)\n")
    print("Built by `analysis/a18-layout.py`. Bits per weight include block scales.\n")
    for role in ("ffn_gate_up_exps", "ffn_down_exps"):
        print(f"## {role} per layer\n")
        print("| layer | " + " | ".join(FILES) + " |")
        print("|---" * (len(FILES) + 1) + "|")
        for i in range(24):
            k = f"blk.{i}.{role}.weight"
            print(f"| {i} | " + " | ".join(f"{f[n][k][0]} {bpw(f[n][k]):.2f}" for n in FILES) + " |")
        tot = {n: sum(v[1] for k, v in f[n].items() if role in k) for n in FILES}
        print("| GB | " + " | ".join(f"{tot[n] / 1e9:.3f}" for n in FILES) + " |\n")
    print("## Non-expert tensors by role (types where any file differs; GB summed over layers)\n")
    roles = defaultdict(lambda: defaultdict(lambda: [set(), 0]))
    for n in FILES:
        for k, (ty, nb, _) in f[n].items():
            if "exps.weight" in k or k.endswith(("exps.scale",)):
                continue
            role = re.sub(r"^blk\.\d+\.", "", k)
            roles[role][n][0].add(ty)
            roles[role][n][1] += nb
    print("| role | " + " | ".join(FILES) + " |")
    print("|---" * (len(FILES) + 1) + "|")
    tot = defaultdict(int)
    for role in sorted(roles):
        cells = {n: roles[role][n] for n in FILES}
        for n in FILES:
            tot[n] += cells[n][1]
        if len({"/".join(sorted(c[0])) for c in cells.values()}) == 1:
            continue
        print(f"| {role} | " + " | ".join(f"{'/'.join(sorted(c[0]))} {c[1] / 1e6:.0f}M" for c in cells.values()) + " |")
    print("| all non-expert GB | " + " | ".join(f"{tot[n] / 1e9:.3f}" for n in FILES) + " |\n")
    print("## Per-layer types of the mixed non-expert roles in ista-IQ3_S and v2c\n")
    for role in sorted(roles):
        ts = [f["ista-IQ3_S"].get(f"blk.{i}.{role}") for i in range(24)]
        if all(t is None for t in ts) or len({t[0] for t in ts if t}) <= 1:
            continue
        ours = [f["v2c"].get(f"blk.{i}.{role}") for i in range(24)]
        print(f"- {role}: ISTA " + " ".join(f"{i}:{t[0]}" for i, t in enumerate(ts) if t)
              + " | v2c " + " ".join(f"{i}:{t[0]}" for i, t in enumerate(ours) if t))


if __name__ == "__main__":
    main()
