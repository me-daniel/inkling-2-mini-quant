"""v2m against v1 (cand) speed and memory from scripts/bench-v2m.sh: results/pilot/{bench3,depth3}-NAME.txt and
logs/depth3-NAME.log (/usr/bin/time -l). Prints a Markdown table; t/s as llama-bench reports it (mean ± sd)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R, L = ROOT / "results/pilot", ROOT / "logs"


def rows(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) == 7 and re.match(r"(pp|tg)\d", c[5]):
            out[c[5]] = c[6]
    return out


def main() -> None:
    names = ("cand", "v2m")
    data = {n: {**rows(R / f"depth3-{n}.txt"), **{f"{k} (bench3)": v for k, v in rows(R / f"bench3-{n}.txt").items()}} for n in names}
    mem = {n: re.search(r"(\d+)\s+peak memory footprint", (L / f"depth3-{n}.log").read_text()).group(1) for n in names}
    rss = {n: re.search(r"(\d+)\s+maximum resident set size", (L / f"depth3-{n}.log").read_text()).group(1) for n in names}
    print(f"llama.cpp {(R / 'bench3-commit.txt').read_text().strip()}, -ngl 99, t/s mean ± sd\n")
    print("| test | v1 | v2m |\n|---|---|---|")
    for k in data["cand"]:
        print(f"| {k} | {data['cand'][k]} | {data['v2m'].get(k, 'n/a')} |")
    print(f"| peak memory footprint (GB) | {int(mem['cand']) / 1e9:.2f} | {int(mem['v2m']) / 1e9:.2f} |")
    print(f"| max resident set (GB, includes the mapped file) | {int(rss['cand']) / 1e9:.2f} | {int(rss['v2m']) / 1e9:.2f} |")


if __name__ == "__main__":
    main()
