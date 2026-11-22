"""Check that every measured number in the paper comes from a file the run scripts wrote.

    .venv/bin/python -I paper/check-numbers.py [--logs DIR]      # numbers in paper/main.tex against the sources
    .venv/bin/python -I paper/check-numbers.py --diff OLD NEW    # same numbers, same counts, in two versions

Sources: results/ (every text file), CHANGELOG.md (dated records written from the logs), README.md, docs/report.md
and, if given, the run logs (not in git). A number counts as found if it appears verbatim (ignoring a leading "+"
and thousands separators), or if it has at least 3 decimals and a source number with more decimals rounds to it.
Small integers and numbers without a decimal point or separator are skipped: counts, layer and pass indices.
DERIVED lists the few numbers the text computes from sourced ones; each carries its derivation.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?:e-?\d+)?(?![\w])")

DERIVED = {
    "11.5": "v1 is 11,514,054,944 bytes (results/pilot/gate.md), 11.5 GB in decimal units",
}


def body(tex: str) -> list[tuple[int, str]]:
    """Lines after \\begin{document}, LaTeX comments removed, with their line numbers."""
    lines = tex.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("\\begin{document}"))
    out, tikz = [], False
    for i, l in enumerate(lines[start + 1:], start + 2):
        tikz = tikz or "\\begin{tikzpicture}" in l  # drawing coordinates, not measurements
        if not tikz:
            out.append((i, re.sub(r"(?<!\\)%.*", "", l)))
        tikz = tikz and "\\end{tikzpicture}" not in l
    return out


def tokens(tex: str) -> list[tuple[int, str]]:
    out = []
    for n, line in body(tex):
        line = re.sub(r"\\(ref|label|cite[pt]?|nolinkurl|code)\{[^}]*\}", " ", line)  # hashes and keys, not numbers
        line = line.replace("\\,", " ").replace("{,}", ",")
        for m in NUM.finditer(line):
            t = m.group(0).rstrip(",")
            if "." in t or "," in t or "e-" in t:
                out.append((n, t))
    return out


def norm(t: str) -> str:
    return t.lstrip("+").replace(",", "")


def sources(logs: Path | None) -> str:
    files = [p for p in (ROOT / "results").rglob("*") if p.is_file() and p.suffix in {".md", ".tsv", ".txt", ".json"}]
    files += [ROOT / "CHANGELOG.md", ROOT / "README.md", ROOT / "docs/report.md"]
    if logs:
        files += [p for p in logs.rglob("*") if p.is_file() and p.stat().st_size < 50_000_000]
    return "\n".join(p.read_text(errors="replace") for p in files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path)
    ap.add_argument("--diff", nargs=2, type=Path)
    a = ap.parse_args()
    if a.diff:
        old, new = (Counter(norm(t) for _, t in tokens(p.read_text())) for p in a.diff)
        lost, added = old - new, new - old
        for k, v in sorted(lost.items()):
            print(f"lost  {k} x{v}")
        for k, v in sorted(added.items()):
            print(f"added {k} x{v}")
        print("numbers identical" if not (lost or added) else "numbers differ")
        return 1 if lost or added else 0
    text = sources(a.logs)
    verbatim = {norm(m.group(0)) for m in NUM.finditer(text)}
    rounded = set()
    for v in verbatim:
        if "." in v and "e" not in v:
            dec = len(v.split(".")[1])
            for d in range(3, dec):
                rounded.add(f"{float(v):.{d}f}")
    missing = []
    toks = tokens((ROOT / "paper/main.tex").read_text())
    for n, t in toks:
        k = norm(t)
        if k in DERIVED or k in verbatim or k.lstrip("-") in verbatim or k in rounded or k.lstrip("-") in rounded:
            continue
        missing.append((n, t))
    for n, t in missing:
        print(f"main.tex:{n}: {t} not found in the sources")
    print(f"{len(toks) - len(missing)} of {len(toks)} numbers found")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
