"""Amendment 23: the development set DV, A3's rules with new documents, built once and hashed before any scoring.

    python -I scripts/a23-dev-data.py   (network; reuses scripts/a18-data.py and its caches; writes data/battery/DV-*.txt,
                                         results/a23/DV-manifest.json and results/a23/data.md)

- python, javascript: the 12 + 12 repositories below, used in no earlier set, calibration text or A3 (asserted).
  Licences as reported by the GitHub API on 28 Nov 2026 (repositories reporting NOASSERTION were left out).
  Same halves as A3: the first six at their last commit before 2023-01-01, the last six from files added since 2025.
- german, english: the next 12 qualifying articles after A3's in A3's cached recent-changes list (page-id order).
- math: the next 12 qualifying arXiv papers after A3's in A3's cached OAI record set.
A4, later, takes the documents after DV's under the same rules.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("a18", ROOT / "scripts/a18-data.py")
a18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a18)
bd, a16 = a18.bd, a18.a16
REP = ROOT / "results/a23"

DV = {
    "python": [("bottlepy/bottle", "MIT"), ("pyparsing/pyparsing", "MIT"), ("python-jsonschema/jsonschema", "MIT"),
               ("more-itertools/more-itertools", "MIT"), ("pallets/itsdangerous", "BSD-3-Clause"), ("pallets/markupsafe", "BSD-3-Clause"),
               ("Kludex/uvicorn", "BSD-3-Clause"), ("agronholm/anyio", "MIT"), ("fastapi/typer", "MIT"),
               ("Textualize/rich", "MIT"), ("urllib3/urllib3", "MIT"), ("pypa/pip-tools", "BSD-3-Clause")],
    "javascript": [("jquery/jquery", "MIT"), ("jashkenas/underscore", "MIT"), ("jashkenas/backbone", "MIT"),
                   ("caolan/async", "MIT"), ("request/request", "Apache-2.0"), ("handlebars-lang/handlebars.js", "MIT"),
                   ("fastify/fastify", "MIT"), ("mochajs/mocha", "MIT"), ("highlightjs/highlight.js", "BSD-3-Clause"),
                   ("Leaflet/Leaflet", "BSD-2-Clause"), ("koajs/koa", "MIT"), ("yargs/yargs", "MIT")],
}
FALLBACK = {"python": [("PyCQA/isort", "MIT"), ("pallets/quart", "MIT"), ("python-hyper/h11", "MIT")],
            "javascript": [("chalk/chalk", "MIT"), ("pinojs/pino", "MIT"), ("expressjs/morgan", "MIT")]}


def after_a3(docs: list, a3: list) -> list:
    """The documents of a 24-long qualifying list that are not A3's, first 12 (A3 took the first 12 of the same list)."""
    seen = {m["source"] for m in a3}
    out = [(t, m) for t, m in docs if m["source"] not in seen]
    if len(out) < a18.NDOC:
        raise SystemExit(f"only {len(out)} documents after A3's")
    return out[:a18.NDOC]


def main() -> None:
    a3m = json.loads((ROOT / "results/a18/A3-manifest.json").read_text())
    earlier = {r for v in bd.REPOS.values() for r, _ in v} | {r for v in (a16.CALIB, a16.A2) for x in v.values() for r, _ in x}
    earlier |= {r for v in a18.A3.values() for r, _ in v} | {r for v in a18.FALLBACK.values() for r, _ in v}
    names = [r for v in DV.values() for r, _ in v] + [r for v in FALLBACK.values() for r, _ in v]
    assert len(set(names)) == len(names) and not earlier & set(names), "a repository is reused"
    a18.A3, a18.FALLBACK = DV, FALLBACK
    docs, notes = {}, []
    for d in DV:
        docs[d], n = a18.code_docs(d)
        notes += n
    a18.NDOC = 24
    docs["german"] = after_a3(a18.wiki_docs("de"), a3m["german"])
    docs["english"] = after_a3(a18.wiki_docs("en"), a3m["english"])
    docs["math"] = after_a3(a18.math_docs(), a3m["math"])
    a18.NDOC = 12
    calib, a3w = set(), set()
    for name in ("calib-mix.txt", "calib-mixc.txt", "calib-code.txt"):
        calib |= a16.windows((ROOT / "data" / name).read_text(encoding="utf-8"))
    REP.mkdir(parents=True, exist_ok=True)
    lines = ["# Amendment 23 data: development set DV", "",
             f"Built by `scripts/a23-dev-data.py`; 12 documents x 4,096 tokens per domain (the model's tokenizer.json).", ""]
    if notes:
        lines += ["Replacements by rule:", ""] + [f"- {n}" for n in notes] + [""]
    manifest, overlap = {}, {}
    for d, v in docs.items():
        assert not {m["source"] for _, m in v} & {m["source"] for m in a3m[d]}, f"{d}: an A3 document is reused"
        text, n, h = a16.write(bd.OUT / f"DV-{d}.txt", v)
        w = a16.windows(text)
        a3text = (bd.OUT / f"A3-{d}.txt").read_text(encoding="utf-8")
        overlap[d] = (len(w & calib) / len(w), len(w & a16.windows(a3text)) / len(w))
        manifest[d] = [m for _, m in v]
        lines += [f"## {d}", "", f"`data/battery/DV-{d}.txt`: {n:,} tokens, sha256 {h}", ""]
        lines += [f"{i + 1}. {json.dumps(m, ensure_ascii=False)}" for i, (_, m) in enumerate(v)] + [""]
    lines += ["## Overlap (share of 13-token windows also in the calibration texts / in A3)", "",
              "| domain | calibration | A3 |", "|---|---|---|"] + [f"| {d} | {c:.3%} | {a:.3%} |" for d, (c, a) in overlap.items()]
    (REP / "data.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REP / "DV-manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
