"""Amendment 18 data: the fresh set A3, built once and hashed before any model is scored on it.

    python -I scripts/a18-data.py   (network; caches under data/raw/battery/ like scripts/battery-data.py, whose
                                      functions it reuses; writes data/battery/A3-*.txt, results/a18/A3-manifest.json
                                      and results/a18/data.md)

Per domain 12 documents x 4,096 tokens of the model's tokenizer.json, joined in order (chunk i of a domain file at
-c 1024 belongs to document i // 4). Every choice is fixed by rule here, before any model is scored on A3:
- python, javascript: 12 repositories each, permissively licensed, used in no earlier set or calibration text. The
  first six at their last commit before 2023-01-01, the last six restricted to files added since 2025-01-01
  (battery A's rule). JavaScript (.js) is in no calibration text. If a repository has fewer than 4,096 eligible
  tokens, the next one from its domain's FALLBACK list (in order, same half) replaces it, and the report says so.
- german, english: Wikipedia articles created in the 30 days before the build (recent-changes "new", main
  namespace, no redirects, no bots), at least 25,000 bytes at creation, sorted by page id; the plain-text extract
  (TextExtracts) of the current revision; the first 12 that hold 4,096 tokens. Licence CC BY-SA 4.0.
- math: arXiv papers in the OAI records of 13-26 Nov 2026, CC BY 4.0, primary category math.*, ids 2611.* not used in
  battery A, sorted by id; the first 12 whose LaTeX body holds 4,096 tokens (battery A's rule).
Overlap: the share of each domain's 13-token windows that also occur in data/calib-mix.txt, data/calib-mixc.txt or
data/calib-code.txt (reported, no threshold).
"""

import hashlib
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bd", ROOT / "scripts/battery-data.py")
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)
spec2 = importlib.util.spec_from_file_location("a16", ROOT / "scripts/a16-data.py")
a16 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(a16)
bd.EXT["javascript"] = ".js"
JS_SKIP = re.compile(r"(^|/)(dist|build|node_modules|lib/vendor|packages/.*/test)(/|$)|\.min\.js$|\.bundle\.js$")
NDOC = 12
REP = ROOT / "results/a18"

A3 = {
    "python": [("httpie/cli", "BSD-3-Clause"), ("tornadoweb/tornado", "Apache-2.0"), ("pallets/werkzeug", "BSD-3-Clause"),
               ("sphinx-doc/sphinx", "BSD-2-Clause"), ("scrapy/scrapy", "BSD-3-Clause"), ("celery/celery", "BSD-3-Clause"),
               ("streamlit/streamlit", "Apache-2.0"), ("sympy/sympy", "BSD-3-Clause"), ("networkx/networkx", "BSD-3-Clause"),
               ("mitmproxy/mitmproxy", "MIT"), ("pyca/cryptography", "Apache-2.0/BSD-3-Clause"), ("boto/boto3", "Apache-2.0")],
    "javascript": [("expressjs/express", "MIT"), ("lodash/lodash", "MIT"), ("axios/axios", "MIT"), ("chartjs/Chart.js", "MIT"),
                   ("mrdoob/three.js", "MIT"), ("moment/moment", "MIT"),
                   ("nodejs/node", "MIT"), ("eslint/eslint", "MIT"), ("facebook/react", "MIT"), ("mozilla/pdf.js", "Apache-2.0"),
                   ("prettier/prettier", "MIT"), ("webpack/webpack", "MIT")],
}
FALLBACK = {"python": [("pytorch/vision", "BSD-3-Clause"), ("Kludex/starlette", "BSD-3-Clause"), ("jupyter/notebook", "BSD-3-Clause")],
            "javascript": [("nodejs/undici", "MIT"), ("sveltejs/svelte", "MIT"), ("adobe/react-spectrum", "Apache-2.0")]}


def wiki_docs(lang: str) -> list[tuple[str, dict]]:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    cache = bd.RAW / "wiki" / lang
    cache.mkdir(parents=True, exist_ok=True)
    rc = cache / "recentchanges.json"
    if not rc.exists():
        items, cont = [], ""
        while True:
            q = (f"{api}?action=query&list=recentchanges&rctype=new&rcnamespace=0&rcshow=!redirect|!bot&rclimit=500"
                 f"&rcprop=title|ids|sizes|timestamp&format=json{cont}")
            j = json.loads(bd.get(q))
            items += j["query"]["recentchanges"]
            if "continue" not in j:
                break
            cont = f"&rccontinue={j['continue']['rccontinue']}"
        rc.write_text(json.dumps(items), encoding="utf-8")
    items = sorted({i["pageid"]: i for i in json.loads(rc.read_text()) if i.get("newlen", 0) >= 25000}.values(),
                   key=lambda i: i["pageid"])
    out = []
    for it in items:
        p = cache / f"{it['pageid']}.json"
        if not p.exists():
            p.write_bytes(bd.get(f"{api}?action=query&prop=extracts|revisions&explaintext=1&exsectionformat=plain"
                                 f"&rvprop=ids|timestamp&pageids={it['pageid']}&format=json"))
        page = json.loads(p.read_bytes())["query"]["pages"].get(str(it["pageid"]), {})
        text = bd.cut(page.get("extract") or "")
        if text:
            rev = (page.get("revisions") or [{}])[0]
            out.append((text, {"source": f"{lang}.wikipedia.org/?curid={it['pageid']}", "title": it["title"],
                               "created": it["timestamp"], "revid": rev.get("revid"), "license": "CC BY-SA 4.0"}))
        if len(out) == NDOC:
            return out
    raise SystemExit(f"{lang}: only {len(out)} articles")


def math_docs() -> list[tuple[str, dict]]:
    used = {m["source"].rsplit("/", 1)[1] for m in json.loads((bd.REP / "A-manifest.json").read_text())["math"]}
    f = bd.RAW / "arxiv-oai-2026-11-13-26.xml"
    if not f.exists():
        f.write_bytes(bd.get("https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXiv&set=math:math&from=2026-11-13&until=2026-11-26"))
    ns = {"a": "http://arxiv.org/OAI/arXiv/"}
    cands = []
    for rec in ET.fromstring(f.read_bytes()).iter("{http://www.openarchives.org/OAI/2.0/}record"):
        meta = rec.find(".//a:arXiv", ns)
        if meta is None:
            continue
        aid = meta.findtext("a:id", "", ns) or ""
        cats = (meta.findtext("a:categories", "", ns) or "").split()
        if (meta.findtext("a:license", "", ns) or "").strip() == "http://creativecommons.org/licenses/by/4.0/" \
                and cats and cats[0].startswith("math.") and aid.startswith("2611.") and aid not in used:
            cands.append((aid, cats[0]))
    out = []
    for aid, cat in sorted(cands):
        p = bd.RAW / "arxiv" / f"{aid}.src"
        if not p.exists():
            p.write_bytes(bd.get(f"https://arxiv.org/e-print/{aid}"))
        body = bd.latex_body(p.read_bytes())
        text = bd.cut(body) if body else None
        if text:
            out.append((text, {"source": f"arxiv.org/abs/{aid}", "category": cat, "license": "CC BY 4.0"}))
        if len(out) == NDOC:
            return out
    raise SystemExit(f"math: only {len(out)} documents")


def code_docs(domain: str) -> tuple[list[tuple[str, dict]], list[str]]:
    if domain == "javascript":
        base = bd.SKIP
        bd.SKIP = re.compile(base.pattern + "|" + JS_SKIP.pattern)
    out, notes, fb = [], [], list(FALLBACK[domain])
    for i, (repo, lic) in enumerate(A3[domain]):
        recent = i >= NDOC // 2
        while True:
            try:
                t, m = bd.code_doc(domain, repo, recent)
                out.append((t, {**m, "license": lic}))
                break
            except SystemExit as e:
                if not fb:
                    raise
                nxt = fb.pop(0)
                notes.append(f"{domain}: {repo} replaced by {nxt[0]} ({e})")
                repo, lic = nxt
    if domain == "javascript":
        bd.SKIP = base
    return out, notes


def main() -> None:
    earlier = {r for v in bd.REPOS.values() for r, _ in v} | {r for v in (a16.CALIB, a16.A2) for x in v.values() for r, _ in x}
    names = [r for v in A3.values() for r, _ in v] + [r for v in FALLBACK.values() for r, _ in v]
    assert len(set(names)) == len(names) and not earlier & set(names), "a repository is reused"
    REP.mkdir(parents=True, exist_ok=True)
    docs, notes = {}, []
    for d in A3:
        docs[d], n = code_docs(d)
        notes += n
    docs["german"] = wiki_docs("de")
    docs["english"] = wiki_docs("en")
    docs["math"] = math_docs()
    seen = set()
    for name in ("calib-mix.txt", "calib-mixc.txt", "calib-code.txt"):
        seen |= a16.windows((ROOT / "data" / name).read_text(encoding="utf-8"))
    lines = ["# Amendment 18 data: fresh set A3", "",
             f"Built by `scripts/a18-data.py`; {NDOC} documents x 4,096 tokens per domain (the model's tokenizer.json).", ""]
    if notes:
        lines += ["Replacements by rule:", ""] + [f"- {n}" for n in notes] + [""]
    manifest, overlap = {}, {}
    for d, v in docs.items():
        text, n, h = a16.write(bd.OUT / f"A3-{d}.txt", v)
        w = a16.windows(text)
        overlap[d] = len(w & seen) / len(w)
        manifest[d] = [m for _, m in v]
        lines += [f"## {d}", "", f"`data/battery/A3-{d}.txt`: {n:,} tokens, sha256 {h}", ""]
        lines += [f"{i + 1}. {json.dumps(m, ensure_ascii=False)}" for i, (_, m) in enumerate(v)] + [""]
    lines += ["## Overlap with the calibration texts (13-token windows also in calib-mix, calib-mixc or calib-code)", "",
              "| domain | share |", "|---|---|"] + [f"| {d} | {x:.3%} |" for d, x in overlap.items()]
    (REP / "data.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REP / "A3-manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
