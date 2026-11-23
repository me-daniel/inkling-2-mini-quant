"""Amendment 15, component A: the four fresh-domain KL texts, built once and hashed before any model scores them.

    python -I scripts/battery-data.py   (network; caches sources under data/raw/battery/, writes data/battery/A-*.txt
                                          and results/battery/data.md)

Per domain 8 documents, each cut to exactly 4,096 tokens of the model's own tokenizer (4 chunks of 1,024), joined in
order, so chunk i of a domain file belongs to document i // 4. Every choice below is fixed by rule, not by looking:
- python, rust: 8 permissively licensed repositories each, fixed here. The first four at their last commit before
  2023-01-01, the last four restricted to files added since 2025-01-01. Per repository the largest source files
  (tests, docs, examples, vendored and generated files excluded) are joined, largest first, until 4,096 tokens.
- german: Bundestag plenary protocols 21/101 to 21/108 (2026, official works under section 5 UrhG); per protocol
  the speech paragraphs (<rede> ... <p>) in order, from the first speech on.
- math: new arXiv math papers (ids 2611.*) in the OAI records of 2-12 Nov 2026 under CC BY 4.0 whose primary category is math.*, sorted by arXiv
  id, the first 8 whose LaTeX body (\\begin{document} to \\end{document} of the largest .tex file, comments
  dropped) holds at least 4,096 tokens.
Also measures, for the freshness rule, the share of lines in data/calib-mix.txt and data/calib-wt.txt that look like
code, LaTeX or German (simple per-line detectors, stated in the report).
"""

import hashlib
import io
import json
import re
import subprocess
import tarfile
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/battery"
OUT = ROOT / "data/battery"
REP = ROOT / "results/battery"
TOK = Tokenizer.from_file(str(ROOT / "models/Inkling-2-Mini/tokenizer.json"))
NTOK = 4096
UA = {"User-Agent": "research-battery/1.0"}

REPOS = {
    "python": [("psf/requests", "Apache-2.0"), ("pallets/flask", "BSD-3-Clause"), ("pallets/click", "BSD-3-Clause"),
               ("python-attrs/attrs", "MIT"), ("pydantic/pydantic", "MIT"), ("fastapi/fastapi", "MIT"),
               ("Textualize/textual", "MIT"), ("huggingface/huggingface_hub", "Apache-2.0")],
    "rust": [("BurntSushi/ripgrep", "MIT/Unlicense"), ("serde-rs/serde", "MIT/Apache-2.0"), ("tokio-rs/tokio", "MIT"),
             ("clap-rs/clap", "MIT/Apache-2.0"), ("astral-sh/uv", "MIT/Apache-2.0"), ("astral-sh/ruff", "MIT"),
             ("tokio-rs/axum", "MIT"), ("bevyengine/bevy", "MIT/Apache-2.0")],
}
EXT = {"python": ".py", "rust": ".rs"}
SKIP = re.compile(r"(^|/)(tests?|docs?|examples?|benches|vendor|third_party|fixtures|scripts)(/|$)|_test\.|test_|generated")


def get(url: str) -> bytes:
    # curl, not urllib: this Python's CA bundle lacks bundestag.de's intermediate certificate
    return subprocess.run(["curl", "-sfL", "--max-time", "300", "-A", UA["User-Agent"], url], check=True, capture_output=True).stdout


def cut(text: str) -> str | None:
    enc = TOK.encode(text, add_special_tokens=False)
    return TOK.decode(enc.ids[:NTOK]) if len(enc.ids) >= NTOK else None


def git(repo_dir: Path, *args: str) -> str:
    r = subprocess.run(["git", "-c", "core.commitGraph=false", "-C", str(repo_dir), *args], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if r.returncode:
        raise SystemExit(f"git {' '.join(args)} in {repo_dir.name}: {r.stderr.strip()[-400:]}")
    return r.stdout


def code_doc(domain: str, repo: str, recent: bool) -> tuple[str, dict]:
    d = RAW / "git" / repo.replace("/", "__")
    if not d.exists():
        subprocess.run(["git", "-c", "fetch.writeCommitGraph=false", "clone", "-q", "--filter=blob:none", "--no-checkout", f"https://github.com/{repo}.git", str(d)], check=True)
    head = git(d, "rev-parse", "HEAD").strip()
    if recent:
        rev = head
        added = set(git(d, "log", "--diff-filter=A", "--since=2025-01-01", "--name-only", "--format=", rev).split())
    else:
        rev = git(d, "rev-list", "-1", "--before=2023-01-01", head).strip()
        added = None
    # contents come from GitHub's tarball of rev (the blob:none clone holds commits and trees only; its lazy blob
    # fetch broke on 23 Nov), history and the added-files list from the clone
    tb = RAW / "tar" / f"{repo.replace('/', '__')}-{rev[:12]}.tar.gz"
    if not tb.exists():
        tb.parent.mkdir(parents=True, exist_ok=True)
        tb.write_bytes(get(f"https://codeload.github.com/{repo}/tar.gz/{rev}"))
    src = {}
    with tarfile.open(tb) as t:
        for m in t.getmembers():
            path = m.name.split("/", 1)[1] if "/" in m.name else ""
            if m.isfile() and path.endswith(EXT[domain]) and not SKIP.search(path) and (added is None or path in added):
                src[path] = t.extractfile(m).read().decode("utf-8", "replace")
    files = [(len(v.encode()), k) for k, v in src.items()]
    files.sort(key=lambda x: (-x[0], x[1]))
    parts, used = [], []
    for _, path in files:
        parts.append(f"// {path}\n" if domain == "rust" else f"# {path}\n")
        parts.append(src[path])
        used.append(path)
        text = cut("".join(parts))
        if text:
            return text, {"source": f"github.com/{repo}", "commit": rev[:12], "files": used}
    raise SystemExit(f"{repo}: fewer than {NTOK} tokens in eligible files")


def german_docs() -> list[tuple[str, dict]]:
    page = get("https://www.bundestag.de/ajax/filterlist/de/services/opendata/1058442-1058442?limit=40&noFilterSet=true").decode()
    urls = dict((int(m.group(2)), m.group(1)) for m in re.finditer(r'href="(https://www\.bundestag\.de/resource/blob/\d+/21(\d{3})\.xml)"', page))
    out = []
    for n in range(101, 109):
        f = RAW / "bundestag" / f"21{n:03d}.xml"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(get(urls[n]))
        root = ET.fromstring(f.read_bytes())
        paras = [" ".join("".join(p.itertext()).split()) for rede in root.iter("rede") for p in rede.iter("p")
                 if p.get("klasse") != "redner"]
        text = cut("\n\n".join(x for x in paras if x))
        if not text:
            raise SystemExit(f"protocol 21/{n}: too short")
        out.append((text, {"source": urls[n], "protocol": f"21/{n}", "date": root.get("sitzung-datum")}))
    return out


def latex_body(blob: bytes) -> str | None:
    texs = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            for m in t.getmembers():
                if m.isfile() and m.name.endswith(".tex"):
                    texs[m.name] = t.extractfile(m).read().decode("utf-8", "replace")
    except tarfile.ReadError:
        try:
            texs["main.tex"] = gzip.decompress(blob).decode("utf-8", "replace")
        except OSError:
            return None
    if not texs:
        return None
    src = max(texs.values(), key=len)
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", src, re.S)
    body = m.group(1) if m else src
    return "\n".join(line for line in (re.sub(r"(?<!\\)%.*", "", ln) for ln in body.splitlines()) if line.strip())


def math_docs() -> list[tuple[str, dict]]:
    f = RAW / "arxiv-oai-2026-11-02-12.xml"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(get("https://oaipmh.arxiv.org/oai?verb=ListRecords&metadataPrefix=arXiv&set=math:math&from=2026-11-02&until=2026-11-12"))
    ns = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
    cands = []
    for rec in ET.fromstring(f.read_bytes()).iter("{http://www.openarchives.org/OAI/2.0/}record"):
        meta = rec.find(".//a:arXiv", ns)
        if meta is None:
            continue
        lic = (meta.findtext("a:license", "", ns) or "").strip()
        cats = (meta.findtext("a:categories", "", ns) or "").split()
        if lic == "http://creativecommons.org/licenses/by/4.0/" and cats and cats[0].startswith("math.") \
                and (meta.findtext("a:id", "", ns) or "").startswith("2611."):
            cands.append((meta.findtext("a:id", "", ns), cats[0]))
    out = []
    for aid, cat in sorted(cands):
        p = RAW / "arxiv" / f"{aid.replace('/', '_')}.src"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(get(f"https://arxiv.org/e-print/{aid}"))
        body = latex_body(p.read_bytes())
        text = cut(body) if body else None
        if text:
            out.append((text, {"source": f"arxiv.org/abs/{aid}", "category": cat, "license": "CC BY 4.0"}))
        if len(out) == 8:
            return out
    raise SystemExit(f"only {len(out)} math documents")


DETECT = {
    "code": re.compile(r"^\s*(def |class |import |from \S+ import|fn |let |pub |use |#include|return\b|[{}];?$)|[;{}]\s*$"),
    "latex": re.compile(r"\\(begin|end|frac|mathbb|section|cite|label|ref)\b|\$[^$]+\$"),
    "german": re.compile(r"\b(und|der|die|das|nicht|ist|mit|sich|auch|wir|für)\b"),
}


def calib_shares() -> dict:
    res = {}
    for name in ("calib-mix.txt", "calib-wt.txt"):
        lines = [ln for ln in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines() if ln.strip()]
        res[name] = {k: sum(1 for ln in lines if (len(rx.findall(ln)) >= 3 if k == "german" else rx.search(ln))) / len(lines)
                     for k, rx in DETECT.items()}
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    docs = {d: [code_doc(d, r, i >= 4) + ({"license": lic},) for i, (r, lic) in enumerate(REPOS[d])] for d in REPOS}
    docs = {d: [(t, {**m, **x}) for t, m, x in v] for d, v in docs.items()}
    docs["german"] = german_docs()
    docs["math"] = math_docs()
    lines = ["# Amendment 15 component A: fresh-domain texts", "",
             "Built by `scripts/battery-data.py`; 8 documents x 4,096 tokens per domain (the model's tokenizer.json).", ""]
    manifest = {}
    for d, v in docs.items():
        text = "".join(t if t.endswith("\n") else t + "\n" for t, _ in v)
        p = OUT / f"A-{d}.txt"
        p.write_text(text, encoding="utf-8")
        n = len(TOK.encode(text, add_special_tokens=False).ids)
        h = hashlib.sha256(text.encode()).hexdigest()
        manifest[d] = [m for _, m in v]
        lines += [f"## {d}", "", f"`{p.relative_to(ROOT)}`: {n:,} tokens, sha256 {h}", ""]
        lines += [f"{i + 1}. {json.dumps(m, ensure_ascii=False)}" for i, (_, m) in enumerate(v)] + [""]
    sh = calib_shares()
    lines += ["## Freshness check (share of non-empty lines; code and LaTeX: one pattern match, German: 3+ stopwords)", "",
              "| file | code | latex | german |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['code']:.2%} | {v['latex']:.2%} | {v['german']:.2%} |" for k, v in sh.items()]
    (REP / "data.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REP / "A-manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
