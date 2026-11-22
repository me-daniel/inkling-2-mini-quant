"""Render paper/main.tex as GitHub Markdown (paper/paper.md), so the PDF and the Markdown come from one source.

    .venv/bin/python -I paper/tex2md.py      # needs paper/main.bbl from the LaTeX build for citation numbers

Handles only the LaTeX this paper uses: sections, paragraphs, lists, tables (booktabs), figures, inline and display
math, \\cite, \\ref, \\code. The TikZ figure becomes figures/loop-diagram.svg, the repository's drawing of the same
design. Anything it does not know stops the conversion instead of leaking into the Markdown.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def arg(s: str, i: int) -> tuple[str, int]:
    """The brace group starting at s[i] == '{' and the index after it."""
    assert s[i] == "{", s[i:i + 40]
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0) if s[j - 1] != "\\" else 0
        if depth == 0:
            return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def bib() -> tuple[dict[str, tuple[int, str]], list[str]]:
    """Citation numbers and author labels from the .bbl, and the reference list."""
    bbl = (HERE / "main.bbl").read_text()
    bbl = re.sub(r"\\natexlab\{(\w*)\}", r"\1", bbl)
    keys, refs = {}, []
    for n, it in enumerate(re.split(r"\\bibitem", bbl)[1:], 1):
        depth, k = 0, 0
        for k, c in enumerate(it):  # the [label] may hold braces with ] inside, so scan by depth
            depth += {"{": 1, "}": -1}.get(c, 0)
            if c == "]" and depth == 0:
                break
        label = it[1:k]
        key, k = arg(it, k + 1)
        keys[key] = (n, re.sub(r"[{}]", "", label.split("(")[0]).strip())
        body = it[k:].split("\\end{thebibliography}")[0]
        body = re.sub(r"\\newblock\s*", "", body)
        body = re.sub(r"\\url\{([0-9a-f]{40})\}", r"`\1`", body)
        body = re.sub(r"\\url\{(.*?)\}", r"<\1>", body)
        body = re.sub(r"\\emph\{(.*?)\}", r"*\1*", body)
        body = body.replace("\\_", "_").replace("{\\'a}", "á").replace("~", " ")
        body = re.sub(r"\s+", " ", re.sub(r"[{}]", "", body)).strip()
        if "\\" in body:
            raise ValueError(f"unknown command in reference {key}: {body[:120]}")
        refs.append(f"{n}. {body}")
    return keys, refs


def numbering(tex: str) -> dict[str, str]:
    """Numbers for \\label targets, in document order, as LaTeX would give them."""
    out, sec, sub, fig, tab = {}, 0, 0, 0, 0
    env = None
    for m in re.finditer(r"\\(section\*?|subsection|begin\{(figure|table)\}|label)\{?([^}]*)\}?", tex):
        cmd = m.group(1)
        if cmd == "section":
            sec, sub, last = sec + 1, 0, str(sec + 1)
        elif cmd == "subsection":
            sub += 1
            last = f"{sec}.{sub}"
        elif m.group(2) == "figure":
            fig += 1
            env = str(fig)
        elif m.group(2) == "table":
            tab += 1
            env = str(tab)
        elif cmd == "label":
            key = m.group(3)
            out[key] = env if key.startswith(("fig:", "tab:")) else last
    return out


def inline(s: str, cites: dict, refs: dict) -> str:
    """Inline LaTeX to Markdown; math spans are kept as $...$ with their LaTeX."""
    parts = re.split(r"(\$[^$]+\$)", s)
    out = []
    for p in parts:
        if p.startswith("$"):
            out.append(p.replace("\\,", " "))
            continue
        p = re.sub(r"~?\\citep\{([^}]*)\}", lambda m: " [" + ", ".join(str(cites[k][0]) for k in m.group(1).split(",")) + "]", p)
        p = re.sub(r"\\citet\{([^}]*)\}", lambda m: ", ".join(f"{cites[k][1]} [{cites[k][0]}]" for k in m.group(1).split(",")), p)
        p = re.sub(r"\\ref\{([^}]*)\}", lambda m: refs[m.group(1)], p)
        p = re.sub(r"\\label\{[^}]*\}", "", p)
        p = re.sub(r"\\(?:code|texttt|nolinkurl)\{((?:[^{}]|\{[^{}]*\})*)\}", lambda m: "`" + m.group(1).replace("\\_", "_").replace("\\%", "%") + "`", p)
        p = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", p)
        p = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", p)
        p = re.sub(r"\\shortstack\{(.*?)\}", lambda m: m.group(1).replace("\\\\", " "), p)
        p = p.replace("``", '"').replace("''", '"').replace("\\_", "_").replace("\\%", "%").replace("\\&", "&")
        p = p.replace("\\,", " ").replace("~", " ").replace("\\ ", " ")
        if re.search(r"\\[a-zA-Z]", p.replace("`", "")):
            raise ValueError(f"unknown command in: {p.strip()[:120]}")
        out.append(p)
    return "".join(out)


def table(env: str, num: str, cites: dict, refs: dict) -> str:
    cap = inline(arg(env, env.index("\\caption") + 8)[0], cites, refs)
    i = env.index("\\begin{tabular}")
    _, j = arg(env, i + len("\\begin{tabular}"))  # skip the column spec
    body = env[j:env.index("\\end{tabular}")]
    body = re.sub(r"\\shortstack\{(.*?)\}", lambda m: m.group(1).replace("\\\\", " "), body)  # before rows split on \\
    head, _, rest = body.partition("\\midrule")
    rows = lambda t: [r.strip() for r in re.sub(r"\\(top|bottom)rule", "", t).split("\\\\") if r.strip()]

    def cells(r: str) -> list[str]:
        out = []
        for c in re.split(r"(?<!\\)&", r):
            c = c.strip()
            m = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", c)
            out += [m.group(2)] * int(m.group(1)) if m else [c]
        return [inline(c, cites, refs).strip() for c in out]

    hs = [cells(r) for r in rows(head)]
    header = [" ".join(dict.fromkeys(h[k] for h in hs if h[k])) for k in range(len(hs[0]))]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cells(r)) + " |" for r in rows(rest)]
    return f"*Table {num}: {cap}*\n\n" + "\n".join(lines)


def figure(env: str, num: str, cites: dict, refs: dict) -> str:
    cap = inline(arg(env, env.index("\\caption") + 8)[0], cites, refs)
    m = re.search(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", env)
    src = m.group(1).replace(".pdf", ".png") if m else "../figures/loop-diagram.svg"
    return f"![Figure {num}]({src})\n\n*Figure {num}: {cap}*"


def main() -> None:
    tex = (HERE / "main.tex").read_text()
    cites, reflist = bib()
    refs = numbering(tex)
    title = inline(re.search(r"\\title\{(.*?)\}\n", tex).group(1), cites, refs)
    author = re.search(r"\\author\{(.*?)\}\n", tex).group(1).split("\\\\")[0]
    date = re.search(r"\\date\{(.*?)\}", tex).group(1)
    doc = tex[tex.index("\\maketitle") + 10:tex.index("\\bibliographystyle")]
    doc = re.sub(r"(?m)^\s*%.*\n", "", doc)
    out = [f"# {title}", "", f"{author}, {date}. Generated from `paper/main.tex` by `paper/tex2md.py`; the PDF is `paper/main.pdf`.", ""]
    sec = sub = 0
    pos = 0
    pat = re.compile(r"\\begin\{(abstract|itemize|table|figure|equation)\}|\\(section\*?|subsection|paragraph)\{")
    text_buf = []

    def flush() -> None:
        for para in re.split(r"\n\s*\n", "".join(text_buf)):
            para = " ".join(l.strip() for l in para.strip().splitlines())
            if para:
                out.extend([inline(para, cites, refs), ""])
        text_buf.clear()

    while True:
        m = pat.search(doc, pos)
        if not m:
            text_buf.append(doc[pos:])
            flush()
            break
        text_buf.append(doc[pos:m.start()])
        flush()
        if m.group(1):
            name = m.group(1)
            end = doc.index(f"\\end{{{name}}}", m.end())
            env = doc[m.end():end]
            pos = end + len(f"\\end{{{name}}}")
            lab = re.search(r"\\label\{([^}]*)\}", env)
            env = re.sub(r"\\label\{[^}]*\}", "", env)
            if name == "abstract":
                out.extend(["## Abstract", "", inline(" ".join(env.split()), cites, refs), ""])
            elif name == "itemize":
                for item in env.split("\\item")[1:]:
                    out.append("- " + inline(" ".join(item.split()), cites, refs))
                out.append("")
            elif name == "equation":
                out.extend(["$$", " ".join(env.split()).replace("\\qquad", "\\qquad"), "$$", ""])
            elif name == "table":
                out.extend([table(env, refs[lab.group(1)], cites, refs), ""])
            else:
                out.extend([figure(env, refs[lab.group(1)], cites, refs), ""])
        else:
            cmd = m.group(2)
            head, pos = arg(doc, m.end() - 1)
            if cmd == "section":
                sec, sub = sec + 1, 0
                out.extend([f"## {sec} {inline(head, cites, refs)}", ""])
            elif cmd == "section*":
                out.extend([f"## {inline(head, cites, refs)}", ""])
            elif cmd == "subsection":
                sub += 1
                out.extend([f"### {sec}.{sub} {inline(head, cites, refs)}", ""])
            else:
                text_buf.append(f"**{head}** ")
            pos = doc.index("\n", pos) + 1 if doc[pos:pos + 1] == "\n" and cmd != "paragraph" else pos
    out.extend(["## References", ""] + reflist + [""])
    md = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
    (HERE / "paper.md").write_text(md)


if __name__ == "__main__":
    sys.exit(main())
