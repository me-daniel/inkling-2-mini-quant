"""Amendment 16 data, built once and hashed before v2c exists: the code calibration text and the fresh test set A2.

    python -I scripts/a16-data.py   (network; caches under data/raw/battery/ like scripts/battery-data.py, whose
                                      functions it reuses; writes data/calib-code.txt, data/battery/A2-*.txt and
                                      results/battery/a16-data.md)

- Calibration code (data/calib-code.txt): 8 Python and 8 Rust repositories, permissive licences, none used in
  battery A or A2. Per repository the largest source files at the last commit before 2023-01-01 (tests, docs,
  examples, vendored and generated files excluded), cut to 4,096 tokens, as battery A builds its documents.
- A2, fresh: python and rust (8 new repositories each, first four before 2023, last four files added since
  2025-01-01, battery A's rule), go and typescript (same rule; no Go or TypeScript in any calibration text).
  8 documents x 4,096 tokens each. No new German text: the Bundestag listing page answered HTTP 400 on 23 Nov, so the
  German guard reuses battery A's data/battery/A-german.txt (scripts/a16-score.sh).
- Overlap: the share of A2's 13-token windows that also occur in data/calib-mix.txt or data/calib-code.txt,
  per domain (reported, no threshold; licence headers can repeat).
"""

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bd", ROOT / "scripts/battery-data.py")
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)
bd.EXT.update({"go": ".go", "typescript": ".ts"})

CALIB = {
    "python": [("django/django", "BSD-3-Clause"), ("numpy/numpy", "BSD-3-Clause"), ("scikit-learn/scikit-learn", "BSD-3-Clause"),
               ("psf/black", "MIT"), ("pytest-dev/pytest", "MIT"), ("sqlalchemy/sqlalchemy", "MIT"),
               ("encode/httpx", "BSD-3-Clause"), ("pallets/jinja", "BSD-3-Clause")],
    "rust": [("rust-lang/cargo", "MIT/Apache-2.0"), ("rayon-rs/rayon", "MIT/Apache-2.0"), ("hyperium/hyper", "MIT"),
             ("sharkdp/fd", "MIT/Apache-2.0"), ("alacritty/alacritty", "Apache-2.0"), ("nushell/nushell", "MIT"),
             ("tauri-apps/tauri", "MIT/Apache-2.0"), ("rust-lang/rust-analyzer", "MIT/Apache-2.0")],
}
A2 = {
    "python": [("tqdm/tqdm", "MIT/MPL-2.0"), ("Delgan/loguru", "MIT"), ("marshmallow-code/marshmallow", "MIT"),
               ("python-trio/trio", "MIT/Apache-2.0"), ("pypa/pip", "MIT"), ("python-poetry/poetry", "MIT"),
               ("aio-libs/aiohttp", "Apache-2.0"), ("pandas-dev/pandas", "BSD-3-Clause")],
    "rust": [("rust-random/rand", "MIT/Apache-2.0"), ("dtolnay/anyhow", "MIT/Apache-2.0"), ("crossbeam-rs/crossbeam", "MIT/Apache-2.0"),
             ("image-rs/image", "MIT/Apache-2.0"), ("denoland/deno", "MIT"), ("biomejs/biome", "MIT/Apache-2.0"),
             ("gfx-rs/wgpu", "MIT/Apache-2.0"), ("rust-lang/rustfmt", "MIT/Apache-2.0")],
    "go": [("spf13/cobra", "Apache-2.0"), ("gin-gonic/gin", "MIT"), ("sirupsen/logrus", "MIT"), ("gorilla/mux", "BSD-3-Clause"),
           ("prometheus/prometheus", "Apache-2.0"), ("caddyserver/caddy", "Apache-2.0"), ("junegunn/fzf", "MIT"),
           ("cli/cli", "MIT")],
    "typescript": [("date-fns/date-fns", "MIT"), ("colinhacks/zod", "MIT"), ("trpc/trpc", "MIT"), ("vitejs/vite", "MIT"),
                   ("excalidraw/excalidraw", "MIT"), ("withastro/astro", "MIT"), ("honojs/hono", "MIT"),
                   ("drizzle-team/drizzle-orm", "Apache-2.0")],
}


def windows(text: str, n: int = 13) -> set[tuple[int, ...]]:
    ids = bd.TOK.encode(text, add_special_tokens=False).ids
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


def write(path: Path, docs: list[tuple[str, dict]]) -> tuple[str, int, str]:
    text = "".join(t if t.endswith("\n") else t + "\n" for t, _ in docs)
    path.write_text(text, encoding="utf-8")
    return text, len(bd.TOK.encode(text, add_special_tokens=False).ids), hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    used = {r for v in bd.REPOS.values() for r, _ in v}
    for v in (CALIB, A2):
        names = [r for x in v.values() for r, _ in x]
        assert len(set(names)) == len(names) and not used & set(names), "a repository is reused"
    assert not {r for x in CALIB.values() for r, _ in x} & {r for x in A2.values() for r, _ in x}
    lines = ["# Amendment 16 data: code calibration and fresh set A2", "",
             "Built by `scripts/a16-data.py`; every document is 4,096 tokens of the model's tokenizer.json.", ""]
    manifest = {}
    cal = [(t, {**m, "license": lic}) for d, v in CALIB.items() for (r, lic) in v for t, m in [bd.code_doc(d, r, False)]]
    ctext, n, h = write(ROOT / "data/calib-code.txt", cal)
    manifest["calib-code"] = [m for _, m in cal]
    lines += ["## calib-code", "", f"`data/calib-code.txt`: {n:,} tokens, sha256 {h}", ""]
    lines += [f"{i + 1}. {json.dumps(m, ensure_ascii=False)}" for i, (_, m) in enumerate(cal)] + [""]
    seen = windows(ctext) | windows((ROOT / "data/calib-mix.txt").read_text(encoding="utf-8"))
    docs = {d: [(t, {**m, "license": lic}) for i, (r, lic) in enumerate(v) for t, m in [bd.code_doc(d, r, i >= 4)]]
            for d, v in A2.items()}
    overlap = {}
    for d, v in docs.items():
        text, n, h = write(bd.OUT / f"A2-{d}.txt", v)
        w = windows(text)
        overlap[d] = len(w & seen) / len(w)
        manifest[d] = [m for _, m in v]
        lines += [f"## {d}", "", f"`data/battery/A2-{d}.txt`: {n:,} tokens, sha256 {h}", ""]
        lines += [f"{i + 1}. {json.dumps(m, ensure_ascii=False)}" for i, (_, m) in enumerate(v)] + [""]
    lines += ["## Overlap with the calibration texts (share of 13-token windows also in calib-mix or calib-code)", "",
              "| domain | share |", "|---|---|"] + [f"| {d} | {x:.3%} |" for d, x in overlap.items()]
    (bd.REP / "a16-data.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (bd.REP / "A2-manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
