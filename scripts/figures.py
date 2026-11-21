"""Render every figure of the pilot into figures/ (PNG at 200 dpi plus SVG).

    .venv/bin/python -I scripts/figures.py [--src results/pilot] [--out figures]

Every plotted value is read from a file in results/pilot/. Nothing is typed in by hand. The one visual style is
analysis/style.py. Colour roles: C["ours"] is our file and the looped layers and passes, C["ref"] is Unsloth
UD-IQ3_XXS and every control, C["other"] is the other comparators and the non-looped layers. Shape, hatching and
direct labels repeat what colour says, so no figure depends on colour alone.

Intervals: where a results file holds per-chunk values, the bar is a 95% percentile bootstrap over chunks
(10,000 resamples, seed 0, the same settings as analysis/compare.py). Speed bars are the spread llama-bench prints.
Residual statistics carry the intervals stored in the JSON files.

A figure whose source files are missing is skipped without an error (depth-speed needs depth-*.txt).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from analysis.style import C, apply  # noqa: E402

SIZE = (6.4, 3.6)
SMALL = 7.5
# Display names. Uploader names are the public Hugging Face accounts. A key not listed here is shown as is.
LABEL = {"cand": "Ours", "UD-IQ3_XXS": "Unsloth UD-IQ3_XXS", "uniform-IQ3_XXS": "Stock IQ3_XXS",
         "bart-IQ2_S": "bartowski IQ2_S", "bart-IQ2_M": "bartowski IQ2_M", "atomic-AD-IQ2_S": "AtomicChat AD-IQ2_S",
         "apex-I-Nano": "mudler APEX-I-Nano", "apex-I-NanoPlus": "IsValorum APEX-I-NanoPlus-V2.1",
         "prithiv-Q2_K": "prithivMLmods Q2_K"}
OURS, REF = "cand", "UD-IQ3_XXS"
LEVELS = [("2bit", "2-bit\ngate_up and down"), ("gu2xxs", "IQ2_XXS gate_up\nIQ4_NL down"),
          ("dnq2o", "IQ3_XXS gate_up\nweighted Q2_0 down"), ("gu2s", "IQ2_S gate_up\nIQ4_NL down"),
          ("3bit", "3-bit\nIQ3_XXS gate_up")]


# Source files of each figure, for figures/README.md.
SOURCES = {"mac-comparison": "kl.tsv, bench-*.txt", "speed-vs-kl": "kl.tsv, bench-*.txt", "per-pass": "perpass-kl.tsv",
           "pass-vs-layer-split": "perpass-residual.json, perpass3bit-residual.json, layersplit-residual.json",
           "noise": "perpass-kl.tsv, noise-ratio.txt", "block-map": "kl.tsv, phase2.md", "pass-sweep": "kl.tsv",
           "depth-speed": "depth-*.txt, memory-*.txt"}
DONE: list[tuple[str, str]] = []


# ---------- reading ----------

def read_kl(src: Path) -> dict[str, dict]:
    rows = csv.DictReader(open(src / "kl.tsv"), delimiter="\t")
    return {r["run"]: {"kl": float(r["mean_kl"]), "se": float(r["se"]), "top": float(r["same_top_p"]),
                       "chunks": np.array([float(x) for x in r["per_chunk"].split(",")])} for r in rows}


def read_perpass(src: Path) -> dict[str, dict]:
    """Conditions scored against the IQ4_NL base (Addendum B). Deleted twins are left out."""
    out = {}
    for r in csv.DictReader(open(src / "perpass-kl.tsv"), delimiter="\t"):
        if r["against"] == "pp-base" and r["mean_kl"] != "deleted":
            out[r["cond"]] = {"kl": float(r["mean_kl"]), "chunks": np.array([float(x) for x in r["per_chunk"].split(",")])}
    return out


def read_bench(path: Path) -> dict:
    """One llama-bench markdown table plus the Metal line written by the gate script."""
    out: dict = {"tests": {}, "off": None, "gib": None, "build": None}
    cols: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if "t/s" in cells:
                cols = cells
            elif cols and not set(cells[0]) <= set("-: "):
                row = dict(zip(cols, cells))
                m = re.match(r"(pp|tg)(\d+)(?:\s*@\s*d(\d+))?", row["test"])
                depth = int(m.group(3) or row.get("n_depth", 0) or 0)
                mean, _, sd = row["t/s"].partition("±")
                out["tests"][(m.group(1) + m.group(2), depth)] = (float(mean), float(sd or 0))
                out["gib"] = float(row["size"].split()[0])
        elif line.startswith("build:"):
            out["build"] = line.split()[1]
        elif "Metal cannot multiply" in line:
            out["off"] = int(line.rsplit(":", 1)[1])
    return out


def read_files(src: Path) -> dict[str, dict]:
    """Every file with a T1 score: KL on T1 and T2, speed, size as llama-bench reports it, tensors off Metal."""
    kl = read_kl(src)
    files = {}
    for run in kl:
        m = re.fullmatch(r"final-(.+)-T1-P3", run)
        if not m:
            continue
        k = m.group(1)
        f = {"T1": kl[run], "T2": kl.get(f"final-{k}-T2-P3"), "bench": None}
        if (src / f"bench-{k}.txt").exists():
            f["bench"] = read_bench(src / f"bench-{k}.txt")
        # file size in GB (10^9 bytes), from the file itself, as on the model card
        for p in (ROOT / f"models/{k}.gguf", ROOT / f"models/ext/{k}.gguf"):
            if p.exists():
                f["gb"] = p.stat().st_size / 1e9
                break
        else:
            f["gb"] = None
        files[k] = f
    limit = files[REF]["bench"]["gib"] if REF in files and files[REF]["bench"] else None
    for f in files.values():
        b = f["bench"]
        f["over"] = bool(b and limit is not None and b["gib"] > limit)
        f["off"] = b["off"] if b else None
    return files


def boot(chunks: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    idx = np.random.default_rng(seed).integers(0, chunks.size, size=(n, chunks.size))
    b = chunks[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_ratio(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    idx = np.random.default_rng(seed).integers(0, a.size, size=(n, a.size))
    r = a[idx].mean(axis=1) / b[idx].mean(axis=1)
    return float(a.mean() / b.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


# ---------- drawing helpers ----------

def role(k: str) -> str:
    return C["ours"] if k == OURS else C["ref"] if k == REF else C["other"]


def name(k: str) -> str:
    return LABEL.get(k, k)


def new(ncols: int = 1, **kw):
    return plt.subplots(1, ncols, figsize=SIZE, layout="constrained", **kw)


def head(fig, title: str, sub: str, bottom: float = 0.0) -> None:
    """Title that states the finding, a wrapped subtitle, and the plot area below them (bottom leaves room for a
    legend or a footnote)."""
    lines = textwrap.wrap(sub, 112)
    fig.finding = title
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=C["ink"])
    fig.text(0.012, 0.905, "\n".join(lines), ha="left", va="top", fontsize=8, color=C["mute"], linespacing=1.35)
    top = 0.885 - 0.05 * len(lines)
    fig.get_layout_engine().set(rect=(0, bottom, 1, top - bottom))


def foot(fig, text: str) -> None:
    fig.text(0.012, 0.012, text, ha="left", va="bottom", fontsize=7, color=C["mute"])


def err(ax, x, y, lo, hi, horizontal: bool = False) -> None:
    kw = dict(fmt="none", ecolor=C["ink"], elinewidth=0.9, capsize=2, capthick=0.9, zorder=4)
    if horizontal:
        ax.errorbar(y, x, xerr=[[y - lo], [hi - y]], **kw)
    else:
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], **kw)


def save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=200)
    fig.savefig(out / f"{stem}.svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    DONE.append((stem, fig.finding))
    print(f"wrote {stem}.png and {stem}.svg")


# ---------- figures ----------

def fig_mac_comparison(src: Path, out: Path) -> None:
    files = read_files(src)
    keys = sorted((k for k in files if files[k]["T2"]), key=lambda k: files[k]["T1"]["kl"])
    fig, axes = new(2, sharey=True)
    y = np.arange(len(keys))[::-1]
    for ax, t, what in zip(axes, ("T1", "T2"), ("T1, held-out text", "T2, held-out reasoning traces")):
        for yi, k in zip(y, keys):
            f = files[k]
            v = f[t]["kl"]
            lo, hi = boot(f[t]["chunks"])
            ax.barh(yi, v, height=0.62, color=role(k), zorder=3, hatch="////" if f["over"] else None,
                    edgecolor="white", linewidth=0)
            err(ax, yi, v, lo, hi, horizontal=True)
            ax.text(hi + 0.0015, yi, f"{v:.4f}", va="center", ha="left", fontsize=SMALL,
                    fontweight="bold" if k in (OURS, REF) else "normal")
        ax.set_title(what, fontsize=9, fontweight="normal", color=C["mute"])
        ax.set_xlabel("KL to the Q8_0 reference (lower is closer)", fontsize=8)
        ax.set_xlim(0, max(files[k][t]["kl"] for k in keys) * 1.3)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="both", length=0, labelsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([""] * len(keys))
    for yi, k in zip(y, keys):
        f = files[k]
        gib = f"{f['gb']:.2f} GB" if f["gb"] else "size not measured"
        metal = "" if f["off"] is None else ", all on Metal" if f["off"] == 0 else f", {f['off']} expert tensors off Metal"
        tr = axes[0].get_yaxis_transform()
        axes[0].text(-0.03, yi + 0.13, name(k), transform=tr, ha="right", va="center", fontsize=8.5,
                     fontweight="bold" if k in (OURS, REF) else "normal")
        axes[0].text(-0.03, yi - 0.22, gib + (", over the limit" if f["over"] else "") + metal, transform=tr,
                     ha="right", va="center", fontsize=6.5, color=C["warn"] if f["over"] or f["off"] else C["mute"])
    o, r = files[OURS], files[REF]
    under = [k for k in keys if k not in (OURS, REF) and not files[k]["over"] and files[k]["off"] == 0]
    lead = all(o[t]["kl"] < files[k][t]["kl"] for k in under for t in ("T1", "T2"))
    title = ("Unsloth UD-IQ3_XXS is closest to the reference. Ours is next." if lead and o["T1"]["kl"] > r["T1"]["kl"]
             else "Held-out KL of every scored file")
    head(fig, title, f"Mean KL at P = 3, 95% chunk intervals. Size limit {r['gb']:.2f} GB, the size of "
                     f"UD-IQ3_XXS. Hatched bars are over it.", bottom=0.07)
    fig.legend(handles=[Patch(color=C["ours"], label="ours"), Patch(color=C["ref"], label="Unsloth UD-IQ3_XXS"),
                        Patch(color=C["other"], label="other files"),
                        Patch(facecolor=C["other"], hatch="////", edgecolor="white", label="over the size limit")],
               loc="lower left", bbox_to_anchor=(0.005, 0.0), ncols=4, fontsize=7.5, handlelength=1.2, columnspacing=1.2)
    save(fig, out, "mac-comparison")


def fig_speed_vs_kl(src: Path, out: Path) -> None:
    files = {k: f for k, f in read_files(src).items() if f["bench"] and f["bench"]["tests"]}
    fig, axes = new(2)
    marker = lambda k: "o" if k == OURS else "s" if k == REF else "^"  # noqa: E731
    # label placement (points): chosen so the names in the cluster do not overlap
    order = sorted(files, key=lambda k: files[k]["T1"]["kl"])
    for ax, test, what in zip(axes, ("tg128", "pp512"), ("Generation, tg128", "Prompt processing, pp512")):
        top = max(f["bench"]["tests"][(test, 0)][0] for f in files.values())
        for i, k in enumerate(order):
            f = files[k]
            x = f["T1"]["kl"]
            v, sd = f["bench"]["tests"][(test, 0)]
            ax.errorbar(x, v, yerr=sd, fmt="none", ecolor=C["mute"], elinewidth=0.8, capsize=1.5, zorder=3)
            ax.scatter(x, v, s=58, marker=marker(k), zorder=5, linewidths=1.4,
                       facecolor="white" if f["over"] else role(k), edgecolor="white" if not f["over"] else role(k))
            if f["over"]:
                ax.scatter(x, v, s=58, marker=marker(k), zorder=6, linewidths=1.4, facecolor="none", edgecolor=role(k))
            key = k in (OURS, REF)
            text = f"{name(k)}\n{v:.1f} t/s" if key else name(k).split(" ", 1)[-1]
            if k == REF:
                ax.annotate(text, (x, v), xytext=(9, 0), textcoords="offset points", ha="left", va="center",
                            fontsize=SMALL, fontweight="bold")
            elif k == OURS:
                ax.annotate(text, (x, v), xytext=(0, -11), textcoords="offset points", ha="center", va="top",
                            fontsize=SMALL, fontweight="bold")
            elif test == "tg128":
                up = [u for u in order if u not in (OURS, REF)].index(k) % 2 == 0
                ax.annotate(text, (x, v), xytext=(0, 9 if up else -9), textcoords="offset points", ha="center",
                            va="bottom" if up else "top", fontsize=6.5, color=C["mute"])
        rest = [(f["T1"]["kl"], f["bench"]["tests"][(test, 0)][0]) for k, f in files.items() if k not in (OURS, REF)]
        if test != "tg128" and rest:
            high = max(sum(f["bench"]["tests"][(test, 0)]) for k, f in files.items() if k not in (OURS, REF))
            ax.text(np.mean([x for x, _ in rest]), high * 1.03,
                    f"other files, {min(v for _, v in rest):.0f} to {max(v for _, v in rest):.0f} t/s", ha="center",
                    va="bottom", fontsize=6.5, color=C["mute"])
        ax.set_ylim(0, top * 1.22)
        xs = [f["T1"]["kl"] for f in files.values()]
        ax.set_xlim(min(xs) - 0.008, max(xs) + 0.009)
        ax.set_title(what, fontsize=9, fontweight="normal", color=C["mute"])
        ax.set_xlabel("T1 KL to the Q8_0 reference", fontsize=8)
        ax.set_ylabel("tokens per second", fontsize=8)
        ax.tick_params(labelsize=8, length=0)
    o, r = files[OURS]["bench"]["tests"], files[REF]["bench"]["tests"]
    metal = [f["bench"]["tests"][("tg128", 0)][0] for f in files.values() if f["off"] == 0]
    build = files[OURS]["bench"]["build"]
    head(fig, f"All-Metal files generate at {min(metal):.1f} to {max(metal):.1f} t/s. UD-IQ3_XXS runs at "
              f"{r[('tg128', 0)][0]:.1f}.",
         f"llama-bench with Metal, build {build}, P = 3. Ours against UD-IQ3_XXS: "
         f"{o[('tg128', 0)][0] / r[('tg128', 0)][0]:.2f}x generation, {o[('pp512', 0)][0] / r[('pp512', 0)][0]:.2f}x prompt.", bottom=0.07)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C["ours"], label="ours"),
                        Line2D([], [], marker="s", ls="", color=C["ref"],
                               label=f"Unsloth UD-IQ3_XXS ({files[REF]['off']} expert tensors off Metal)"),
                        Line2D([], [], marker="^", ls="", color=C["other"], label="other files, all on Metal"),
                        Line2D([], [], marker="^", ls="", markerfacecolor="white", color=C["other"],
                               label="over the size limit")],
               loc="lower left", bbox_to_anchor=(0.005, 0.0), ncols=4, fontsize=7, handletextpad=0.2, columnspacing=1.0)
    save(fig, out, "speed-vs-kl")


def fig_per_pass(src: Path, out: Path) -> None:
    pp = read_perpass(src)
    singles = [pp[f"p{i}"] for i in (1, 2, 3)]
    total = sum(s["chunks"] for s in singles)
    ratio, rlo, rhi = boot_ratio(pp["all"]["chunks"], total)
    fig, ax = new(1)
    shades = [C["ours"], "#ea5a90", "#f3a1bf"]  # lighter steps of C["ours"] for passes 2 and 3 inside the stack
    for i, s in enumerate(singles):
        ax.bar(i, s["kl"], width=0.62, color=C["ours"], zorder=3)
        err(ax, i, s["kl"], *boot(s["chunks"]))
        ax.text(i, boot(s["chunks"])[1] + 0.0015, f"{s['kl']:.4f}", ha="center", va="bottom", fontsize=8)
    base = 0.0
    for i, s in enumerate(singles):
        ax.bar(3, s["kl"], bottom=base, width=0.62, color=shades[i], zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(2.72, base + s["kl"] / 2, f"pass {i + 1}", ha="left", va="center", fontsize=7,
                color="white" if i < 2 else C["ink"])
        base += s["kl"]
    err(ax, 3, float(total.mean()), *boot(total))
    ax.text(3, boot(total)[1] + 0.0015, f"{total.mean():.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    a = pp["all"]
    ax.bar(4, a["kl"], width=0.62, color=C["ours_dark"], zorder=3)
    err(ax, 4, a["kl"], *boot(a["chunks"]))
    ax.text(4, boot(a["chunks"])[1] + 0.0015, f"{a['kl']:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(5), ["pass 1 only", "pass 2 only", "pass 3 only", "sum of the\nthree runs", "all three passes\nin one run"],
                  fontsize=8)
    ax.set_ylabel("KL to the IQ4_NL base", fontsize=8)
    ax.set_ylim(0, boot(a["chunks"])[1] * 1.18)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0, labelsize=8)
    head(fig, f"All three passes at 2-bit cost {ratio:.2f} times the sum of the single passes",
         f"Layers 6-11 at 2-bit, P = 3, 16 chunks of S. Ratio {ratio:.3f} [{rlo:.3f}, {rhi:.3f}]. "
         f"Bars are 95% chunk intervals.")
    save(fig, out, "per-pass")


def fig_split(src: Path, out: Path) -> None:
    rows = [("Pass split, 2-bit", "perpass-residual.json", C["ours"], "o"),
            ("Pass split, 3-bit", "perpass3bit-residual.json", C["ours_dark"], "D"),
            ("Layer split, 2-bit (control)", "layersplit-residual.json", C["ref"], "s")]
    data = [(lab, json.load(open(src / f))["fisher"], col, mk) for lab, f, col, mk in rows]
    fig, axes = new(2, sharey=True)
    stats = [("r", "Residual share r\n0 means the parts add up exactly"),
             ("norm_ratio", "Norm ratio\n1 means the parts add up exactly")]
    y = np.arange(len(data))[::-1]
    for ax, (key, what) in zip(axes, stats):
        target = 0.0 if key == "r" else 1.0
        ax.axvline(target, color=C["mute"], lw=0.9, ls=(0, (3, 2)), zorder=2)
        for yi, (lab, d, col, mk) in zip(y, data):
            lo, hi = d["ci95"][key]
            ax.plot([lo, hi], [yi, yi], color=col, lw=2.2, solid_capstyle="round", zorder=3)
            ax.scatter(d[key], yi, s=64, marker=mk, color=col, edgecolor="white", linewidths=1.4, zorder=5)
            ax.annotate(f"{d[key]:.3f}  [{lo:.3f}, {hi:.3f}]", (d[key], yi), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", fontsize=SMALL)
        ax.set_title(what, fontsize=8.5, fontweight="normal", color=C["mute"])
        ax.set_xlim((-0.04, 1.0) if key == "r" else (0.6, 1.03))
        ax.set_ylim(-0.6, len(data) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0, labelsize=8)
    axes[0].set_yticks(y, [d[0] for d in data], fontsize=8.5)
    axes[0].set_xlabel("||R|| / ||all||", fontsize=8)
    axes[1].set_xlabel("||all|| / ||sum of parts||", fontsize=8)
    by = {d[0]: d[1] for d in data}
    p, l = by["Pass split, 2-bit"], by["Layer split, 2-bit (control)"]
    apart = all(p["ci95"][k][1] < l["ci95"][k][0] or p["ci95"][k][0] > l["ci95"][k][1] for k in ("r", "norm_ratio"))
    closer = p["r"] < l["r"] and abs(1 - p["norm_ratio"]) < abs(1 - l["norm_ratio"])
    title = ("Splitting by pass adds up more closely than splitting by layer" if apart and closer
             else "Pass split against the layer-split control")
    head(fig, title, "Layers 6-11, Fisher norm, P = 3, 16 chunks of S, 95% chunk intervals. "
                     "The two 2-bit rows " + ("do not overlap." if apart else "overlap."))
    save(fig, out, "pass-vs-layer-split")


def fig_noise(src: Path, out: Path) -> None:
    pp = read_perpass(src)
    tied = [k for k in ("noise-tied", "noise-tied2", "noise-tied3") if k in pp]
    fig, ax = new(1)
    bars = [(k, f"same draw on all\npasses, draw {i + 1}", C["ours"]) for i, k in enumerate(tied)]
    bars.append(("noise-indep", "different draw\non each pass", C["ref"]))
    for i, (k, lab, col) in enumerate(bars):
        v = pp[k]["kl"]
        lo, hi = boot(pp[k]["chunks"])
        ax.bar(i, v, width=0.62, color=col, zorder=3, hatch="////" if k == "noise-indep" else None, edgecolor="white",
               linewidth=0)
        err(ax, i, v, lo, hi)
        ax.text(i, hi + 0.0015, f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(bars)), [b[1] for b in bars], fontsize=8)
    ax.set_ylabel("KL to the IQ4_NL base", fontsize=8)
    ax.set_ylim(0, max(boot(pp[k]["chunks"])[1] for k, *_ in bars) * 1.18)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0, labelsize=8)
    ind = pp["noise-indep"]
    ratios = [boot_ratio(pp[k]["chunks"], ind["chunks"]) for k in tied]
    stored = ast.literal_eval((src / "noise-ratio.txt").read_text()) if (src / "noise-ratio.txt").exists() else None
    sub = "Random weight noise of the 2-bit size on layers 6-11, P = 3, 16 chunks of S. "
    if stored:
        sub += f"Draw 1 ratio {stored['ratio']:.3f}, 95% interval {stored['ci95'][0]:.3f} to {stored['ci95'][1]:.3f}."
    head(fig, f"Noise shared by all passes costs {min(r[0] for r in ratios):.2f} to {max(r[0] for r in ratios):.2f} "
              f"times independent noise", sub)
    save(fig, out, "noise")


def block_layers(src: Path) -> dict[str, tuple[int, str]]:
    """Block -> (number of layers, description), from the block map table in phase2.md."""
    out = {}
    for line in (src / "phase2.md").read_text().splitlines():
        m = re.match(r"\|\s*(B\d)\s*\|\s*(\d+)-(\d+),?\s*([^|]*)\|", line)
        if m:
            out[m.group(1)] = (int(m.group(3)) - int(m.group(2)) + 1, f"{m.group(2)}-{m.group(3)}", m.group(4).strip())
    return out


def fig_block_map(src: Path, out: Path) -> None:
    kl = read_kl(src)
    blocks = block_layers(src)
    levels = [(k, lab) for k, lab in LEVELS if all(f"map-{b}-{k}-S-P3" in kl for b in blocks)]
    fig, axes = new(len(levels))
    looped = {b: "looped" in d[2] and "before" not in d[2] and "after" not in d[2] for b, d in blocks.items()}
    per = {}
    for ax, (lev, lab) in zip(axes, levels):
        top = 0.0
        for i, (b, (n, _, _)) in enumerate(blocks.items()):
            r = kl[f"map-{b}-{lev}-S-P3"]
            v = r["kl"] / n * 1000
            lo, hi = (x / n * 1000 for x in boot(r["chunks"]))
            per[(lev, b)] = v
            ax.bar(i, v, width=0.7, color=C["ours"] if looped[b] else C["other"], zorder=3)
            err(ax, i, v, lo, hi)
            ax.text(i, hi + 0.0, f"{v:.1f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
            top = max(top, hi)
        ax.set_ylim(0, top * 1.15)
        ax.set_xticks(range(len(blocks)), list(blocks), fontsize=7)
        ax.set_title(lab, fontsize=7.5, fontweight="normal", color=C["mute"])
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=0, labelsize=7)
    axes[0].set_ylabel("KL per layer, in thousandths", fontsize=8)
    lev0 = levels[0][0]
    inside = max(per[(lev0, b)] for b in blocks if looped[b])
    after = min(per[(lev0, b)] for b, d in blocks.items() if "after" in d[2])
    title = ("At 2-bit, a layer after the loop costs more than a looped layer" if after > inside
             else "KL per layer of each block at the five map levels")
    head(fig, title, "One block changed, everything else Q8_0, P = 3, 16 chunks of S. KL of the block divided by its "
                     "number of layers. Each panel has its own scale. Layers: "
                     + ", ".join(f"{b} {d[1]}" for b, d in blocks.items()) + ".", bottom=0.07)
    fig.legend(handles=[Patch(color=C["ours"], label="looped layers (6-17, run 3 times)"),
                        Patch(color=C["other"], label="layers outside the loop")],
               loc="lower left", bbox_to_anchor=(0.005, 0.0), ncols=2, fontsize=7.5, handlelength=1.2)
    save(fig, out, "block-map")


def fig_pass_sweep(src: Path, out: Path) -> None:
    kl = read_kl(src)
    fig, axes = new(2)
    lows = {}
    for ax, lev, col, mk in zip(axes, ("2bit", "3bit"), (C["ours"], C["ours_dark"]), ("o", "D")):
        ps = sorted(int(m.group(1)) for r in kl if (m := re.fullmatch(rf"sweep-{lev}-S-P(\d+)", r)))
        vals = [kl[f"sweep-{lev}-S-P{p}"] for p in ps]
        ys = [v["kl"] for v in vals]
        ax.plot(ps, ys, color=col, lw=2, zorder=3)
        for p, v in zip(ps, vals):
            lo, hi = boot(v["chunks"])
            err(ax, p, v["kl"], lo, hi)
            ax.scatter(p, v["kl"], s=64, marker=mk, color=col, edgecolor="white", linewidths=1.4, zorder=5)
            ax.annotate(f"{v['kl']:.4f}", (p, max(hi, v["kl"] + 0.06 * max(ys))), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        va="bottom", fontsize=8, fontweight="bold" if v["kl"] == min(ys) else "normal")
        lows[lev] = ps[int(np.argmin(ys))]
        ax.set_xticks(ps, [f"P = {p}" for p in ps], fontsize=8)
        ax.set_xlim(min(ps) - 0.4, max(ps) + 0.4)
        ax.set_ylim(0, max(ys) * 1.2)
        ax.set_title(f"Every expert tensor at {lev[0]}-bit", fontsize=9, fontweight="normal", color=C["mute"])
        ax.set_ylabel("KL to the Q8_0 reference", fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=0, labelsize=8)
    same = len(set(lows.values())) == 1
    p0 = lows["2bit"]
    title = f"KL is lowest at P = {p0} at both bit levels and higher on either side" if same else "KL against the pass count"
    head(fig, title, "Uniform builds on 16 chunks of S, with 95% chunk intervals. Each panel has its own scale.", bottom=0.07)
    foot(fig, "Each P is scored against the Q8_0 reference run at the same P, so the three points have three different "
              "references.")
    save(fig, out, "pass-sweep")


def fig_depth_speed(src: Path, out: Path) -> None:
    """Speed against context depth. Needs depth-*.txt (scripts/depth-bench.sh). Skips without a word otherwise."""
    paths = sorted(src.glob("depth-*.txt"))
    files = {p.stem[len("depth-"):]: read_bench(p) for p in paths}
    files = {k: f for k, f in files.items() if f["tests"]}
    if not files:
        return
    fig, axes = new(2)
    marker = lambda k: "o" if k == OURS else "s" if k == REF else "^"  # noqa: E731
    for ax, kind, what in zip(axes, ("tg", "pp"), ("Generation", "Prompt processing")):
        top = 0.0
        for k, f in files.items():
            pts = sorted((d, v, sd) for (t, d), (v, sd) in f["tests"].items() if t.startswith(kind))
            if not pts:
                continue
            ds = list(range(len(pts)))
            ax.plot(ds, [p[1] for p in pts], color=role(k), lw=2, zorder=3)
            ax.errorbar(ds, [p[1] for p in pts], yerr=[p[2] for p in pts], fmt="none", ecolor=C["mute"], elinewidth=0.8,
                        capsize=1.5, zorder=4)
            ax.scatter(ds, [p[1] for p in pts], s=52, marker=marker(k), color=role(k), edgecolor="white", linewidths=1.4,
                       zorder=5)
            for i, p in enumerate(pts):
                if k in (OURS, REF):
                    ax.annotate(f"{p[1]:.1f}", (i, p[1]), xytext=(0, 8 if k == OURS else -8), textcoords="offset points",
                                ha="center", va="bottom" if k == OURS else "top", fontsize=SMALL)
            ax.set_xticks(ds, ["empty" if p[0] == 0 else f"{p[0]:,}" for p in pts], fontsize=8)
            top = max(top, max(p[1] for p in pts))
        ax.set_ylim(0, top * 1.2)
        ax.set_xlim(-0.35, len(ds) - 0.65)
        ax.set_title(what, fontsize=9, fontweight="normal", color=C["mute"])
        ax.set_xlabel("tokens already in the context", fontsize=8)
        ax.set_ylabel("tokens per second", fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=0, labelsize=8)
    mem = []
    for k in files:
        p = src / f"memory-{k}.txt"
        m = re.search(r"(\d+)\s+peak memory footprint", p.read_text()) if p.exists() else None
        if m:
            mem.append(f"{name(k)} {int(m.group(1)) / 2**30:.2f} GiB")
    deep = max(d for f in files.values() for (_, d) in f["tests"])
    both = OURS in files and REF in files and ("tg128", deep) in files[OURS]["tests"] and ("tg128", deep) in files[REF]["tests"]
    title = (f"At {deep:,} tokens of context ours generates at {files[OURS]['tests'][('tg128', deep)][0]:.1f} t/s, "
             f"UD-IQ3_XXS at {files[REF]['tests'][('tg128', deep)][0]:.1f}" if both else "Speed against context depth")
    head(fig, title, "llama-bench with Metal, P = 3, 2 repetitions per depth.", bottom=0.12)
    if mem:
        foot(fig, f"Peak memory footprint at {deep:,} tokens: " + ", ".join(mem) + ".")
    fig.legend(handles=[Line2D([], [], marker=marker(k), color=role(k), label=name(k)) for k in files],
               loc="lower left", bbox_to_anchor=(0.005, 0.045), ncols=len(files), fontsize=7.5)
    save(fig, out, "depth-speed")


def write_readme(out: Path) -> None:
    """figures/README.md: one row per figure, with the finding taken from the title the script computed."""
    L = ["# Figures", "",
         "Written by `scripts/figures.py`. Do not edit by hand. Rerun with `.venv/bin/python -I scripts/figures.py`.", "",
         "Every value is read from `results/pilot/`. Each figure exists as PNG (200 dpi) and SVG, 6.4 x 3.6 in.", "",
         "| Figure | Source files in `results/pilot/` | Finding |", "|---|---|---|"]
    L += [f"| [`{stem}.png`]({stem}.png), [`{stem}.svg`]({stem}.svg) | {SOURCES[stem]} | {title.rstrip('.')}. |"
          for stem, title in DONE]
    L += ["", "## How to read them", "",
          "- Colour comes from `analysis/style.py`. Magenta is our file and the looped layers and passes. Navy is Unsloth "
          "UD-IQ3_XXS and every control. Muted blue is the other files and the layers outside the loop.",
          "- Shape, hatching and direct labels repeat what colour says.",
          "- Error bars on KL are 95% percentile bootstrap intervals over chunks (10,000 resamples, seed 0). T2 has 5 "
          "chunks, so its intervals are coarse.",
          "- Error bars on speed are the spread `llama-bench` prints.",
          "- File size is the size `llama-bench` reports. The size limit is the size of UD-IQ3_XXS by the same measure.",
          "- KL in `mac-comparison`, `speed-vs-kl`, `block-map` and `pass-sweep` is against the Q8_0 reference. KL in "
          "`per-pass` and `noise` is against the IQ4_NL base.",
          "- In `pass-sweep` each P is scored against the reference run at the same P.",
          "- `per-pass`, `pass-vs-layer-split` and `noise` cover layers 6-11 only and are exploratory.",
          "- `depth-speed` is drawn only when `depth-*.txt` exist."]
    (out / "README.md").write_text("\n".join(L) + "\n")
    print("wrote README.md")


FIGURES = [fig_mac_comparison, fig_speed_vs_kl, fig_per_pass, fig_split, fig_noise, fig_block_map, fig_pass_sweep,
           fig_depth_speed]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=ROOT / "results/pilot")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    apply()
    plt.rcParams.update({"savefig.bbox": "standard", "hatch.linewidth": 1.2, "hatch.color": "white",
                         "svg.hashsalt": "figures"})
    for f in FIGURES:
        try:
            f(args.src, args.out)
        except FileNotFoundError as e:
            print(f"skipped {f.__name__}: {Path(e.filename).name} is missing")
    write_readme(args.out)


if __name__ == "__main__":
    main()
