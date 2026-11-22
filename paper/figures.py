"""Figures for the paper, read from results/pilot/ only.

    .venv/bin/python -I paper/figures.py

Writes paper/fig/{additivity,comparison,transfer}.pdf, and a PNG of each for paper.md. Intervals are 95% percentile
bootstraps over chunks (10,000 resamples, seed 0), as in analysis/compare.py and analysis/noise.py.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import noise  # noqa: E402
from style import C  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("perpass_summary", ROOT / "scripts/perpass-summary.py")
pps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pps)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "paper/fig"
OTHER = "#6f84b8"  # C["other"] darkened for 3:1 contrast on white paper
N_BOOT, SEED = 10000, 0


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": C["mute"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
        "xtick.color": C["mute"], "ytick.color": C["ink"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": C["rule"], "grid.linewidth": 0.5,
        "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)  # for paper.md


def kl_runs() -> dict[str, np.ndarray]:
    rows = csv.DictReader(open(ROOT / "results/pilot/kl.tsv"), delimiter="\t")
    return {r["run"]: np.array([float(x) for x in r["per_chunk"].split(",")]) for r in rows}


def boot_idx(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))


def mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    b = x[boot_idx(x.size)].mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def additivity() -> None:
    """KL with every visit changed over the sum of the single-visit KLs, against the three reference values."""
    kc = pps.chunks()
    rows = [  # label, all, parts, colour, marker
        ("B2, 2-bit quantization", "all", ["p1", "p2", "p3"], C["ours"], "o"),
        ("B3, 2-bit quantization", "b3-all", ["b3-p1", "b3-p2", "b3-p3"], C["ours"], "o"),
        ("B2, tied noise (same draw every pass)", "noise-tied", ["g1-p1", "g1-p2", "g1-p3"], C["ref"], "s"),
        ("B2, independent noise (draw k on pass k)", "noise-indep", ["g1-p1", "g2-p2", "g3-p3"], C["ref"], "s"),
        ("B2, 3-bit layer split (control, not a visit split)", "3b-all", ["3b-ls67", "3b-ls89", "3b-ls1011"], OTHER, "D"),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 2.35))
    ax.grid(False)
    labels = []
    for i, (lab, a, parts, col, mk) in enumerate(rows):
        r = noise.paired_ratio([kc[a]], [kc[p] for p in parts])
        lo, hi = r["ci95"]
        y = len(rows) - i
        ax.errorbar(r["ratio"], y, xerr=[[r["ratio"] - lo], [hi - r["ratio"]]], fmt=mk, color=col, ms=5, lw=1.2, capsize=0)
        ax.annotate(f"{r['ratio']:.3f}", (hi, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7, color=C["mute"],
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        labels.append(lab)
    # the 3-bit pass split has means only (its per-chunk logits were deleted), so it is a point without an interval
    p2 = {r["cond"]: float(r["kl_vs_base"]) for r in csv.DictReader(open(ROOT / "results/pilot/perpass2-kl.tsv"), delimiter="\t")}
    ratio3 = p2["3b-all"] / (p2["3b-p1"] + p2["3b-p2"] + p2["3b-p3"])
    ax.plot(ratio3, 0, "o", mfc="white", mec=C["ours"], ms=5, mew=1.2)
    ax.annotate(f"{ratio3:.3f} (point only)", (ratio3, 0), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7, color=C["mute"],
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    labels.append("B2, 3-bit quantization")
    ax.set_yticks(range(len(rows), -1, -1), labels)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 4.2)
    ax.set_ylim(-0.7, len(rows) + 1.1)
    ax.set_xticks([1 / 3, 0.5, 1, 2, 3], ["1/3", "0.5", "1", "2", "3"])
    ax.minorticks_off()
    for x, txt in ((1 / 3, "counts once"), (1, "independent per visit"), (3, "aligned visits")):
        ax.axvline(x, color=C["mute"], lw=0.7, ls=(0, (3, 2)), zorder=0)
        ax.text(x, len(rows) + 0.75, txt, ha="center", va="center", fontsize=7, color=C["mute"], bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.set_xlabel("KL with all three visits changed / sum of the three single-visit KLs (log scale)")
    ax.tick_params(axis="y", length=0)
    save(fig, "additivity")
    plt.close(fig)


COMPARE = [  # run name in kl.tsv, label, over the size limit, all expert tensors in Metal types
    ("UD-IQ3_XXS", "Unsloth UD-IQ3_XXS (sets the limit)", False, False),
    ("cand", "v1, our candidate", False, True),
    ("bart-IQ2_M", "bartowski IQ2_M (over the limit)", True, True),
    ("atomic-AD-IQ2_S", "AtomicChat AD-IQ2_S", False, True),
    ("bart-IQ2_S", "bartowski IQ2_S", False, True),
    ("isvalorum-NanoPlus", "IsValorum APEX-I-NanoPlus-V2.1", False, True),
    ("uniform-IQ3_XXS", "stock IQ3_XXS (over the limit)", True, True),
    ("apex-I-Nano", "mudler APEX-I-Nano", False, True),
    ("arelius-Q2K-Q2_0", "arelius Q2K-Q2_0", False, True),
    ("devquasar-Q2_K", "DevQuasar Q2_K", False, True),
    ("nikolaus-Q2_K", "nikolaus714 Q2_K", False, True),
    ("prithiv-Q2_K", "prithivMLmods Q2_K", False, True),
]


def comparison() -> None:
    runs = kl_runs()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    for ax, t in zip(axes, ("T1", "T2")):
        for i, (name, lab, over, metal) in enumerate(COMPARE):
            m, lo, hi = mean_ci(runs[f"final-{name}-{t}-P3"])
            col = C["ours"] if name == "cand" else C["ref"] if name == "UD-IQ3_XXS" else OTHER
            mk = "s" if name == "UD-IQ3_XXS" else "o"
            y = len(COMPARE) - 1 - i
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=4.5, lw=1.1, mfc="white" if over else col, mew=1.1)
        ax.set_xscale("log")
        ax.set_xlim(0.01, 0.15)
        ax.set_xticks([0.01, 0.02, 0.05, 0.1], ["0.01", "0.02", "0.05", "0.1"])
        ax.minorticks_off()
        ax.set_title(f"{t} ({'16 chunks, WikiText-2' if t == 'T1' else '5 chunks, reasoning traces'})")
        ax.set_xlabel("mean KL to the Q8_0 reference (log scale)")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(COMPARE) - 1, -1, -1), [c[1] for c in COMPARE])
    save(fig, "comparison")
    plt.close(fig)


def transfer() -> None:
    """v2a, v2r and UD-IQ3_XXS on S, T1 and T2, with the share of the v2a-to-UD gap that v2r closes."""
    runs = kl_runs()
    name = {"S": "{}-S-P3", "T1": "final-{}-T1-P3", "T2": "final-{}-T2-P3"}
    files = [("v2a", "v2a (llama-quantize)", OTHER, "o"), ("v2r", "v2r (refined codes)", C["ours"], "o"), ("UD-IQ3_XXS", "UD-IQ3_XXS", C["ref"], "s")]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.75), sharey=True)
    for ax, s in zip(axes, ("S", "T1", "T2")):
        x = {f: runs[name[s].format(f)] for f, *_ in files}
        for i, (f, lab, col, mk) in enumerate(files):
            m, lo, hi = mean_ci(x[f])
            ax.errorbar(m, len(files) - 1 - i, xerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=4.5, lw=1.1)
        idx = boot_idx(x["v2a"].size)
        num = (x["v2a"] - x["v2r"])[idx].mean(axis=1)
        den = (x["v2a"] - x["UD-IQ3_XXS"])[idx].mean(axis=1)
        share = (x["v2a"] - x["v2r"]).mean() / (x["v2a"] - x["UD-IQ3_XXS"]).mean()
        lo, hi = np.percentile(num / den, [2.5, 97.5])
        ax.set_title(f"{s}: gap closed {share:.2f} [{lo:.2f}, {hi:.2f}]", fontsize=7.5)
        ax.set_xlim(0.0115, 0.0305)
        ax.set_xticks([0.015, 0.02, 0.025, 0.03])
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([2, 1, 0], [f[1] for f in files])
    axes[1].set_xlabel("mean KL to the Q8_0 reference")
    save(fig, "transfer")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    additivity()
    comparison()
    transfer()


if __name__ == "__main__":
    main()
