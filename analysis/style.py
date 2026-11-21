"""One visual style for every figure, the hero image and the README (same palette as the pilot status page).

    from analysis.style import C, apply
    apply()            # matplotlib rcParams
    C["ours"]          # hex colours by role

SVG hand-drawn images use the same hex values (see SVG_CSS).
"""

C = {
    "ink": "#0b1020",      # text, axes
    "mute": "#4a5470",     # secondary text, ticks
    "rule": "#d9deea",     # grid, hairlines
    "panel": "#f3f5fb",    # backgrounds of boxes
    "ours": "#e0115f",     # our file; looped layers / passes
    "ours_dark": "#b80d4d",
    "ref": "#0b1f4b",      # Unsloth UD-IQ3_XXS; controls; the reference
    "other": "#9fb0dc",    # other comparators; non-looped layers
    "ok": "#1f7a4d",       # pass / better
    "warn": "#8a5a00",     # caution / over the size limit
}
FONT = ["Red Hat Text", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]
SVG_CSS = "\n".join(f"  --{k.replace('_', '-')}: {v};" for k, v in C.items())


def apply() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 10,
        "axes.edgecolor": C["mute"], "axes.labelcolor": C["ink"], "axes.titlecolor": C["ink"],
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["rule"], "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": C["mute"], "ytick.color": C["mute"], "text.color": C["ink"],
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
