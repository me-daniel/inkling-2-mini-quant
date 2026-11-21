# Figures

Written by `scripts/figures.py`. Do not edit by hand. Rerun with `.venv/bin/python -I scripts/figures.py`.

Every value is read from `results/pilot/`. Each figure exists as PNG (200 dpi) and SVG, 6.4 x 3.6 in.

| Figure | Source files in `results/pilot/` | Finding |
|---|---|---|
| [`mac-comparison.png`](mac-comparison.png), [`mac-comparison.svg`](mac-comparison.svg) | kl.tsv, bench-*.txt | Unsloth UD-IQ3_XXS is closest to the reference. Ours is next. |
| [`speed-vs-kl.png`](speed-vs-kl.png), [`speed-vs-kl.svg`](speed-vs-kl.svg) | kl.tsv, bench-*.txt | All-Metal files generate at 18.3 to 20.5 t/s. UD-IQ3_XXS runs at 10.9. |
| [`per-pass.png`](per-pass.png), [`per-pass.svg`](per-pass.svg) | perpass-kl.tsv | All three passes at 2-bit cost 1.14 times the sum of the single passes. |
| [`pass-vs-layer-split.png`](pass-vs-layer-split.png), [`pass-vs-layer-split.svg`](pass-vs-layer-split.svg) | perpass-residual.json, perpass3bit-residual.json, layersplit-residual.json | Splitting by pass adds up more closely than splitting by layer. |
| [`noise.png`](noise.png), [`noise.svg`](noise.svg) | perpass-kl.tsv, noise-ratio.txt | Noise shared by all passes costs 1.20 to 1.22 times independent noise. |
| [`block-map.png`](block-map.png), [`block-map.svg`](block-map.svg) | kl.tsv, phase2.md | At 2-bit, a layer after the loop costs more than a looped layer. |
| [`pass-sweep.png`](pass-sweep.png), [`pass-sweep.svg`](pass-sweep.svg) | kl.tsv | KL is lowest at P = 3 at both bit levels and higher on either side. |
| [`depth-speed.png`](depth-speed.png), [`depth-speed.svg`](depth-speed.svg) | depth-*.txt, memory-*.txt | At 16,384 tokens of context ours generates at 17.9 t/s, UD-IQ3_XXS at 12.2. |

## How to read them

- Colour comes from `analysis/style.py`. Magenta is our file and the looped layers and passes. Navy is Unsloth UD-IQ3_XXS and every control. Muted blue is the other files and the layers outside the loop.
- Shape, hatching and direct labels repeat what colour says.
- Error bars on KL are 95% percentile bootstrap intervals over chunks (10,000 resamples, seed 0). T2 has 5 chunks, so its intervals are coarse.
- Error bars on speed are the spread `llama-bench` prints.
- File size is the size `llama-bench` reports. The size limit is the size of UD-IQ3_XXS by the same measure.
- KL in `mac-comparison`, `speed-vs-kl`, `block-map` and `pass-sweep` is against the Q8_0 reference. KL in `per-pass` and `noise` is against the IQ4_NL base.
- In `pass-sweep` each P is scored against the reference run at the same P.
- `per-pass`, `pass-vs-layer-split` and `noise` cover layers 6-11 only and are exploratory.
- `depth-speed` is drawn only when `depth-*.txt` exist.
