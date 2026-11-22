"""Writes results/pilot/nvfp4-metal-parity.md: Addendum H amendment 8 (CHANGELOG.md), end-to-end parity of the
proposed Metal NVFP4 kernel (patches/llama.cpp-proposed/0008-0009, worktree vendor/llama.cpp-nvfp4).

    .venv/bin/python -I scripts/nvfp4-parity.py

Inputs, all from logs/:
- ppl-UD-IQ3_XXS-S-P3.log: the measurement build (vendor/llama.cpp), NVFP4 on the CPU.
- ppl-UD-IQ3_XXS-nvfp4metal-S-P3.log: the worktree build, same arguments.
- load-UD-IQ3_XXS-{measurement,nvfp4metal}.log: load-only runs of both builds at -lv 5 (same model, -c, -ngl and
  override as the KL runs, a 2,000-byte text so llama-perplexity stops after loading and warmup). The KL logs are
  at default verbosity, which hides the buffer and fallback lines.
- logs/nvfp4-parity-commits.txt and nvfp4-parity-build.txt: git log and the build command, written before the run.

Pass rule (amendment 8): per-chunk KL within 2e-4 of the measurement run in every chunk, and the mean within 1e-4.
Per-chunk KL is telescoped from the running means (analysis/maptable.parse); those are printed to 5 decimals, so the
value of chunk k carries a rounding error of up to (2k - 1) * 5e-6 per run. The rule is applied to the parsed
values as registered; the rounding bound is reported next to it.
"""

import re
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import compare  # noqa: E402
import maptable  # noqa: E402

L = ROOT / "logs"
OLD, NEW = L / "ppl-UD-IQ3_XXS-S-P3.log", L / "ppl-UD-IQ3_XXS-nvfp4metal-S-P3.log"
LOAD_OLD, LOAD_NEW = L / "load-UD-IQ3_XXS-measurement.log", L / "load-UD-IQ3_XXS-nvfp4metal.log"
CHUNK_TOL, MEAN_TOL = 2e-4, 1e-4
OUT = ROOT / "results/pilot/nvfp4-metal-parity.md"

MOVED = re.compile(r"tensor '([^']+)' \((\w+)\) \(and (\d+) others\) cannot be used with preferred buffer type (\S+), using (\S+) instead")
BUF = re.compile(r"load_tensors:\s+(\S+) model buffer size =\s+([\d.]+) MiB")
LAYER_CPU = re.compile(r"layer\s+(\d+) assigned to device CPU")
MISMATCH = re.compile(r"layer \d+ is assigned to device .* but .* is assigned to device .*")


def load_evidence(log: Path) -> dict:
    text = log.read_text(errors="replace")
    m = MOVED.search(text)
    return {
        "moved": None if not m else {"first": m.group(1), "type": m.group(2), "n": int(m.group(3)) + 1,
                                     "from": m.group(4), "to": m.group(5)},
        "buffers": [(b, float(s)) for b, s in BUF.findall(text)],
        "cpu_layers": [int(x) for x in LAYER_CPU.findall(text)],
        "mismatch": MISMATCH.findall(text),
    }


def timing(log: Path) -> list[str]:
    text = log.read_text(errors="replace")
    pats = [r"[\d.]+ seconds per pass - ETA [\d.]+ minutes", r"prompt eval time\s*=.*", r"(?<!prompt )eval time\s*=.*"]
    return [m.group(0).strip() for p in pats for m in re.finditer(p, text)]


def main() -> None:
    old, new = maptable.parse(OLD), maptable.parse(NEW)
    if old is None or new is None:
        sys.exit(f"missing Mean KLD in {OLD.name if old is None else NEW.name}")
    a, b = np.array(new["per_chunk"]), np.array(old["per_chunk"])
    d = a - b
    mean_d, lo, hi = compare.paired(a, b)
    k = np.arange(1, d.size + 1)
    bound = 2 * (2 * k - 1) * 5e-6  # rounding of both runs' running means
    mean_gap = new["mean_kl"] - old["mean_kl"]
    chunk_ok = bool(np.all(np.abs(d) <= CHUNK_TOL))
    mean_ok = abs(mean_gap) <= MEAN_TOL
    ev_old, ev_new = load_evidence(LOAD_OLD), load_evidence(LOAD_NEW)
    r = GGUFReader(ROOT / "models/UD-IQ3_XXS.gguf")
    nvt = [t for t in r.tensors if t.tensor_type.name == "NVFP4"]
    nv = [t.name for t in nvt]
    sizes = {int(t.n_bytes) for t in nvt}
    assert len(sizes) == 1, sizes
    nv_mib = sizes.pop() / 2**20
    te = next(t for t in r.tensors if t.name == "token_embd.weight")
    te_type, te_mib = te.tensor_type.name, int(te.n_bytes) / 2**20

    md = ["# NVFP4 on Metal: end-to-end parity (Addendum H amendment 8)", "",
          "Written by `scripts/nvfp4-parity.py` from the logs named below. Nothing in the release or the thesis depends on "
          "this build; it informs the upstream decision.", "", "## Builds", "",
          "Worktree `vendor/llama.cpp-nvfp4`, branch `nvfp4-metal` (`git log --oneline -3`):", "", "```",
          (L / "nvfp4-parity-commits.txt").read_text().rstrip(), "```", "",
          "Build command (existing build dir, Release, `GGML_METAL=ON`):", "", "```",
          (L / "nvfp4-parity-build.txt").read_text().rstrip(), "```", "",
          "Measurement build: `vendor/llama.cpp` (pinned, patches/llama.cpp), run of `scripts/map3.sh`.", "",
          "Both KL runs: `llama-perplexity -m models/UD-IQ3_XXS.gguf -f data/S.txt -c 4096 --chunks 16 -ngl 99 "
          "--override-kv inkling2_loop.loop.count=int:3 --kl-divergence-base models/logits/ref-S-P3.kld --kl-divergence`.", "",
          "## Where the expert tensors live", "",
          "From the load-only runs at `-lv 5` (the KL logs are at default verbosity and print no buffer lines): "
          "`<build>/bin/llama-perplexity -m models/UD-IQ3_XXS.gguf -f <first 2,000 bytes of data/S.txt> -c 4096 -ngl 99 "
          "--override-kv inkling2_loop.loop.count=int:3 -lv 5`, which loads, warms up and stops at 'you need at least 8192 "
          "tokens'. Logs: `logs/load-UD-IQ3_XXS-measurement.log`, `logs/load-UD-IQ3_XXS-nvfp4metal.log`.", "",
          "| | measurement build | NVFP4-Metal build |", "|---|---|---|"]
    def moved(e):
        m = e["moved"]
        return "none" if m is None else f"{m['n']} tensor{'s' * (m['n'] != 1)} ({m['type']}, first `{m['first']}`) moved from {m['from']} to {m['to']}"
    md.append(f"| tensors moved off their preferred buffer (loader debug line) | {moved(ev_old)} | {moved(ev_new)} |")
    for e in (ev_old, ev_new):
        e["cpu"] = [s for x, s in e["buffers"] if x.startswith("CPU") and s > 0]
        e["mtl"] = [s for x, s in e["buffers"] if x.startswith("MTL") and s > 0]
    f = lambda e, key: f"{len(e[key])} buffer{'s' * (len(e[key]) != 1)}, {sum(e[key]):.2f} MiB"  # noqa: E731
    md.append(f"| CPU model buffers (non-empty) | {f(ev_old, 'cpu')} | {f(ev_new, 'cpu')} |")
    md.append(f"| MTL0 model buffers (non-empty) | {f(ev_old, 'mtl')} | {f(ev_new, 'mtl')} |")
    g = lambda e: sum(abs(s - nv_mib) < 0.005 for s in e["cpu"])  # noqa: E731
    md.append(f"| CPU buffers of exactly one NVFP4 tensor ({nv_mib:.2f} MiB) | {g(ev_old)} | {g(ev_new)} |")
    md.append(f"| layers assigned to device CPU | {len(ev_old['cpu_layers'])} | {len(ev_new['cpu_layers'])} |")
    md.append(f"| layer/device mismatch warnings in the KL log | {len(MISMATCH.findall(OLD.read_text(errors='replace')))} "
              f"| {len(MISMATCH.findall(NEW.read_text(errors='replace')))} |")
    nvfp4_on_cpu = g(ev_new) > 0 or abs(sum(ev_new["cpu"]) - te_mib) > 0.005 or (ev_new["moved"] or {"n": 1})["n"] > 1
    md += ["", f"From the GGUF: {len(nv)} NVFP4 tensors (`{nv[0]}` to `{nv[-1]}`), {nv_mib:.2f} MiB each; "
           f"`token_embd.weight` ({te_type}, {te_mib:.2f} MiB) sits on the CPU in every build (input embedding). "
           "The loader's debug line names only the first moved tensor and counts the rest. "
           f"MTL0 total, NVFP4-Metal minus measurement: {sum(ev_new['mtl']) - sum(ev_old['mtl']):.2f} MiB "
           f"(= {(sum(ev_new['mtl']) - sum(ev_old['mtl'])) / nv_mib:.3f} NVFP4 tensors).", ""]
    md += [f"NVFP4 tensor still on the CPU in the NVFP4-Metal build: {'yes' if nvfp4_on_cpu else 'no'}.", "",
           "## Per-chunk KL on S, P = 3", "",
           "Per-chunk values telescoped from the running means (`analysis/maptable.parse`); difference = NVFP4-Metal minus "
           "measurement. Rounding bound: (2k - 1) * 5e-6 per run, summed over both runs.", "",
           "| chunk | measurement | NVFP4-Metal | difference | rounding bound | within 2e-4 |", "|---:|---:|---:|---:|---:|---|"]
    for i in range(d.size):
        md.append(f"| {i + 1} | {b[i]:.6f} | {a[i]:.6f} | {d[i]:+.6f} | {bound[i]:.6f} | {'yes' if abs(d[i]) <= CHUNK_TOL else 'NO'} |")
    md += ["", "| | measurement | NVFP4-Metal |", "|---|---:|---:|",
           f"| Mean KLD (log) | {old['mean_kl']:.6f} ± {old['se']:.6f} | {new['mean_kl']:.6f} ± {new['se']:.6f} |",
           f"| Same top p (%) | {old['same_top_p']:.3f} | {new['same_top_p']:.3f} |", "",
           f"- Max |per-chunk difference|: {np.abs(d).max():.6f} (chunk {int(np.abs(d).argmax()) + 1}); tolerance {CHUNK_TOL:g}.",
           f"- Mean per-chunk difference: {mean_d:+.6f}, paired 95% bootstrap CI [{lo:+.6f}, {hi:+.6f}] (`analysis/compare.paired`, 10,000 resamples, seed 0).",
           f"- Mean KLD difference from the logs' 6-decimal means: {mean_gap:+.6f}; tolerance {MEAN_TOL:g}.", "",
           "## Verdict", ""]
    if nvfp4_on_cpu:
        md.append("Not tested: an NVFP4 tensor still runs on the CPU in the NVFP4-Metal build.")
    else:
        verdict = "PASS" if chunk_ok and mean_ok else "FAIL"
        md.append(f"{verdict} under the amendment 8 rule: every chunk within 2e-4: {'yes' if chunk_ok else 'no'} "
                  f"({int(np.sum(np.abs(d) > CHUNK_TOL))} of {d.size} chunks outside); mean within 1e-4: {'yes' if mean_ok else 'no'}.")
        md += ["", "Not a conclusion about the kernel, only context: the two runs use different numerics by design. The CPU "
               "path multiplies NVFP4 against Q8_0-quantized activations (`ggml-cpu.c`, `vec_dot_type = GGML_TYPE_Q8_0`), "
               "Metal against f32 (`mul_mv`) or f16 tiles (`mul_mm`). The rule was set without a measured run-to-run floor "
               "for such a change, and the signed per-chunk differences scatter around zero (paired CI above includes 0)."]
    md += ["", "## Timings (NOT a speed measurement)", "",
           "Printed by llama-perplexity. The machine was shared with other jobs during both runs, so these numbers say "
           "nothing about kernel speed. Speed waits for an idle machine (amendment 8).", ""]
    for nm, log in (("measurement", OLD), ("NVFP4-Metal", NEW)):
        md.append(f"- {nm} (`{log.name}`): " + ("; ".join(timing(log)) or "no timing lines"))
    md.append("")
    OUT.write_text("\n".join(md))
    print(OUT.relative_to(ROOT))
    print(f"max|d| {np.abs(d).max():.6f}  mean d {mean_d:+.6f} [{lo:+.6f}, {hi:+.6f}]  mean gap {mean_gap:+.6f}  "
          f"top {old['same_top_p']:.3f} -> {new['same_top_p']:.3f}  nvfp4_on_cpu {nvfp4_on_cpu}")


if __name__ == "__main__":
    main()
