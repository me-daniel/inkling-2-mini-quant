"""CHANGELOG Addendum H amendment 5, items 1 and 2: CPU audit of the source.

Item 1: decode the release NVFP4 expert weights with an independent numpy
decoder (value = e2m1 * fp8_e4m3(block scale) * amax / outer_scale_denominator)
and compare them with our BF16 conversion and with Unsloth's UD-IQ3_XXS
(NVFP4 down times .scale, IQ gate_up times .scale).
Item 2: shapes, per-expert counts and loop-visit counts of the ggml-org imatrix.

Reads only sampled experts (a few MB each); memory stays far below 1 GB.
Usage: .venv/bin/python -I scripts/source-audit.py  -> results/pilot/source-audit.md
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor/llama.cpp/gguf-py"))
import gguf  # noqa: E402
from gguf import quants  # noqa: E402

REL = ROOT / "models/Inkling-2-Mini/model-00001-of-00001.safetensors"
BF16 = ROOT / "models/Inkling-2-Mini-BF16.gguf"
UD = ROOT / "models/UD-IQ3_XXS.gguf"
IMAT = ROOT / "models/ggml-org/imatrix.gguf"
OUT = ROOT / "results/pilot/source-audit.md"
LAYERS = [0, 6, 12, 19]
EXPERTS = [0, 37, 64, 127]
LOOPED = range(6, 18)


# ---------- independent NVFP4 decoder ----------

def e2m1_table():
    """16 E2M1 codes: bit3 sign, bits2-1 exponent (bias 1), bit0 mantissa."""
    out = np.zeros(16, np.float64)
    for c in range(16):
        s, e, m = (c >> 3) & 1, (c >> 1) & 3, c & 1
        v = (m / 2) if e == 0 else (1 + m / 2) * 2.0 ** (e - 1)  # e=0: subnormal 0 or 0.5
        out[c] = -v if s else v
    return out


def fp8_e4m3fn_table():
    """256 E4M3FN codes: bit7 sign, 4-bit exponent (bias 7), 3-bit mantissa; 0x7F/0xFF NaN."""
    out = np.zeros(256, np.float64)
    for c in range(256):
        s, e, m = (c >> 7) & 1, (c >> 3) & 15, c & 7
        v = (m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        if e == 15 and m == 7:
            v = np.nan
        out[c] = -v if s else v
    return out


# candidate maps from the release w13 row order [2304 out, 2304 in] to our fused gate_up rows
LAYOUTS = {
    "identity": lambda w: w,
    "halves swapped": lambda w: np.concatenate([w[1152:], w[:1152]]),
    "transposed": lambda w: w.T,
    "rows de-interleaved (even rows -> 0-1151, odd -> 1152-2303)": lambda w: np.concatenate([w[0::2], w[1::2]]),
    "rows de-interleaved (odd rows -> 0-1151, even -> 1152-2303)": lambda w: np.concatenate([w[1::2], w[0::2]]),
}

E2M1 = e2m1_table()
FP8 = fp8_e4m3fn_table()


def decode_nvfp4(packed, scales, gscale, low_first=True):
    """packed uint8 [R, C/2], scales uint8 [R, C/16], gscale float32 -> float32 [R, C].

    Product order: (e2m1 * fp8) is exact in float32 (<= 6 significant bits); one
    float32 multiply by the global scale. Returns float32 values.
    """
    lo, hi = packed & 0x0F, packed >> 4
    codes = np.stack([lo, hi] if low_first else [hi, lo], axis=-1).reshape(packed.shape[0], -1)
    q = E2M1[codes].astype(np.float32)  # [R, C]
    s = FP8[scales].astype(np.float32)  # [R, C/16]
    assert not np.isnan(s).any()
    w = (q.reshape(q.shape[0], -1, 16) * s[:, :, None]).reshape(q.shape)
    return w * np.float32(gscale)


def to_bf16_bits(x):
    """float32 -> bf16 bit pattern, round to nearest even."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


# ---------- readers ----------

class SafeTensors:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.base = 8 + n

    def info(self, k):
        return self.h[k]["dtype"], self.h[k]["shape"]

    def read(self, k, expert=None):
        v = self.h[k]
        a, b = v["data_offsets"]
        dt = {"U8": np.uint8, "F8_E4M3": np.uint8, "F32": np.float32}[v["dtype"]]
        shape = v["shape"]
        if expert is None:
            self.f.seek(self.base + a)
            return np.frombuffer(self.f.read(b - a), dt).reshape(shape)
        per = (b - a) // shape[0]
        self.f.seek(self.base + a + expert * per)
        return np.frombuffer(self.f.read(per), dt).reshape(shape[1:])


def gg_tensors(path):
    r = gguf.GGUFReader(path)
    return r, {t.name: t for t in r.tensors}


def rel_l2(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def fmt(x, p=3):
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if x == 0:
        return "0"
    return f"{x:.{p}e}" if abs(x) < 1e-2 or abs(x) >= 1e5 else f"{x:.{p + 1}g}"


# ---------- item 1 ----------

def item1(md):
    st = SafeTensors(REL)
    _, bf = gg_tensors(BF16)
    _, ud = gg_tensors(UD)
    L = lambda l, w, s="": f"model.llm.layers.{l}.mlp.experts.{w}_weight{s}"  # noqa: E731

    md.append("## Item 1: source audit\n")
    md.append("### Release NVFP4 layout (from the safetensors header and hf_quant_config.json)\n")
    q = json.loads((REL.parent / "hf_quant_config.json").read_text())
    grp = [g for g in q["model"]["parameter_groups"] if g["compute_dtype"] == "nvfp4"][0]
    nv_layers = sorted({int(t.split(".")[3]) for t in grp["tensors"]})
    md.append(f"- NVFP4 group: {len(grp['tensors'])} tensors, layers {nv_layers[0]}-{nv_layers[-1]}; "
              f"weight_block_shape {grp['weight_block_shape']}, weight_scale_format `{grp['weight_scale_format']}`, "
              f"weight_transform `{grp['weight_transform']}`, scale_selection_rule "
              f"`{grp['required_activation_quantization']['scale_selection_rule']}`.")
    for w in ["w13", "w2"]:
        for s in ["", "_scale", "_amax", "_outer_scale_denominator"]:
            dt, sh = st.info(L(0, w, s))
            md.append(f"- `{L(0, w, s)}`: {dt} {sh}")
    dens = sorted({float(st.read(L(l, w, "_outer_scale_denominator"))[0]) for l in nv_layers for w in ["w13", "w2"]})
    md.append(f"- outer_scale_denominator over all {len(nv_layers)} NVFP4 layers and both tensors: {dens} "
              f"(6 x 448 = {6 * 448}). Global scale per expert = amax[e] / denominator.")
    md.append("- Decoder: value = e2m1(nibble) * fp8_e4m3fn(block scale, one per 16 consecutive values along the "
              "input dim) * amax[e] / denominator. Rows are output channels: w13 has 2304 rows of 2304 inputs "
              "(gate and up fused in one tensor; row order found below), w2 has 2304 rows of 1152 inputs. "
              "Experts on axis 0.\n")

    # nibble order and layout, decided on layer 0 expert 0 against our BF16
    e, l = 0, 0
    g = float(st.read(L(l, "w13", "_amax"))[e] / st.read(L(l, "w13", "_outer_scale_denominator"))[0])
    ours = bf16_to_f32(bf[f"blk.{l}.ffn_gate_up_exps.weight"].data[e].view(np.uint16))
    rows = []
    for low in [True, False]:
        w = decode_nvfp4(st.read(L(l, "w13"), e), st.read(L(l, "w13", "_scale"), e), g, low)
        for name, f in LAYOUTS.items():
            c = f(w)
            rows.append((("low nibble first" if low else "high nibble first"), name, rel_l2(c, ours)))
    md.append("### Layout match (layer 0, expert 0, gate_up; relative L2 of our BF16 against each candidate)\n")
    md.append("| nibble order | layout | rel L2 |\n|---|---|---|")
    for a, b, r in rows:
        md.append(f"| {a} | {b} | {fmt(r)} |")
    best = min(rows, key=lambda r: r[2])
    low_first = best[0] == "low nibble first"
    gu_layout = LAYOUTS[best[1]]
    md.append(f"\nBest: {best[0]}, {best[1]}. Used below for gate_up; down is compared as stored (identity).\n")

    # BF16 against release, element by element
    md.append("### Our BF16 conversion against the decoded release\n")
    md.append("Release decoded in float32 then rounded to BF16 (round to nearest even). "
              "`bf16 mismatches` counts elements whose BF16 bits differ; `max |d|` and `rel L2` compare our BF16 "
              "with the unrounded float32 decode.\n")
    md.append("| layer | expert | tensor | elements | max abs value | max \\|d\\| | rel L2 | bf16 mismatches |")
    md.append("|---|---|---|---|---|---|---|---|")
    tot_mis = tot_el = 0
    worst = 0.0
    for l in LAYERS:
        for e in EXPERTS:
            for w, gname in [("w13", "ffn_gate_up_exps"), ("w2", "ffn_down_exps")]:
                g = np.float32(st.read(L(l, w, "_amax"))[e]) / np.float32(st.read(L(l, w, "_outer_scale_denominator"))[0])
                dec = decode_nvfp4(st.read(L(l, w), e), st.read(L(l, w, "_scale"), e), g, low_first)
                if w == "w13":
                    dec = np.ascontiguousarray(gu_layout(dec))
                bits = bf[f"blk.{l}.{gname}.weight"].data[e].view(np.uint16)
                assert bits.shape == dec.shape, (bits.shape, dec.shape)
                ours = bf16_to_f32(bits)
                mis = int((to_bf16_bits(dec) != bits).sum())
                d = float(np.abs(ours - dec).max())
                r = rel_l2(ours, dec)
                worst = max(worst, r)
                tot_mis += mis
                tot_el += dec.size
                md.append(f"| {l} | {e} | {gname} | {fmt(dec.size)} | {fmt(float(np.abs(dec).max()))} | "
                          f"{fmt(d)} | {fmt(r)} | {fmt(mis)} |")
    md.append(f"\nTotal: {fmt(tot_mis)} BF16 mismatches in {fmt(tot_el)} elements; worst rel L2 {fmt(worst)}.\n")
    res = {"bf16_mis": tot_mis, "bf16_el": tot_el, "bf16_worst": worst, "layout": best[1], "nibble": best[0]}

    # UD down (NVFP4 * .scale) against release; UD .scale against amax/den
    md.append("### UD-IQ3_XXS down (NVFP4 dequantized by gguf-py, times .scale) against the decoded release\n")
    md.append("`exact` counts float32 elements equal to the release decode; `.scale ratio` is UD's "
              "`ffn_down_exps.scale[e]` over the release amax[e] / denominator.\n")
    md.append("| layer | expert | UD type | .scale ratio | max \\|d\\| | rel L2 | exact | elements |")
    md.append("|---|---|---|---|---|---|---|---|")
    ud_worst, ud_exact, ud_el = 0.0, 0, 0
    ratios = []
    for l in LAYERS:
        t = ud[f"blk.{l}.ffn_down_exps.weight"]
        sc = ud[f"blk.{l}.ffn_down_exps.scale"].data
        for e in EXPERTS:
            g = np.float32(st.read(L(l, "w2", "_amax"))[e]) / np.float32(st.read(L(l, "w2", "_outer_scale_denominator"))[0])
            rel = decode_nvfp4(st.read(L(l, "w2"), e), st.read(L(l, "w2", "_scale"), e), g, low_first)
            u = quants.dequantize(np.asarray(t.data[e]), t.tensor_type).astype(np.float32) * np.float32(sc[e])
            assert u.shape == rel.shape, (u.shape, rel.shape)
            ratio = float(sc[e] / g)
            ratios.append(ratio)
            ex = int((u == rel).sum())
            r = rel_l2(u, rel)
            ud_worst = max(ud_worst, r)
            ud_exact += ex
            ud_el += u.size
            md.append(f"| {l} | {e} | {t.tensor_type.name} | {ratio:.6f} | {fmt(float(np.abs(u - rel).max()))} | "
                      f"{fmt(r)} | {fmt(ex)} | {fmt(u.size)} |")
    md.append(f"\nTotal: {fmt(ud_exact)} of {fmt(ud_el)} elements exactly equal; worst rel L2 {fmt(ud_worst)}.\n")
    res.update(ud_exact=ud_exact, ud_el=ud_el, ud_worst=ud_worst)

    # UD gate_up (IQ * .scale): half order against ours, error level against release
    md.append("### UD-IQ3_XXS gate_up (dequantized by gguf-py, times .scale): half order and error\n")
    md.append("Row blocks: A = rows 0-1151, B = rows 1152-2303 of the fused tensor. `UD A vs ours A` is the "
              "relative L2 of UD's block A against our BF16 block A; `UD A vs ours B` crosses the halves. "
              "Same order if the matched pair is far below the crossed pair. `.scale ratio` as above for "
              "`ffn_gate_up_exps.scale`.\n")
    md.append("| layer | expert | UD type | .scale ratio | UD A vs ours A | UD B vs ours B | UD A vs ours B | UD B vs ours A | rel L2 whole |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    same = 0
    n = 0
    gu_rel = []
    for l in LAYERS:
        t = ud[f"blk.{l}.ffn_gate_up_exps.weight"]
        sc = ud[f"blk.{l}.ffn_gate_up_exps.scale"].data
        for e in EXPERTS:
            g = np.float32(st.read(L(l, "w13", "_amax"))[e]) / np.float32(st.read(L(l, "w13", "_outer_scale_denominator"))[0])
            ours = bf16_to_f32(bf[f"blk.{l}.ffn_gate_up_exps.weight"].data[e].view(np.uint16))
            u = quants.dequantize(np.asarray(t.data[e]), t.tensor_type).astype(np.float32) * np.float32(sc[e])
            assert u.shape == ours.shape
            A, B = slice(0, 1152), slice(1152, 2304)
            aa, bb = rel_l2(u[A], ours[A]), rel_l2(u[B], ours[B])
            ab, ba = rel_l2(u[A], ours[B]), rel_l2(u[B], ours[A])
            whole = rel_l2(u, ours)
            gu_rel.append(whole)
            same += int(max(aa, bb) < min(ab, ba))
            n += 1
            ratios.append(float(sc[e] / g))
            md.append(f"| {l} | {e} | {t.tensor_type.name} | {float(sc[e] / g):.6f} | {fmt(aa)} | {fmt(bb)} | "
                      f"{fmt(ab)} | {fmt(ba)} | {fmt(whole)} |")
    md.append(f"\nSame half order in {same} of {n} sampled experts. UD gate_up rel L2 against ours (= release, "
              f"see above): {fmt(min(gu_rel))} to {fmt(max(gu_rel))}. All {len(ratios)} sampled `.scale` ratios "
              f"lie in [{min(ratios):.6f}, {max(ratios):.6f}].\n")
    res.update(same=same, n=n, ratio_lo=min(ratios), ratio_hi=max(ratios), gu_lo=min(gu_rel), gu_hi=max(gu_rel))
    return res


# ---------- item 2 ----------

def item2(md):
    r, t = gg_tensors(IMAT)
    meta = {}
    for k in ["imatrix.datasets", "imatrix.chunk_count", "imatrix.chunk_size", "imatrix.stats_schema"]:
        meta[k] = r.fields[k].contents()
    md.append("## Item 2: imatrix audit (models/ggml-org/imatrix.gguf)\n")
    for k, v in meta.items():
        md.append(f"- `{k}`: {v}")
    n_entries = len({n.rsplit('.', 1)[0] for n in t if n.endswith('.in_sum2')})
    md.append(f"- {len(t)} tensors, {n_entries} entries (in_sum2 per entry).")
    tokens = meta["imatrix.chunk_count"] * meta["imatrix.chunk_size"]
    md.append(f"- chunk_count x chunk_size = {fmt(tokens)} tokens; with top-8 routing one visit per token gives "
              f"{fmt(tokens * 8)} expert assignments per layer.\n")

    md.append("| tensor | in_sum2 shape | counts shape | count min | median | max | experts with 0 | total count | total / (tokens x 8) | in_sum2 columns distinct |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    totals = {}
    for l in LAYERS:
        for name in ["ffn_gate_up_exps", "ffn_down_exps"]:
            base = f"blk.{l}.{name}.weight"
            s2, c = t[base + ".in_sum2"], t[base + ".counts"]
            cnt = np.asarray(c.data, np.float64).reshape(-1)
            s2d = np.asarray(s2.data, np.float64)  # [n_expert, n_in]
            distinct = len({row.tobytes() for row in s2d})
            totals[(l, name)] = cnt.sum()
            md.append(f"| {base} | {list(map(int, s2.shape))} | {list(map(int, c.shape))} | {fmt(int(cnt.min()))} | "
                      f"{fmt(float(np.median(cnt)))} | {fmt(int(cnt.max()))} | {int((cnt == 0).sum())} | "
                      f"{fmt(int(cnt.sum()))} | {cnt.sum() / (tokens * 8):.4f} | {distinct} of {s2d.shape[0]} |")
    md.append("\nTotal-count ratios, looped over non-looped (3.0 expected if all three visits are accumulated, "
              "1.0 if only one):\n")
    md.append("| tensor | 6 / 0 | 12 / 0 | 6 / 19 | 12 / 19 |\n|---|---|---|---|---|")
    rat = {}
    for name in ["ffn_gate_up_exps", "ffn_down_exps"]:
        v = [totals[(a, name)] / totals[(b, name)] for a, b in [(6, 0), (12, 0), (6, 19), (12, 19)]]
        rat[name] = v
        md.append(f"| {name} | " + " | ".join(f"{x:.4f}" for x in v) + " |")
    # gate_up and down of the same layer should see the same expert assignments
    same_counts = all(np.array_equal(np.asarray(t[f"blk.{l}.ffn_gate_up_exps.weight.counts"].data),
                                     np.asarray(t[f"blk.{l}.ffn_down_exps.weight.counts"].data)) for l in LAYERS)
    md.append(f"\nPer-expert counts of gate_up and down identical within each sampled layer: {same_counts}.\n")
    nl_tok = totals[(0, "ffn_gate_up_exps")] / 8
    nl_chunks = nl_tok / meta["imatrix.chunk_size"]
    md.append(f"Tokens implied by the non-looped layer-0 total (total / 8): {fmt(int(nl_tok))} = "
              f"{nl_chunks:g} chunks of {meta['imatrix.chunk_size']}; stored chunk_count / implied chunks = "
              f"{meta['imatrix.chunk_count'] / nl_chunks:.4f}.\n")
    return {"rat": rat, "tokens": tokens, "totals": totals, "same_counts": same_counts, "meta": meta,
            "nl_chunks": nl_chunks}


def main():
    md = ["# Source audit (Addendum H amendment 5, items 1 and 2)\n",
          "Generated by `scripts/source-audit.py`. Layers " + ", ".join(map(str, LAYERS)) +
          "; experts " + ", ".join(map(str, EXPERTS)) + ". Release: `models/Inkling-2-Mini` safetensors; ours: "
          "`models/Inkling-2-Mini-BF16.gguf`; UD: `models/UD-IQ3_XXS.gguf`.\n"]
    body = []
    a = item1(body)
    b = item2(body)
    rg, rd = b["rat"]["ffn_gate_up_exps"], b["rat"]["ffn_down_exps"]
    reading = [
        "## Reading\n",
        (f"Our BF16 conversion is {'faithful' if a['bf16_mis'] == 0 else 'not bit-faithful'} to the release: "
         f"{fmt(a['bf16_mis'])} of {fmt(a['bf16_el'])} sampled elements differ from the independently decoded "
         f"release after BF16 rounding (worst rel L2 {fmt(a['bf16_worst'])}), with experts in release order, "
         f"{a['nibble']}, and gate_up matched by: {a['layout']}. "),
        (f"UD's NVFP4 down times its `.scale` reproduces the release "
         f"{'exactly' if a['ud_exact'] == a['ud_el'] else 'not exactly'} ({fmt(a['ud_exact'])} of {fmt(a['ud_el'])} "
         f"elements equal in float32, worst rel L2 {fmt(a['ud_worst'])}), and UD's `.scale` equals the release "
         f"amax / denominator to within a ratio of [{a['ratio_lo']:.6f}, {a['ratio_hi']:.6f}]. "),
        (f"UD's gate_up halves are in the same order as ours in {a['same']} of {a['n']} sampled experts. "),
        (f"The imatrix is expert-indexed (per-expert in_sum2 and counts, n_expert = 128 rows), and its loop "
         f"counts give looped-over-non-looped total ratios of {min(rg + rd):.4f} to {max(rg + rd):.4f}, "
         f"against 3.0 if every visit were accumulated and 1.0 if one. The non-looped totals imply "
         f"{b['nl_chunks']:g} chunks of tokens, so the stored chunk_count {b['meta']['imatrix.chunk_count']} is "
         f"{b['meta']['imatrix.chunk_count'] / b['nl_chunks']:.4f} times the chunks actually seen."),
    ]
    OUT.write_text("\n".join(md + body + reading) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
