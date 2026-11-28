"""Amendment 23: the release's own NVFP4 down experts, repacked into a GGUF that scripts/splice.py can take from.

    python -I scripts/nvfp4-repack.py CHECKPOINT.safetensors OUT.gguf [--layers 0-19] [--check DONOR.gguf]

Per layer L the checkpoint holds model.llm.layers.L.mlp.experts.w2_weight (uint8, two E2M1 codes per byte, low nibble
first; [128, 2304, 576]), w2_weight_scale (E4M3, one per 16 codes; [128, 2304, 72]), w2_weight_amax ([128]) and
w2_weight_outer_scale_denominator ([1]). The value of a weight is E2M1(code) * E4M3(block scale) * amax / denominator.
ggml's NVFP4 block is 4 UE4M3 scales then 32 code bytes per 64 weights (upstream's conversion/base.py _nvfp4_pack);
Inkling's per-expert multiplier goes into blk.L.ffn_down_exps.scale (F32, [128]). Nothing is re-rounded: the codes and
block-scale bits are copied, and the only arithmetic is amax / denominator in float32.

--check compares every written tensor's bytes with the donor's (e.g. Unsloth's UD-IQ3_XXS) and exits 1 on a mismatch.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf


def pack(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """codes uint8 [rows, cols/2] (low nibble first), scales uint8 E4M3 bits [rows, cols/16] -> ggml NVFP4 bytes."""
    rows, nb = scales.shape
    v = codes.reshape(rows, nb, 8)
    vals = np.stack([v & 0x0F, v >> 4], axis=-1).reshape(rows, nb, 16)
    qs = (vals[:, :, :8] | (vals[:, :, 8:] << 4)).astype(np.uint8)
    d = (scales & 0x7F).reshape(rows, nb // 4, 4)
    return np.concatenate([d, qs.reshape(rows, nb // 4, 32)], axis=-1).reshape(rows, nb // 4 * 36)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--layers", default="0-19")
    ap.add_argument("--check")
    a = ap.parse_args()
    lo, hi = map(int, a.layers.split("-"))
    f = safe_open(a.checkpoint, "pt")  # numpy has no float8 dtype; the E4M3 scales are read as raw bytes via torch
    w = gguf.GGUFWriter(a.out, "inkling2_loop")
    out = {}
    for L in range(lo, hi + 1):
        p = f"model.llm.layers.{L}.mlp.experts.w2_weight"
        codes, sc = f.get_tensor(p).numpy(), f.get_tensor(p + "_scale").view(torch.uint8).numpy()
        if np.any(sc & 0x80):
            sys.exit(f"{p}_scale: negative block scale, UE4M3 cannot hold it")
        blocks = np.stack([pack(codes[e], sc[e]) for e in range(codes.shape[0])])
        mult = (f.get_tensor(p + "_amax").float().numpy() / f.get_tensor(p + "_outer_scale_denominator").float().numpy()).astype(np.float32)
        out[f"blk.{L}.ffn_down_exps.weight"] = blocks
        out[f"blk.{L}.ffn_down_exps.scale"] = mult
        w.add_tensor(f"blk.{L}.ffn_down_exps.weight", blocks, raw_dtype=gguf.GGMLQuantizationType.NVFP4)
        w.add_tensor(f"blk.{L}.ffn_down_exps.scale", mult)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    if a.check:
        r = gguf.GGUFReader(a.check)
        donor = {t.name: t for t in r.tensors}
        bad = 0
        for n, arr in out.items():
            t = donor.get(n)
            mine = np.ascontiguousarray(arr).view(np.uint8).ravel()
            theirs = None if t is None else np.asarray(t.data).view(np.uint8).ravel()
            if theirs is None or mine.size != theirs.size or not np.array_equal(mine, theirs):
                bad += 1
                where = "missing" if theirs is None else ("size" if mine.size != theirs.size else int(np.flatnonzero(mine != theirs)[0]))
                print(f"DIFF {n}: {where}")
            else:
                print(f"same {n} ({mine.size:,} bytes)")
        print(f"{len(out) - bad} of {len(out)} tensors byte-identical to {Path(a.check).name}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
