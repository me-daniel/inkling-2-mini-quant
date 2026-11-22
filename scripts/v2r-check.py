"""Checks for the v2r chain (Addendum H amendment 12, scripts/v2r.sh).

    python scripts/v2r-check.py group REFINED BASE GRAMDIR LAYER ...
    python scripts/v2r-check.py final V2R BASE LAYER ...

group: REFINED holds exactly blk.L.ffn_gate_up_exps.weight for the given layers, each with BASE's type, shape and
byte count; d and sub-block scale bits equal BASE's in every expert; experts under 2304 Gram tokens (GRAMDIR counts)
keep BASE's bytes; experts 0, 64 and 127 decode to representable codes (refine-codes.validate); REFINED.json is a
refine-codes log with shrink 0.1, min tokens 2304 and the gain guard on, and at least one refined expert per layer.
final: V2R has BASE's file size, tensor names and types exactly, no expert tensor in a type Metal cannot multiply,
gate_up of layers 0 and 4 byte-identical to BASE's, and gate_up of every given layer different from BASE's.
Prints one OK line per check; exits 1 on the first failure.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

import gguf

_spec = importlib.util.spec_from_file_location("refine_codes", Path(__file__).resolve().parent / "refine-codes.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

OFF_METAL = ("NVFP4", "F8_E4M3", "TQ1_0")  # as scripts/gguf-types.py


def fail(msg: str) -> None:
    sys.exit(f"FAIL: {msg}")


def gu(l: int) -> str:
    return f"blk.{l}.ffn_gate_up_exps.weight"


def group(refined: str, base: str, grams: str, layers: list[int]) -> None:
    rt = {t.name: t for t in gguf.GGUFReader(refined).tensors}
    bt = {t.name: t for t in gguf.GGUFReader(base).tensors}
    if sorted(rt) != sorted(gu(l) for l in layers):
        fail(f"{refined} holds {sorted(rt)}")
    for l in layers:
        r, b = rt[gu(l)], bt[gu(l)]
        if r.tensor_type != b.tensor_type or list(r.shape) != list(b.shape) or int(r.n_bytes) != int(b.n_bytes):
            fail(f"layer {l}: {r.tensor_type.name} {list(r.shape)} {r.n_bytes} vs base {b.tensor_type.name} "
                 f"{list(b.shape)} {b.n_bytes}")
        f = rc.FMTS[r.tensor_type]
        rd, bd = np.asarray(r.data), np.asarray(b.data)
        m = rc.fixed_bits_mask(f, rd.shape[2])
        if not np.array_equal(rd & m, bd & m):
            fail(f"layer {l}: d or scale bits differ from base")
        counts = np.fromfile(f"{grams}/{gu(l)}.gram.counts.i64", dtype=np.int64)
        low = np.nonzero(counts < 2304)[0]
        if not np.array_equal(rd[low], bd[low]):
            fail(f"layer {l}: an expert under 2304 tokens changed")
        for e in (0, 64, 127):
            rc.validate(f, np.asarray(rd[e]))
        nchg = int(np.sum(np.any((rd != bd).reshape(rd.shape[0], -1), axis=1)))
        print(f"OK layer {l} {r.tensor_type.name}: type, shape, bytes, d/scale bits; {len(low)} experts under 2304 "
              f"tokens unchanged; {nchg} experts changed")
    log = json.loads(Path(refined + ".json").read_text(encoding="utf-8"))
    a = log["args"]
    if log.get("script") != "scripts/refine-codes.py" or a["shrink"] != 0.1 or a["min_tokens"] != 2304 or not a["gain_guard"]:
        fail(f"{refined}.json: script {log.get('script')}, args {a}")
    for l in layers:
        if log["layers"][str(l)]["n_refined"] < 1:
            fail(f"layer {l}: no refined expert in the log")
    print(f"OK {refined}.json: guards 9b/9c on; refined experts " +
          ", ".join(f"{l}: {log['layers'][str(l)]['n_refined']}" for l in layers))


def final(v2r: str, base: str, layers: list[int]) -> None:
    if os.path.getsize(v2r) != os.path.getsize(base):
        fail(f"size {os.path.getsize(v2r)} != base {os.path.getsize(base)}")
    print(f"OK size {os.path.getsize(v2r):,} bytes = base")
    vt = {t.name: t for t in gguf.GGUFReader(v2r).tensors}
    bt = {t.name: t for t in gguf.GGUFReader(base).tensors}
    if list(vt) != list(bt):
        fail("tensor names or order differ from base")
    bad = [n for n in vt if vt[n].tensor_type != bt[n].tensor_type]
    if bad:
        fail(f"types differ from base: {bad[:5]}")
    off = [n for n, t in vt.items() if "exps" in n and t.tensor_type.name in OFF_METAL]
    if off:
        fail(f"expert tensors off Metal: {off[:5]}")
    print(f"OK {len(vt)} tensors, names and types equal base's; 0 expert tensors off Metal")
    for l in (0, 4):
        if not np.array_equal(np.asarray(vt[gu(l)].data), np.asarray(bt[gu(l)].data)):
            fail(f"layer {l} gate_up differs from base")
    print("OK gate_up of layers 0 and 4 byte-identical to base")
    same = [l for l in layers if np.array_equal(np.asarray(vt[gu(l)].data), np.asarray(bt[gu(l)].data))]
    if same:
        fail(f"gate_up of layers {same} equal base (not refined)")
    print(f"OK gate_up differs from base in all {len(layers)} refined layers")


def main() -> None:
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "group":
        group(args[0], args[1], args[2], [int(x) for x in args[3:]])
    elif mode == "final":
        final(args[0], args[1], [int(x) for x in args[2:]])
    else:
        fail(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
