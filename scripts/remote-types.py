"""Reads only the header of a remote GGUF on Hugging Face (one range request) and prints its expert tensor types and
how many expert tensors are in types the pinned Metal backend cannot multiply (the list in scripts/gguf-types.py).
Used to pick all-Metal comparators before downloading anything.
    python scripts/remote-types.py REPO@REV PATH [REPO@REV PATH ...]
"""
import struct, sys, urllib.request
from collections import Counter
from gguf import GGMLQuantizationType as Q

OFF_METAL = ("NVFP4", "F8_E4M3", "TQ1_0")  # same as scripts/gguf-types.py
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def header(b: bytes):
    o = 0
    def rd(f):
        nonlocal o
        v = struct.unpack_from("<" + f, b, o)[0]; o += struct.calcsize("<" + f); return v
    def rs():
        nonlocal o
        n = rd("Q"); s = b[o:o + n].decode("utf-8", "replace"); o += n; return s
    def skip(t):
        nonlocal o
        if t == 8: rs()
        elif t == 9:
            et, n = rd("I"), rd("Q")
            if et in SZ: o += SZ[et] * n
            else:
                for _ in range(n): skip(et)
        else: o += SZ[t]
    assert b[:4] == b"GGUF", "not a GGUF"
    o = 4; rd("I"); nt = rd("Q"); nkv = rd("Q")
    kv = {}
    for _ in range(nkv):
        k = rs(); t = rd("I")
        if t == 8 and k in ("general.architecture",): kv[k] = rs()
        else: skip(t)
    out = []
    for _ in range(nt):
        n = rs(); nd = rd("I"); [rd("Q") for _ in range(nd)]; t = rd("I"); rd("Q"); out.append((n, t))
    return kv, out


for spec, path in zip(sys.argv[1::2], sys.argv[2::2]):
    repo, rev = spec.split("@")
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{path}"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-50331647"})
    kv, ts = header(urllib.request.urlopen(req, timeout=180).read())
    exps = Counter(f"{n.split('.')[2]} {Q(t).name}" for n, t in ts if "_exps" in n and n.endswith(".weight"))
    off = sum(Q(t).name in OFF_METAL for n, t in ts if "_exps" in n)
    print(f"{repo}@{rev} {path}\n  arch {kv.get('general.architecture')}, expert tensors off Metal: {off}\n  " + ", ".join(f"{k} x{v}" for k, v in sorted(exps.items())))
