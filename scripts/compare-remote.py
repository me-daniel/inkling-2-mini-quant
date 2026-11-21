"""Compares tensors of a local GGUF with the same tensors of a remote GGUF, fetching only byte ranges.

    python scripts/compare-remote.py LOCAL.gguf URL HEAD.bin REMOTE_SIZE TENSOR [TENSOR ...]

HEAD.bin holds the first bytes of the remote file (enough for its header). For each tensor it compares the
first, a middle and the last MiB of the data, and reports the remote tensor's type, shape and offset.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

import gguf

MIB = 1 << 20


def remote_reader(head: str, size: int) -> gguf.GGUFReader:
    # a sparse file of the remote's full size with only the header bytes present
    fd, path = tempfile.mkstemp(suffix=".gguf")
    with os.fdopen(fd, "wb") as f, open(head, "rb") as h:
        f.write(h.read())
        f.truncate(size)
    return gguf.GGUFReader(path)


def fetch(url: str, start: int, n: int) -> bytes:
    out = subprocess.run(["curl", "-sfL", "-r", f"{start}-{start + n - 1}", url], check=True, capture_output=True).stdout
    if len(out) != n:
        raise RuntimeError(f"range {start}+{n}: got {len(out)} bytes")
    return out


def main() -> None:
    local_path, url, head, size, *names = sys.argv[1:]
    local = {t.name: t for t in gguf.GGUFReader(local_path).tensors}
    remote = {t.name: t for t in remote_reader(head, int(size)).tensors}
    ok = True
    for name in names:
        lt, rt = local[name], remote[name]
        same_meta = lt.tensor_type == rt.tensor_type and list(lt.shape) == list(rt.shape)
        n = int(rt.n_bytes)
        local_bytes = np.asarray(lt.data).view(np.uint8).reshape(-1)
        spans = [(0, MIB), (n // 2, MIB), (n - MIB, MIB)]
        equal = same_meta and all(
            fetch(url, int(rt.data_offset) + s, k) == local_bytes[s : s + k].tobytes() for s, k in spans
        )
        ok &= equal
        print(f"{name}: {rt.tensor_type.name} {list(rt.shape)} {n / 1e6:.1f} MB -> {'identical' if equal else 'DIFFERENT'} (3 MiB sampled)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
