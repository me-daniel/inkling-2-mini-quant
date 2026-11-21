"""splice.py on a tiny GGUF pair built with gguf-py: --add appends donor tensors byte for byte, refuses names that
exist, and leaves --take unchanged."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gguf

SPLICE = Path(__file__).resolve().parent.parent / "scripts" / "splice.py"
N_EXP, ROWS, COLS = 4, 3, 32


def write_gguf(path: Path, tensors: dict[str, np.ndarray]) -> None:
    w = gguf.GGUFWriter(str(path), "inkling2_loop")
    w.add_uint32("inkling2_loop.loop.count", 3)
    for name, a in tensors.items():
        w.add_tensor(name, a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def down(seed: int, dtype=np.float32) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((N_EXP, ROWS, COLS)).astype(dtype)


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    base = {"token_embd.weight": down(0)[0]}
    donor = {"token_embd.weight": down(100)[0]}
    for l in range(3):
        base[f"blk.{l}.ffn_down_exps.weight"] = down(l + 1)
        base[f"blk.{l}.attn_q.weight"] = down(l + 10, np.float16)[0]
        donor[f"blk.{l}.ffn_down_exps.weight"] = down(l + 101)
        donor[f"blk.{l}.attn_q.weight"] = down(l + 110, np.float16)[0]
        donor[f"blk.{l}.ffn_down_exps.scale"] = np.random.default_rng(l + 200).random(N_EXP).astype(np.float32)
    write_gguf(tmp_path / "base.gguf", base)
    write_gguf(tmp_path / "donor.gguf", donor)
    return tmp_path / "base.gguf", tmp_path / "donor.gguf"


def splice(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(SPLICE), *map(str, args)], capture_output=True, text=True)


def tensors(path: Path) -> dict[str, gguf.ReaderTensor]:
    return {t.name: t for t in gguf.GGUFReader(str(path)).tensors}


TAKE = r"^blk\.[01]\.ffn_down_exps\.weight$"
ADD = r"^blk\.[01]\.ffn_down_exps\.scale$"


def test_no_options_is_byte_identical(pair, tmp_path):
    base, _ = pair
    assert splice(base, tmp_path / "out.gguf", "--verify").returncode == 0
    assert (tmp_path / "out.gguf").read_bytes() == base.read_bytes()


def test_take_and_add(pair, tmp_path):
    base, donor = pair
    take_only, both = tmp_path / "take.gguf", tmp_path / "both.gguf"
    r = splice(base, take_only, "--take", donor, TAKE, "--verify")
    assert r.returncode == 0, r.stderr
    r = splice(base, both, "--take", donor, TAKE, "--add", donor, ADD, "--verify")
    assert r.returncode == 0, r.stderr

    b, d, t, o = tensors(base), tensors(donor), tensors(take_only), tensors(both)
    # --take: same names in base order, taken tensors from the donor, the rest from base
    assert list(t) == list(b)
    for name in b:
        src = d[name] if name in ("blk.0.ffn_down_exps.weight", "blk.1.ffn_down_exps.weight") else b[name]
        assert t[name].tensor_type == src.tensor_type and np.array_equal(t[name].data, src.data), name
    # --add: the --take file plus the two matched scales appended in donor order, byte for byte; blk.2's is not added
    assert list(o) == list(b) + ["blk.0.ffn_down_exps.scale", "blk.1.ffn_down_exps.scale"]
    for name in b:
        assert o[name].tensor_type == t[name].tensor_type and np.array_equal(o[name].data, t[name].data), name
    for name in ("blk.0.ffn_down_exps.scale", "blk.1.ffn_down_exps.scale"):
        assert o[name].tensor_type == gguf.GGMLQuantizationType.F32
        assert list(o[name].shape) == [N_EXP]
        assert o[name].data.tobytes() == d[name].data.tobytes()
    # metadata copied from base
    assert gguf.GGUFReader(str(both)).fields["inkling2_loop.loop.count"].contents() == 3


def test_add_refuses_existing_names(pair, tmp_path):
    base, donor = pair
    r = splice(base, tmp_path / "out.gguf", "--add", donor, r"^blk\.0\.ffn_down_exps\.")
    assert r.returncode != 0 and "already exists" in r.stderr
    r = splice(base, tmp_path / "out2.gguf", "--add", donor, ADD, "--add", donor, r"^blk\.1\.ffn_down_exps\.scale$")
    assert r.returncode != 0 and "already exists" in r.stderr


def test_add_refuses_empty_match(pair, tmp_path):
    base, donor = pair
    r = splice(base, tmp_path / "out.gguf", "--add", donor, r"^blk\.9\.ffn_down_exps\.scale$")
    assert r.returncode != 0 and "matches no tensor" in r.stderr
