"""analysis/maptable.py: per-chunk KL from the running means of a llama-perplexity log."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import maptable  # noqa: E402


def log(tmp_path: Path, kl: list[float], skip: int | None = None, nan_se: bool = True) -> Path:
    """A llama-perplexity KL log whose chunk lines hold running means of `kl`; chunk `skip` is left out."""
    lines = [f"kl_divergence: computing over {len(kl)} chunks, n_ctx=4096, batch_size=2048, n_seq=1",
             "chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p"]
    for k in range(1, len(kl) + 1):
        if k == skip:
            continue
        run = sum(kl[:k]) / k
        se = "nan" if nan_se and k <= 2 else "0.00010"
        lines.append(f"{k:4d}       7.5511 ±    0.3921       0.00000 ±    {se:>7s}       {run:.5f} ±    0.00000     0.001 ±  0.000 %    99.000 ±  0.000 %")
    lines += [f"Mean    KLD:   {sum(kl) / len(kl):.6f} ±   0.000100", "Same top p: 99.000 ± 0.100 %"]
    p = tmp_path / "ppl-test-S-P3.log"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_per_chunk_values_from_running_means_with_nan_se(tmp_path: Path) -> None:
    kl = [0.02, 0.03, 0.04, 0.01]
    r = maptable.parse(log(tmp_path, kl))
    assert r["run"] == "test-S-P3"
    assert r["per_chunk"] == pytest.approx(kl, abs=2e-5)
    assert r["same_top_p"] == 99.0


def test_a_missing_chunk_line_is_an_error_not_a_shifted_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk lines"):
        maptable.parse(log(tmp_path, [0.02, 0.03, 0.04, 0.01], skip=3))


def test_log_without_kl_is_skipped(tmp_path: Path) -> None:
    p = tmp_path / "ppl-x.log"
    p.write_text("perplexity only\nFinal estimate: PPL = 7.1\n")
    assert maptable.parse(p) is None
