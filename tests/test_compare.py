"""analysis/compare.py: the paired chunk bootstrap and the gate rule (CI of cand - baseline below 0, top-1 not worse)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import compare  # noqa: E402


def test_paired_difference_and_interval() -> None:
    b = np.array([0.03, 0.05, 0.04, 0.06])
    d, lo, hi = compare.paired(b - 0.01, b)
    assert (d, lo, hi) == pytest.approx((-0.01, -0.01, -0.01))
    d, lo, hi = compare.paired(np.array([0.01, 0.03, 0.02, 0.04]), b)
    assert lo <= d <= hi and hi < 0


def test_paired_refuses_unequal_chunks() -> None:
    with pytest.raises(ValueError, match="same chunks"):
        compare.paired(np.zeros(16), np.zeros(5))


def run_gate(monkeypatch, capsys, cand_top: float) -> str:
    runs = {}
    for t in ("T1", "T2"):
        runs[f"final-cand-{t}-P3"] = {"chunks": np.array([0.020, 0.022, 0.021, 0.019]), "top": cand_top}
        runs[f"final-base-{t}-P3"] = {"chunks": np.array([0.030, 0.031, 0.029, 0.032]), "top": 95.0}
    monkeypatch.setattr(compare, "load", lambda: runs)
    monkeypatch.setattr(sys, "argv", ["compare.py", "cand", "base"])
    compare.main()
    return capsys.readouterr().out


def test_gate_passes_on_lower_kl_and_equal_top1(monkeypatch, capsys) -> None:
    assert run_gate(monkeypatch, capsys, 95.0).strip().endswith("quality gate: PASS")


def test_gate_fails_when_top1_is_worse(monkeypatch, capsys) -> None:
    assert run_gate(monkeypatch, capsys, 94.9).strip().endswith("quality gate: FAIL")
