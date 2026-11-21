"""analysis/addendum_h.py: the registered Addendum H reading rules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import addendum_h as h  # noqa: E402


def test_two_by_two_hybrid_within_two_se_counts_as_equal_down() -> None:
    lines = h.two_by_two(cand=(0.0293, 0.0004), ud=(0.0205, 0.0003), hybrid=(0.0213, 0.0003), mirror=None)
    assert "at least as good" in lines[0]          # +0.0008 <= 2 * 0.000424
    lines = h.two_by_two(cand=(0.0293, 0.0004), ud=(0.0205, 0.0003), hybrid=(0.0240, 0.0003), mirror=None)
    assert "worse than UD's" in lines[0]


def test_two_by_two_mirror_must_beat_cand_by_more_than_two_se() -> None:
    assert "better than ours" in h.two_by_two((0.0293, 0.0004), (0.0205, 0.0003), None, (0.0270, 0.0004))[0]
    assert "not shown" in h.two_by_two((0.0293, 0.0004), (0.0205, 0.0003), None, (0.0285, 0.0004))[0]


def test_all_above_one() -> None:
    assert h.all_above_one([1.1, 1.2, 1.01])
    assert not h.all_above_one([1.1, 0.99])
    assert not h.all_above_one([])
