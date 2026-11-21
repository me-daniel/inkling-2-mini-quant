"""analysis/allocate.py: the additive per-projection cost model and the structurally diverse picks."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import allocate  # noqa: E402
import budget  # noqa: E402


def test_block_options_add_the_projection_changes(monkeypatch) -> None:
    monkeypatch.setattr(allocate, "BLOCKS", {"B1": range(0, 1)})
    kl = {"map-B1-3bit-S-P3": 0.010, "map-B1-gu2s-S-P3": 0.013, "map-B1-gu2xxs-S-P3": 0.020, "map-B1-dnq2o-S-P3": 0.016}
    el = {(0, "gu"): 2 * 256 * 64, (0, "dn"): 256 * 64}
    opts = {(gu, dn): (pred, nbytes) for gu, dn, pred, nbytes in allocate.block_options(kl, el)["B1"]}
    assert set(opts) == {(g, d) for g in allocate.GU for d in allocate.DN}
    assert opts[("IQ3_XXS", "IQ4_NL")][0] == pytest.approx(0.010)
    assert opts[("IQ2_S", "IQ4_NL")][0] == pytest.approx(0.013)               # 3bit + (gu2s - 3bit)
    assert opts[("IQ2_XXS", "Q2_0O")][0] == pytest.approx(0.010 + 0.010 + 0.006)  # both changes add
    q = budget.Q
    assert opts[("IQ2_S", "Q2_0O")][1] == budget.tensor_bytes(el[(0, "gu")], q.IQ2_S) + budget.tensor_bytes(el[(0, "dn")], q.Q2_0)


def test_a_missing_map_level_removes_only_that_option(monkeypatch) -> None:
    monkeypatch.setattr(allocate, "BLOCKS", {"B1": range(0, 1)})
    kl = {"map-B1-3bit-S-P3": 0.010, "map-B1-gu2s-S-P3": 0.013, "map-B1-gu2xxs-S-P3": 0.020}  # no dnq2o
    el = {(0, "gu"): 2 * 256 * 64, (0, "dn"): 256 * 64}
    assert {dn for _, dn, _, _ in allocate.block_options(kl, el)["B1"]} == {"IQ4_NL"}


def test_diverse_picks_differ_in_at_least_two_choices(monkeypatch) -> None:
    monkeypatch.setattr(allocate, "BLOCKS", {"B1": None, "B2": None})
    o = lambda gu, dn: (gu, dn, 0.0, 0)  # noqa: E731
    best = [(0.010, 1, "Q8_0", (o("A", "x"), o("A", "x"))),
            (0.011, 1, "Q8_0", (o("B", "x"), o("A", "x"))),   # one change from the first: skipped
            (0.012, 1, "Q6_K", (o("B", "x"), o("A", "x"))),   # two changes: kept
            (0.013, 1, "Q8_0", (o("B", "y"), o("B", "y")))]   # kept
    picks = allocate.diverse_picks(best, n_pick=3)
    assert [p["predicted_kl"] for p in picks] == [0.010, 0.012, 0.013]
    assert picks[1]["choice"] == {"B1": ["B", "x"], "B2": ["A", "x"]} and picks[1]["ne"] == "Q6_K"
