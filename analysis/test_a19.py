import numpy as np

from a19 import paired_delta, pick, subsets


def test_paired_delta_of_identical_runs_is_zero():
    a = np.linspace(0.01, 0.05, 16)
    assert paired_delta(a, a) == (0.0, 0.0, 0.0)


def test_s6_and_s7_never_combine():
    s = {"s6": (-0.001, 2, 2), "s7": (-0.002, 27, 27), "s1": (0.001, -200, -200)}
    assert not any({"s6", "s7"} <= set(c) for *_, c in subsets(s))


def test_rules_respect_bytes():
    # s4-like change: +232 MB file, 0 Metal. Alone it exceeds UD's file size (v2c + 232 MB), so even the Metal
    # rule needs a cut beside it; the file rule cannot take it with this cut either (net +20 MB).
    s = {"s4": (-0.002, 232_000_000, 0), "s1": (0.0005, -212_000_000, -212_000_000)}
    file_best = pick(subsets(s), "file", k=10)
    metal_best = pick(subsets(s), "metal", k=10)
    assert [c for *_, c in file_best] == [("s1",)]
    assert metal_best[0][3] == ("s1", "s4")
