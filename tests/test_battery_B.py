import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from battery_B import h3, ok_value, score_b4  # noqa: E402


def test_h3_symmetric_and_bounds_order():
    a = [True] * 90 + [False] * 10
    d, lo, hi, w, l = h3(a, a)
    assert d == 0 and w == l == 0
    v2m = [True] * 10 + [False] * 0 + [True] * 180 + [False] * 10
    v1 = [False] * 10 + [True] * 180 + [True] * 10   # v2m wins 10, loses 10
    d, lo, hi, w, l = h3(v2m, v1)
    assert w == 10 and l == 10 and d == 0 and lo < 0 < hi


def test_values_and_json():
    assert ok_value("San Francisco, CA", ["san francisco ca"]) and ok_value(3, [3.0]) and not ok_value(True, [1])
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    assert score_b4({"content": '```json\n{"a": 2}\n```'}, schema)
    assert not score_b4({"content": '{"a": "x"}'}, schema)
