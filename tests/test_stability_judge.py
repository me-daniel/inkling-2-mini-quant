"""scripts/stability.py: judge() applies the docstring thresholds to a (reference, candidate) pair."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("stability", ROOT / "scripts/stability.py")
stability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stability)


def _run(correct=30, capped=0, d4=0.99, rep=1, found=True, valid=5, lang_ok=(True, True)) -> dict:
    return {"gsm8k": {"n": 30, "correct": correct, "capped": capped},
            "long_generations": [{"seed": s, "tokens": 2000, "distinct4": d4, "repeat_run": rep} for s in range(1, 11)],
            "long_context": {"prompt_tokens": 16359, "found": found},
            "tool_calls": {"n": 5, "valid": valid},
            "other_languages": [{"chars": 250, "letter_share": 0.8, "ok": ok} for ok in lang_ok]}


def test_pass_at_the_thresholds() -> None:
    v = stability.judge(_run(), _run(correct=26, capped=2, d4=0.30, rep=5, valid=4))
    assert v["pass"] and all(c["pass"] for c in v["checks"])
    assert len(v["checks"]) == 7


def test_each_breach_fails_its_own_check() -> None:
    ref = _run()
    cases = {"lowest distinct 4-gram ratio over the long generations": _run(d4=0.29),
             "longest repeated-line run over the long generations": _run(rep=6),
             "planted fact found in the long prompt": _run(found=False),
             "valid tool calls": _run(valid=3),
             "non-English replies that pass": _run(lang_ok=(True, False)),
             "GSM8K correct": _run(correct=25),
             "GSM8K replies that hit the cap": _run(capped=3)}
    for name, cand in cases.items():
        v = stability.judge(ref, cand)
        failed = [c["name"] for c in v["checks"] if not c["pass"]]
        assert not v["pass"] and failed == [name], (name, failed)


def test_fact_rule_is_relative_to_the_reference() -> None:
    assert stability.judge(_run(found=False), _run(found=False))["pass"]


def test_one_bad_generation_is_enough() -> None:
    cand = _run()
    cand["long_generations"][3]["distinct4"] = 0.1
    assert not stability.judge(_run(), cand)["pass"]


def test_different_batteries_are_refused() -> None:
    cand = copy.deepcopy(_run())
    cand["long_generations"] = cand["long_generations"][:9]
    with pytest.raises(ValueError):
        stability.judge(_run(), cand)
