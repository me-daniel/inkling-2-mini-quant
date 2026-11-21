"""analysis/perpass_report.py: the prediction parser on the real registration, and a full report on synthetic inputs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import perpass_report  # noqa: E402


def test_registered_ranges_parse() -> None:
    pred = perpass_report.predictions(sorted((ROOT / "reviews").glob("*prediction-perpass*.md"))[0])
    assert [len(p) for p in pred] == [2, 1, 1, 1]
    assert all(lo < hi for item in pred for lo, hi in item)


def _res(v: float) -> dict:
    keys = [k for k, _ in perpass_report.STATS]
    one = {**{k: v for k in keys}, "ci95": {k: [v - 0.01, v + 0.01] for k in keys}, "r_chunks": [v, v]}
    return {"fisher": one, "head_euclid": one}


def test_report_builds_and_fills_controls(tmp_path: Path, monkeypatch) -> None:
    root, r = tmp_path, tmp_path / "results/pilot"
    r.mkdir(parents=True)
    (root / "reviews").mkdir()
    (root / "reviews/x-prediction-perpass-y.md").write_text(
        (ROOT / "reviews/2026-11-21-prediction-perpass-gpt-6.2-astra.md").read_text(encoding="utf-8"), encoding="utf-8")
    rows = ["cond\tagainst\tmean_kl\tper_chunk"]
    for i, c in enumerate(perpass_report.CONDS):
        for ref in ("pp-base", "ref-S-P3"):
            rows.append(f"{c}\t{ref}\tdeleted\tx" if c in ("noop", "allslots") else f"{c}\t{ref}\t{0.01 * i}\t{0.01 * i},{0.01 * i}")
    (r / "perpass-kl.tsv").write_text("\n".join(rows) + "\n")
    (r / "perpass-residual.json").write_text(json.dumps(_res(0.5)))
    (r / "noise-ratio.txt").write_text(str({"ratio": 1.2, "ci95": (1.1, 1.3), "kl_tied": 0.02, "kl_indep": 0.0167}) + "\n")
    (r / "noise-t.tsv").write_text("layer\tkind\texpert\tt_2bit\tt_seed1\n6\tffn_down_exps\t0\t0.5\t0.5\n")
    (r / "perpass-parity.txt").write_text("parity: ok\n")
    monkeypatch.setattr(perpass_report, "ROOT", root)
    monkeypatch.setattr(perpass_report, "R", r)
    perpass_report.main()
    md = (r / "perpass.md").read_text()
    assert "Placeholder: filled when `results/pilot/layersplit-residual.json`" in md
    assert "| noop | deleted | deleted |" in md
    assert "| 6.4 tied / independent KL (one draw per pass) | 1.05-1.35 | 1.200 [1.100, 1.300] | inside | inside |" in md
    # scope caveats (audit finding 2) and the Addendum F rule in the placeholder
    assert "Amplitude-specific (Addendum F)" in md and "0.5000 on ffn_down_exps" in md
    assert "B2 only." in md and "One noise draw per pass." in md
    assert "Until this section is filled, no loop-specific reading is licensed." in md
    assert "makes no loop-specific claim and no H1 classification" in md
    # the head norm is labelled a substitute, not "as registered" (finding 3)
    assert "as registered" not in md and "head-restricted substitute" in md
    # KL ordering with CIs (finding 4): conditions i = 2, 3, 4 give p1/p3 = 0.5, p2/p3 = 0.75 on every chunk
    assert "| KL (chunk means) | 0.500 [0.500, 0.500], CI below 1 | 0.750 [0.750, 0.750], CI below 1 |" in md
    # all / sum = 5 / (2 + 3 + 4) on every chunk (finding 10)
    assert "| all / sum of single passes | 0.556 [0.556, 0.556] | 0.556-0.556 |" in md
    assert "Addendum A also predicts independent ~ sum of single-pass effects. This prediction is not tested" in md
    assert not any(w in md.lower() for w in perpass_report.FORBIDDEN)
    (r / "layersplit-residual.json").write_text(json.dumps(_res(0.9)))
    (r / "perpass2-kl.tsv").write_text("cond\tkl_vs_base\nls67\t0.003\n")
    perpass_report.main()
    md = (r / "perpass.md").read_text()
    assert "Placeholder: filled when `results/pilot/layersplit-residual.json`" not in md
    assert "pass split 0.500, this split 0.900; CIs do not overlap." in md
    assert "| ls89 | not run |" in md
    assert "Placeholder: filled when `results/pilot/perpass3bit-residual.json`" in md


def test_derived_and_direction() -> None:
    res = perpass_report.with_derived(_res(0.6))
    assert abs(res["fisher"]["sq_over_sum"] - 0.2) < 1e-12
    assert all(abs(a - b) < 1e-12 for a, b in zip(res["fisher"]["ci95"]["sq_over_sum"], (0.59 / 3, 0.61 / 3)))
    assert perpass_report.direction(1.01, 1.2) == "CI above 1"
    assert perpass_report.direction(0.8, 0.99) == "CI below 1"
    assert perpass_report.direction(0.95, 1.05) == "CI contains 1"


def test_norms_in_skips_missing() -> None:
    old = {"fisher": {}, "head_euclid": {}}
    new = {"fisher": {}, "head_euclid": {}, "full_euclid": {}}
    assert [n for n, _ in perpass_report.norms_in(new)] == ["fisher", "head_euclid", "full_euclid"]
    assert [n for n, _ in perpass_report.norms_in(old, new)] == ["fisher", "head_euclid"]


def test_verdict_applies_rule(tmp_path: Path) -> None:
    split, rung = tmp_path / "split.json", tmp_path / "rung.json"
    main = perpass_report.with_derived(_res(0.5))
    assert "Not yet decidable" in "\n".join(perpass_report.verdict(main, split, rung))
    split.write_text(json.dumps(_res(0.9)))
    text = "\n".join(perpass_report.verdict(main, split, rung))
    assert "The controls license one loop-specific claim" in text and "3-bit rung" not in text
    assert "the two statistics move in opposite directions" in text  # r 0.5 < 0.9, but norm ratio 0.5 farther from 1
    far = _res(0.9)
    far["fisher"] = {**far["fisher"], "norm_ratio": 0.3, "ci95": {**far["fisher"]["ci95"], "norm_ratio": [0.29, 0.31]}}
    split.write_text(json.dumps(far))
    assert "the pass split is nearer additive than the layer split" in "\n".join(perpass_report.verdict(main, split, rung))
    split.write_text(json.dumps(_res(0.505)))  # CIs overlap, points inside each other's CI
    rung.write_text(json.dumps(_res(0.7)))
    text = "\n".join(perpass_report.verdict(main, split, rung))
    assert "license no loop-specific claim: neither r nor the norm ratio clears the rule." in text
    assert "Fisher r and norm ratio differ from the 2-bit pass split beyond both CIs" in text
    one = _res(0.9)
    one["fisher"] = {**one["fisher"], "norm_ratio": 0.5, "ci95": {**one["fisher"]["ci95"], "norm_ratio": [0.49, 0.51]}}
    split.write_text(json.dumps(one))
    text = "\n".join(perpass_report.verdict(main, split, rung))
    assert "only the r clears the rule, and the rule needs both." in text
    # disjoint CIs but a point inside the other's CI cannot happen; points outside but CIs overlapping can
    near = _res(0.515)
    split.write_text(json.dumps(near))
    text = "\n".join(perpass_report.verdict(main, split, rung))
    assert "disagree here, and the stricter one is applied" in text and "license no loop-specific claim" in text
    assert not any(w in text.lower() for w in perpass_report.FORBIDDEN)
