"""Amendment 15, component B scoring and H3: per item correct or not, for every model with results/battery/B-*.jsonl.

    python -I analysis/battery_B.py   (needs Docker for B3; caches per-item scores in results/battery/B-scores.json)

- B1: the last number in the reply equals the GSM8K gold answer.
- B2: exactly one tool call, its name equals the ground truth's, its arguments parse, every ground-truth parameter
  takes an allowed value (strings compared as BFCL does: lowercased, spaces and ,./-_*^ removed; numbers by value;
  a parameter whose allowed list holds "" may be left out; an empty allowed list means omitted or []), and no argument outside the function's properties.
- B3: the last ```python block (or the last ``` block) defines the entry point, and on every MBPP+ base and plus input
  it returns what the canonical solution returns (MBPP+'s atol for floats). Run in python:3.12-slim with no
  network, 1 CPU, 1 GB and a 60 s limit per item; any error or timeout is wrong.
- B4: the reply (code fences stripped, first "{" to last "}") parses as JSON and is valid under the schema
  (format checks off).
Capped (finish_reason length) or errored replies are wrong. H3 pools B1, B3 and B4, v2m against v1 (cand): the paired
difference with a one-sided 95% lower bound from the exact mid-p binomial on the discordant pairs (conditional on
their number). Pass: lower bound above -5 points. Regression: upper bound (one-sided 95%) below 0.
"""

import gzip
import importlib.util
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import jsonschema
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/battery"
TASKS = ROOT / "data/raw/battery/tasks"
spec = importlib.util.spec_from_file_location("bB", ROOT / "scripts/battery-B.py")
bB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bB)

HARNESS = r'''
import json, math, sys
d = json.load(open("/w/job.json"))
ns_c, ns_r = {}, {}
exec(d["canonical"], ns_c); exec(d["candidate"], ns_r)
f_c, f_r = ns_c[d["entry"]], ns_r[d["entry"]]
def same(a, b, atol):
    if isinstance(a, float) or isinstance(b, float):
        try: return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=max(atol, 1e-9))
        except Exception: return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, atol) for x, y in zip(a, b))
    return a == b
import copy
for inp in d["inputs"]:
    want = f_c(*copy.deepcopy(inp)); got = f_r(*copy.deepcopy(inp))
    if not same(want, got, d["atol"]):
        print("FAIL"); sys.exit(0)
print("PASS")
'''


def std(s: str) -> str:
    return re.sub(r"[ ,./\-_*^]", "", s.lower())


def ok_value(v, allowed) -> bool:
    for a in allowed:
        if isinstance(a, str) and isinstance(v, str) and std(a) == std(v):
            return True
        if isinstance(a, bool) or isinstance(v, bool):
            if a is v:
                return True
            continue
        if isinstance(a, (int, float)) and isinstance(v, (int, float)) and float(a) == float(v):
            return True
        if isinstance(a, (list, dict)) and a == v:
            return True
    return False


def score_b2(rec, it, gt) -> bool:
    calls = rec.get("tool_calls") or []
    if len(calls) != 1:
        return False
    (fname, params), = gt[0].items()
    fn = calls[0].get("function", {})
    if fn.get("name") != fname:
        return False
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        return False
    props = it["tools"][0]["function"]["parameters"].get("properties", {})
    if any(k not in props for k in args):
        return False
    for k, allowed in params.items():
        if not allowed:  # BFCL writes an expected empty array as an empty list of allowed values
            if args.get(k, []) != []:
                return False
            continue
        if k not in args:
            if "" not in allowed:
                return False
        elif not ok_value(args[k], allowed):
            return False
    return True


def code_block(text: str) -> str | None:
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.S) or re.findall(r"```\w*\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else None


def score_b3(rec, mb) -> bool:
    code = code_block(rec.get("content") or "")
    if not code:
        return False
    job = {"canonical": mb["prompt"].split('"""')[0] + mb["canonical_solution"], "candidate": code, "entry": mb["entry_point"],
           "inputs": mb["base_input"] + mb["plus_input"], "atol": mb["atol"] or 0}
    with tempfile.TemporaryDirectory(dir=ROOT / "models/tmp") as w:  # under /Users: Docker Desktop cannot mount /var/folders
        Path(w, "job.json").write_text(json.dumps(job))
        Path(w, "h.py").write_text(HARNESS)
        try:
            r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--cpus", "1", "--memory", "1g", "-v", f"{w}:/w:ro",
                                "python:3.12-slim", "timeout", "60", "python", "/w/h.py"], capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return False
    return r.stdout.strip().endswith("PASS")


def score_b4(rec, schema) -> bool:
    t = (rec.get("content") or "").strip()
    t = re.sub(r"^```\w*\s*|```$", "", t.strip()).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        return False
    try:
        obj = json.loads(t[i:j + 1])
        jsonschema.validators.validator_for(schema)(schema).validate(obj)
    except Exception:
        return False
    return True


def midp_upper_p(b: int, n: int, alpha: float = 0.05) -> float:
    """Largest p with mid-p P(X <= b | n, p) >= alpha (one-sided upper bound for a binomial proportion)."""
    def cdf_mid(p):
        pmf = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
        return sum(pmf[:b]) + 0.5 * pmf[b]
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf_mid(mid) >= alpha else (lo, mid)
    return lo


def h3(a: list[bool], b: list[bool]) -> tuple[float, float, float, int, int]:
    """a = v2m, b = v1, paired. Returns diff, one-sided 95% lower and upper bounds (points), and the discordant counts."""
    n = len(a)
    win = sum(x and not y for x, y in zip(a, b))
    loss = sum(y and not x for x, y in zip(a, b))
    m = win + loss
    d = 100 * (win - loss) / n
    if m == 0:
        return d, 0.0, 0.0, win, loss
    p_hi = midp_upper_p(win, m)  # upper bound on P(v2m wins | discordant)
    p_lo = 1 - midp_upper_p(loss, m)  # lower bound
    return d, 100 * (2 * p_lo - 1) * m / n, 100 * (2 * p_hi - 1) * m / n, win, loss


def main() -> None:
    its = {it["id"]: it for it in bB.items()}
    gsm = pq.read_table(ROOT / "data/raw/gsm8k/main/test-00000-of-00001.parquet").to_pylist()
    gt = {json.loads(x)["id"]: json.loads(x)["ground_truth"] for x in (TASKS / "bfcl_live_simple_answer.json").read_text().splitlines()}
    mb = {x["task_id"]: x for x in (json.loads(l) for l in gzip.open(TASKS / "MbppPlus.jsonl.gz", "rt"))}
    cache_p = RES / "B-scores.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    for f in sorted(RES.glob("B-*.jsonl")):
        name = f.stem[2:]
        sc = cache.setdefault(name, {})
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            if rec["id"] in sc:
                continue
            it, t = its[rec["id"]], rec["task"]
            if rec.get("finish_reason") != "stop" and not (t == "B2" and rec.get("finish_reason") == "tool_calls"):
                ok = False
            elif t == "B1":
                got = bB_last_number(rec.get("content") or "")
                want = float(gsm[int(rec["id"][3:])]["answer"].split("####")[-1].strip().replace(",", ""))
                ok = got is not None and abs(got - want) < 1e-6
            elif t == "B2":
                ok = score_b2(rec, it, gt[rec["id"]])
            elif t == "B3":
                ok = score_b3(rec, mb[rec["id"]])
            else:
                ok = score_b4(rec, json.loads(it["messages"][0]["content"].split("\n\n", 1)[1]))
            sc[rec["id"]] = bool(ok)
        cache_p.write_text(json.dumps(cache, indent=0, sort_keys=True))
    names = sorted(cache)
    print("| task | n | " + " | ".join(names) + " |")
    print("|---" * (2 + len(names)) + "|")
    for t in ("B1", "B2", "B3", "B4"):
        ids = [i for i in its if its[i]["task"] == t]
        print(f"| {t} | {len(ids)} | " + " | ".join(
            f"{sum(cache[n].get(i, False) for i in ids)}" + ("" if all(i in cache[n] for i in ids) else " (incomplete)") for n in names) + " |")
    if "cand" in cache and "v2m" in cache:
        print("\n| task | v2m right, v1 wrong | v1 right, v2m wrong | capped replies v1 / v2m |\n|---|---|---|---|")
        caps = {n: {json.loads(x)["id"] for x in (RES / f"B-{n}.jsonl").read_text().splitlines()
                    if json.loads(x).get("finish_reason") == "length"} for n in ("cand", "v2m")}
        for t in ("B1", "B2", "B3", "B4"):
            ids = [i for i in its if its[i]["task"] == t]
            w = sum(cache["v2m"].get(i, False) and not cache["cand"].get(i, False) for i in ids)
            l = sum(cache["cand"].get(i, False) and not cache["v2m"].get(i, False) for i in ids)
            print(f"| {t} | {w} | {l} | {len(caps['cand'] & set(ids))} / {len(caps['v2m'] & set(ids))} |")
        ids = [i for i in its if its[i]["task"] in ("B1", "B3", "B4")]
        if all(i in cache["cand"] and i in cache["v2m"] for i in ids):
            d, lo, hi, w, l = h3([cache["v2m"][i] for i in ids], [cache["cand"][i] for i in ids])
            v = "PASS (no regression)" if lo > -5 else ("REGRESSION" if hi < 0 else "inconclusive")
            print(f"\nH3 (B1+B3+B4, n={len(ids)}): v2m - v1 = {d:+.1f} points, one-sided 95% bounds [{lo:+.1f}, {hi:+.1f}], "
                  f"discordant {w} for v2m / {l} for v1 -> {v}")


def bB_last_number(text: str):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return float(nums[-1].replace(",", "").rstrip(".")) if nums else None


if __name__ == "__main__":
    main()
