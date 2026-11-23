"""Amendment 15, component B: task replies of one model through llama-server, logged raw for scoring later.

    python -I scripts/battery-B.py NAME MODEL.gguf   (run under the GPU lock by scripts/battery-B.sh)

Items are drawn once with numpy's default_rng(0), independent of any model:
- B1 math: 80 GSM8K test items. Prompt: the question plus "Give the final answer as a number on the last line." (the
  stability battery's wording).
- B2 tool calls: 100 BFCL v3 live_simple items, the item's functions passed as OpenAI tools (BFCL's "dict" type
  written as "object", "float" as "number", "tuple" as "array", "any" left untyped).
- B3 code: 60 MBPP+ v0.2.0 items. Prompt: the item's prompt plus "Write the Python function `NAME`. Reply with one
  ```python code block that defines it."
- B4 structured output: 60 JSONSchemaBench Github_medium schemas whose schema is valid and under 4,000 characters.
  Prompt: "Reply with one JSON object that is valid under this JSON Schema, and nothing else." plus the schema.
All: temperature 0, seed 0, reasoning_effort low, max_tokens 2,048, one slot, P = 3. Writes
results/battery/B-NAME.jsonl (one line per item: id, task, finish_reason, content, reasoning_content, tool_calls,
usage); resumes on an existing file.
"""

import gzip
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import jsonschema
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "vendor/llama.cpp/build/bin"
RAW = ROOT / "data/raw"
TASKS = RAW / "battery/tasks"
PORT = 8093
URL = f"http://127.0.0.1:{PORT}"
SIZES = {"B1": 80, "B2": 100, "B3": 60, "B4": 60}
TYPES = {"dict": "object", "float": "number", "tuple": "array"}


def draw(n_items: int, k: int, salt: int) -> list[int]:
    return sorted(np.random.default_rng([0, salt]).permutation(n_items)[:k].tolist())


def bfcl_schema(p):
    if isinstance(p, dict):
        out = {k: bfcl_schema(v) for k, v in p.items() if not (k == "type" and v == "any")}
        if isinstance(out.get("type"), str):
            out["type"] = TYPES.get(out["type"], out["type"])
        return out
    if isinstance(p, list):
        return [bfcl_schema(x) for x in p]
    return p


def items() -> list[dict]:
    out = []
    gsm = pq.read_table(RAW / "gsm8k/main/test-00000-of-00001.parquet").to_pylist()
    for i in draw(len(gsm), SIZES["B1"], 1):
        out.append({"id": f"B1-{i}", "task": "B1", "messages": [{"role": "user", "content": gsm[i]["question"]
                    + "\nGive the final answer as a number on the last line."}]})
    bf = [json.loads(x) for x in (TASKS / "bfcl_live_simple.json").read_text().splitlines()]
    for i in draw(len(bf), SIZES["B2"], 2):
        it = bf[i]
        tools = [{"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                  "parameters": bfcl_schema(f["parameters"])}} for f in it["function"]]
        out.append({"id": it["id"], "task": "B2", "messages": it["question"][0], "tools": tools})
    mb = [json.loads(x) for x in gzip.open(TASKS / "MbppPlus.jsonl.gz", "rt")]
    for i in draw(len(mb), SIZES["B3"], 3):
        it = mb[i]
        out.append({"id": it["task_id"], "task": "B3", "messages": [{"role": "user", "content": it["prompt"].strip()
                    + f"\nWrite the Python function `{it['entry_point']}`. Reply with one ```python code block that defines it."}]})
    js = []
    for r in pq.read_table(TASKS / "jsb_github_medium.parquet").to_pylist():
        if len(r["json_schema"]) >= 4000:
            continue
        try:
            jsonschema.validators.validator_for(json.loads(r["json_schema"])).check_schema(json.loads(r["json_schema"]))
        except Exception:
            continue
        js.append(r)
    for i in draw(len(js), SIZES["B4"], 4):
        out.append({"id": f"B4-{js[i]['unique_id']}", "task": "B4", "messages": [{"role": "user", "content":
                    "Reply with one JSON object that is valid under this JSON Schema, and nothing else.\n\n" + js[i]["json_schema"]}]})
    return out


def post(body: dict) -> dict:
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def main() -> None:
    name, model = sys.argv[1], sys.argv[2]
    out_path = ROOT / f"results/battery/B-{name}.jsonl"
    done = {json.loads(x)["id"] for x in out_path.read_text().splitlines()} if out_path.exists() else set()
    todo = [it for it in items() if it["id"] not in done]
    if not todo:
        return
    log = open(ROOT / f"logs/battery-B-{name}-server.log", "a")
    server = subprocess.Popen([str(BIN / "llama-server"), "-m", model, "-ngl", "99", "-c", "8192", "-np", "1", "--jinja",
                               "--port", str(PORT), "--override-kv", "inkling2_loop.loop.count=int:3", "--seed", "0"],
                              stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(900):
            try:
                with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                    if json.load(r).get("status") == "ok":
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            sys.exit("server not ready")
        with open(out_path, "a") as f:
            for k, it in enumerate(todo):
                body = {"messages": it["messages"], "temperature": 0, "seed": 0, "max_tokens": 2048,
                        "chat_template_kwargs": {"reasoning_effort": "low"}}
                if "tools" in it:
                    body["tools"] = it["tools"]
                t0 = time.time()
                try:
                    resp = post(body)
                    c, usage = resp["choices"][0], resp.get("usage")
                    rec = {"finish_reason": c.get("finish_reason"), "content": c["message"].get("content"),
                           "reasoning_content": c["message"].get("reasoning_content"), "tool_calls": c["message"].get("tool_calls")}
                except Exception as e:  # a server error is logged and scored as wrong
                    rec, usage = {"finish_reason": "error", "error": str(e)[:300]}, None
                f.write(json.dumps({"id": it["id"], "task": it["task"], **rec, "usage": usage, "seconds": round(time.time() - t0, 1)}) + "\n")
                f.flush()
                print(f"[{time.strftime('%H:%M:%S')}] {name} {k + 1}/{len(todo)} {it['id']} {rec['finish_reason']}", flush=True)
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
