"""Generates the T2 reasoning traces (CHANGELOG.md, 2026-11-21) with the Q8_0 reference at P = 3.

    python scripts/t2-traces.py

Starts llama-server, sends the 20 prompts of data/T2-prompts.jsonl with reasoning_effort "low", temperature 0
and a 2,048-token cap, and writes data/T2-raw.jsonl (full replies), data/T2.txt (reasoning and answer text of
every reply, the scoring file for llama-perplexity) and results/t2.md (hashes, token counts, truncations).
Must run before any release candidate is built.
"""

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "vendor/llama.cpp/build/bin"
MODEL = ROOT / "models/Inkling-2-Mini-Q8_0.gguf"
PORT = 8091
URL = f"http://127.0.0.1:{PORT}"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main() -> None:
    if (ROOT / "data/T2.txt").exists():
        sys.exit("data/T2.txt exists; T2 is frozen once generated")
    log = open(ROOT / "logs/t2-server.log", "w")
    server = subprocess.Popen(
        [str(BIN / "llama-server"), "-m", str(MODEL), "-ngl", "99", "-c", "8192", "-np", "1", "--port", str(PORT),
         "--override-kv", "inkling2_loop.loop.count=int:3", "--seed", "0"],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                    if json.load(r).get("status") == "ok":
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            sys.exit("server did not become ready, see logs/t2-server.log")

        prompts = [json.loads(line) for line in (ROOT / "data/T2-prompts.jsonl").read_text().splitlines()]
        raw, texts, truncated, n_tokens = [], [], 0, 0
        for p in prompts:
            reply = post("/v1/chat/completions", {
                "messages": [{"role": "user", "content": p["question"]}],
                "max_tokens": 2048, "temperature": 0, "seed": 0,
                "chat_template_kwargs": {"reasoning_effort": "low"},
            })
            choice = reply["choices"][0]
            msg = choice["message"]
            text = ((msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")).strip()
            truncated += choice.get("finish_reason") == "length"
            n_tokens += reply.get("usage", {}).get("completion_tokens", 0)
            raw.append({"id": p["id"], "reply": reply})
            texts.append(text)
            print(f"prompt {p['id']}: {reply.get('usage', {}).get('completion_tokens')} tokens, {choice.get('finish_reason')}", flush=True)
    finally:
        server.terminate()
        server.wait(timeout=60)

    (ROOT / "data/T2-raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw), encoding="utf-8")
    (ROOT / "data/T2.txt").write_text("\n\n".join(texts) + "\n", encoding="utf-8")
    h = hashlib.sha256((ROOT / "data/T2.txt").read_bytes()).hexdigest()
    (ROOT / "results/t2.md").write_text(
        "# T2 traces\n\nWritten by `scripts/t2-traces.py`. Do not edit by hand.\n\n"
        f"- Reference: `models/Inkling-2-Mini-Q8_0.gguf` at P = 3, reasoning_effort low, temperature 0, cap 2,048 tokens\n"
        f"- Prompts: 20 (first GSM8K train questions), completion tokens {n_tokens:,}, hit the cap {truncated}\n"
        f"- `data/T2.txt` sha256 `{h[:16]}`\n", encoding="utf-8")
    print(f"T2: {n_tokens} tokens, {truncated} truncated, sha256 {h[:16]}")


if __name__ == "__main__":
    main()
