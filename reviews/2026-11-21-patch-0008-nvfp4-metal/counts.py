"""Counts for reviews/2026-11-21-patch-0008-nvfp4-metal.md, read from the test-backend-ops logs and the patches beside it.

Run: .venv/bin/python -I reviews/2026-11-21-patch-0008-nvfp4-metal/counts.py > reviews/2026-11-21-patch-0008-nvfp4-metal/counts.txt
"""

import collections
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PATCHES = ROOT / "patches" / "llama.cpp-proposed"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
CASE = re.compile(r"^(?:\[\w+\] ERR = \S+ > \S+\s+)?\s*([A-Z_0-9]+)\((.*)\): (.*?)\s*$")
SUMMARY = re.compile(r"^\s*(\d+)/(\d+) tests passed")
KERNEL = re.compile(r"base = '(kernel_[a-z_0-9]*nvfp4[a-z_0-9]*)'")


def cases(path):
    """(op, params, status) for every NVFP4 case line, and the n/m summary line of the MTL0 run."""
    out, summary = [], None
    for line in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", line)
        m = CASE.match(line)
        if m and "nvfp4" in m.group(2):
            out.append(m.groups())
        s = SUMMARY.match(line)
        if s:
            summary = (int(s.group(1)), int(s.group(2)))
    return out, summary


def table(title, files):
    print(f"## {title}")
    print()
    print("| log | op | status | cases |")
    print("|---|---|---|---:|")
    for f in files:
        c = collections.Counter((op, st) for op, _, st in cases(HERE / f)[0])
        for (op, st), n in sorted(c.items()):
            print(f"| {f} | {op} | {st} | {n} |")
    print()


def summaries(files):
    print("| log | MTL0 tests passed | MTL0 tests run |")
    print("|---|---:|---:|")
    for f in files:
        s = cases(HERE / f)[1]
        print(f"| {f} | {s[0] if s else 'n/a'} | {s[1] if s else 'n/a'} |")
    print()


def main():
    ops = ["GET_ROWS", "MUL_MAT", "MUL_MAT_ID"]

    table("Old test binary, unpatched Metal (-p nvfp4)", [f"before-{o}.txt" for o in ops])
    table("New test binary (patch 0009), unpatched Metal (-p nvfp4)", [f"before-newtests-{o}.txt" for o in ops])
    table("New test binary, patched Metal (-p nvfp4)", [f"after-{o}.txt" for o in ops])
    summaries([f"before-newtests-{o}.txt" for o in ops] + [f"after-{o}.txt" for o in ops])

    print("## New cases (params contain raw=1 or k=1152), patched Metal")
    print()
    print("| log | op | status | cases |")
    print("|---|---|---|---:|")
    for o in ops:
        c = collections.Counter(
            (op, st, "raw" if "raw=1" in p else "quantized")
            for op, p, st in cases(HERE / f"after-{o}.txt")[0]
            if "raw=1" in p or "k=1152" in p
        )
        for (op, st, kind), n in sorted(c.items()):
            print(f"| after-{o}.txt | {op} ({kind}) | {st} | {n} |")
    print()

    print("## Mutation check: decode returns 0 for scale bytes >= 0x80")
    print()
    print("| log | raw cases FAIL | raw cases OK | other cases FAIL | other cases OK |")
    print("|---|---:|---:|---:|---:|")
    for o in ops:
        cs = cases(HERE / f"mutant-{o}.txt")[0]
        raw = [st for _, p, st in cs if "raw=1" in p]
        other = [st for _, p, st in cs if "raw=1" not in p]
        print(f"| mutant-{o}.txt | {raw.count('FAIL')} | {raw.count('OK')} | {other.count('FAIL')} | {other.count('OK')} |")
    print()

    table("All ops (-p nvfp4), old test binary, unpatched Metal", ["before-all-ops.txt"])
    table("All ops (-p nvfp4), new test binary, patched Metal", ["after-all-ops.txt"])

    print("## Full regression on Metal, all types, patched")
    print()
    summaries([f"after-full-{o}.txt" for o in ops])

    print("## NVFP4 kernels compiled during the patched runs (pipeline compiles per log)")
    print()
    k = collections.Counter()
    for o in ops:
        k.update(KERNEL.findall((HERE / f"after-{o}.txt").read_text(errors="replace")))
    for name, n in sorted(k.items()):
        print(f"- `{name}`: {n}")
    print()

    rows = [l.split() for l in (HERE / "debug-get-rows.txt").read_text().splitlines() if "metal=" in l]
    same = sum(r[1].split("=")[1] == r[2].split("=")[1] for r in rows)
    print("## Hand-made block, get_rows on Metal against the CPU to_float (debug-get-rows.cpp, after the fix)")
    print()
    print(f"- {len(rows)} values, {same} identical as printed with %g, {len(rows) - same} different")
    print()

    print("## Kernel arithmetic (no measurement)")
    print()
    k, qk = 1152, 64
    nb = k // qk
    for name, per_step in (("one thread per sub-block (this patch)", 8), ("one lane pair per block (MXFP4 layout, PR #20456)", 16)):
        steps = -(-nb // per_step)
        print(f"- k = {k}, {nb} blocks per row, {name}: {per_step} blocks per step, {steps} steps, lane use {nb / (steps * per_step):.4f}")
    def ue4m3_true(x):
        # the CPU ggml_ue4m3_to_fp32 times 2 (ggml-impl.h), the value the Metal decode returns
        if x in (0, 0x7F):
            return 0.0
        e, m = (x >> 3) & 0xF, x & 7
        return (m * 2.0 ** -9) if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
    vals = [ue4m3_true(x) for x in range(256)]
    print(f"- largest decoded scale over all 256 bytes: {max(vals)}, times the largest E2M1 magnitude 6: {6 * max(vals)} (fp16 max 65504)")
    print(f"- smallest non-zero decoded scale: {min(v for v in vals if v > 0)}, times the smallest E2M1 magnitude 0.5: {0.5 * min(v for v in vals if v > 0)} (fp16 smallest subnormal {2.0 ** -24})")
    print()

    print("## Patch sizes (added and removed lines per file, from the patch text)")
    print()
    print("| patch | file | added | removed |")
    print("|---|---|---:|---:|")
    for p in sorted(PATCHES.glob("*.patch")):
        cur, add, rem = None, collections.Counter(), collections.Counter()
        for line in p.read_text().splitlines():
            if line.startswith("diff --git "):
                cur = line.split(" b/", 1)[1]
            elif cur and line.startswith("+") and not line.startswith("+++"):
                add[cur] += 1
            elif cur and line.startswith("-") and not line.startswith("---"):
                rem[cur] += 1
        for f in sorted(set(add) | set(rem)):
            print(f"| {p.name[:4]} | `{f}` | {add[f]} | {rem[f]} |")
        print(f"| {p.name[:4]} | total | {sum(add.values())} | {sum(rem.values())} |")
    print()


if __name__ == "__main__":
    sys.exit(main())
