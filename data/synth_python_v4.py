"""Synthesize 240+ strictly contamination-free, execution-verified Python problems using local Qwen3.6.

Clean XML-tag output (ZERO JSON-escaping syntax bugs!) + Python-computed oracle tests:
  1. Qwen3.6 writes <func_name>, <prompt>, <plan>, <solution>, <public_asserts> (3 simple
     spec-checking asserts), and <edge_inputs> (10-12 diverse edge-case argument tuples).
  2. Python executes <solution> + <public_asserts> to verify the reference matches the spec.
  3. Python executes <solution> on every tuple in <edge_inputs> to generate exact deterministic
     `assert func(*args) == expected` hidden tests (zero hand-calculation errors!).
  4. Strict anti-contamination against ALL 378 evalplus/mbppplus tasks (func names + Jaccard).
"""
import argparse
import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
from datasets import load_dataset

CATEGORIES = [
    ("arrays_windows", "arrays, sliding windows, two-pointers, prefix sums, subarrays", "en", 20),
    ("string_parsing", "string manipulation, parsing, tokenization, balanced brackets, formatting", "en", 20),
    ("intervals_greedy", "intervals, merging ranges, scheduling, greedy algorithms", "en", 20),
    ("graphs_trees", "graphs, trees, BFS/DFS, topological sort, cycle detection, shortest path", "en", 20),
    ("dynamic_prog", "dynamic programming, memoization, subsequences, knapsack, coin change, paths", "en", 20),
    ("math_number", "number theory, primes, modular math, combinatorics, base conversion, statistics", "en", 20),
    ("matrix_grid", "2D grids, matrix transforms, flood fill, islands, 2D coordinates", "en", 20),
    ("sorting_search", "binary search, custom sorting, top-k, grouping, partitioning", "en", 20),
    ("dicts_nested", "nested dicts/lists, flattening, deep diff/merge, aggregation, caching", "en", 20),
    ("regex_text", "text normalization, log parsing, version compare, URL/template parsing", "en", 18),
    ("bits_encoding", "bit manipulation, binary/hex encoding, bitmasks, Gray code, checksums", "en", 18),
    ("backend_logic", "practical backend logic: pagination, rate-limit windows, dependency order, retry backoff", "en", 18),
    ("french_python", "algorithmes, listes, chaînes, graphes, programmation dynamique (énoncé en français)", "fr", 20),
]

SYSTEM = (
    "You are an expert Python problem author creating clean, medium-difficulty coding problems "
    "with tricky edge cases for reinforcement learning. Use ONLY Python standard library. "
    "Output strictly the requested XML tags with raw Python code inside (no JSON, no markdown fences inside tags)."
)

USER_TMPL = """Create ONE original, self-contained Python function problem #{idx} in category: {desc}.
Statement language: {lang_desc}.
Avoid reusing these function names: {used_sample}.

Rules:
- Single pure function (takes plain Python JSON-like args: ints, floats, strs, lists, tuples, dicts, bools; returns a plain deterministic value — NO classes, NO floats with rounding ambiguity unless rounded, NO randomness, NO I/O).
- Keep the problem concise, clear, and solvable in 10-40 lines of clean Python.
- `<public_asserts>`: exactly 3 simple, 100%-correct `assert func(...) == ...` statements illustrating the spec.
- `<edge_inputs>`: valid Python list literal assigned to `EDGE_INPUTS = [...]` where each item is a tuple of positional arguments `(arg1, arg2, ...)` testing 10 diverse edge cases (empty, single element, duplicates, ties, negative/zero, boundary values, tricky cases).

Reply using EXACTLY these tags:
<func_name>snake_case_name</func_name>
<prompt>Clear concise problem statement specifying exact function name, arguments, return value, and edge-case rules.</prompt>
<plan>Concise 3-5 bullet plan: algorithm, data structure, complexity, and key edge cases.</plan>
<solution>
# clean Python implementation using only stdlib
def snake_case_name(...):
    ...
</solution>
<public_asserts>
assert snake_case_name(...) == ...
assert snake_case_name(...) == ...
assert snake_case_name(...) == ...
</public_asserts>
<edge_inputs>
EDGE_INPUTS = [
    (...),
]
</edge_inputs>"""

DEF_RE = re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
WORD_RE = re.compile(r"[a-z0-9_]{3,}")


def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", re.S)
    if not m:
        return ""
    val = m.group(1).strip()
    # Strip accidental markdown fences if model included them
    val = re.sub(r"^```(?:python|py)?\s*\n", "", val)
    val = re.sub(r"\n```\s*$", "", val)
    return val.strip()


def verify_and_build_hidden(fn: str, solution: str, public_asserts: list[str], edge_inputs_code: str) -> tuple[bool, list[str], str]:
    """Verify solution passes public_asserts AND execute EDGE_INPUTS to build exact hidden asserts."""
    harness = f"""import json, copy
{solution}

# 1. Verify public spec asserts
""" + "\n".join(public_asserts) + f"""

# 2. Execute edge inputs to build deterministic hidden asserts
{edge_inputs_code}
assert isinstance(EDGE_INPUTS, list) and len(EDGE_INPUTS) >= 5, "need >=5 edge inputs"
hidden = []
for args in EDGE_INPUTS[:12]:
    if not isinstance(args, tuple):
        args = (args,)
    args_copy = copy.deepcopy(args)
    out = {fn}(*args_copy)
    # Ensure repr-roundtrippable deterministic result
    if repr(eval(repr(out))) != repr(out):
        continue
    args_str = ", ".join(repr(a) for a in args)
    hidden.append(f"assert {fn}({{args_str}}) == {{repr(out)}}")
print("__HIDDEN_JSON__" + json.dumps(hidden))
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(harness)
        p = Path(tf.name)
    try:
        r = subprocess.run([sys.executable, str(p)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5.0)
        if r.returncode != 0:
            lines = (r.stderr or r.stdout or "").strip().splitlines()
            return False, [], (lines[-1] if lines else f"exit={r.returncode}")[:180]
        for line in r.stdout.splitlines():
            if line.startswith("__HIDDEN_JSON__"):
                hidden = json.loads(line[len("__HIDDEN_JSON__"):])
                if len(hidden) >= 5:
                    return True, hidden, "ok"
                return False, [], f"too_few_hidden({len(hidden)})"
        return False, [], "no_hidden_marker"
    except subprocess.TimeoutExpired:
        return False, [], "timeout"
    except Exception as e:
        return False, [], repr(e)[:180]
    finally:
        p.unlink(missing_ok=True)


def jaccard(a: set[str], b: set[str]) -> float:
    return (len(a & b) / len(a | b)) if (a and b) else 0.0


async def gen_one(client, url, cat, desc, lang, idx, mbpp_funcs, mbpp_wordsets, used_funcs, sem, lock, fout):
    lang_desc = "French" if lang == "fr" else "English"
    for attempt in range(2):
        async with lock:
            sample_used = ", ".join(sorted(list(used_funcs))[-40:]) or "none"
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(idx=idx, desc=desc, lang_desc=lang_desc, used_sample=sample_used)},
            ],
            "temperature": 0.8 if attempt == 0 else 0.9,
            "top_p": 0.92,
            "max_tokens": 3000,
        }
        async with sem:
            try:
                r = await client.post(f"{url}/v1/chat/completions", json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"].get("content") or ""
            except Exception as e:
                print(f"[{cat}_{idx}] err={e!r}", flush=True)
                continue

        fn = extract_tag(text, "func_name").strip().split()[0] if extract_tag(text, "func_name") else ""
        prompt = extract_tag(text, "prompt")
        plan = extract_tag(text, "plan")
        sol = extract_tag(text, "solution")
        pub_raw = extract_tag(text, "public_asserts")
        edge_code = extract_tag(text, "edge_inputs")
        pub = [l.strip() for l in pub_raw.splitlines() if l.strip().startswith("assert ")]

        if not fn or len(prompt) < 35 or len(sol) < 35 or len(pub) < 2 or "EDGE_INPUTS" not in edge_code:
            print(f"[{cat}_{idx}] BAD_FORMAT fn={fn} pub={len(pub)}", flush=True)
            continue
        if len(sol) > 3000 or len(plan) > 1500:
            continue

        # Strict anti-contamination against ALL 378 MBPP+ tasks
        sol_funcs = set(DEF_RE.findall(sol)) | {fn}
        if any(f.lower() in mbpp_funcs for f in sol_funcs):
            continue
        pw = set(WORD_RE.findall(prompt.lower()))
        if any(jaccard(pw, mw) > 0.50 for mw in mbpp_wordsets):
            continue

        async with lock:
            if fn.lower() in used_funcs:
                continue

        ok, hidden, status = await asyncio.to_thread(verify_and_build_hidden, fn, sol, pub[:3], edge_code)
        if not ok:
            print(f"[{cat}_{idx}] FAIL_VERIFY fn={fn}: {status}", flush=True)
            continue

        rec = {
            "id": f"{cat}_{idx:02d}_{fn}",
            "category": cat,
            "lang": lang,
            "func_name": fn,
            "prompt": prompt,
            "public_tests": pub[:3],
            "hidden_tests": hidden,
            "plan": plan,
            "solution": sol,
        }
        async with lock:
            if fn.lower() in used_funcs:
                continue
            used_funcs.add(fn.lower())
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
        print(f"[{cat}_{idx}] OK fn={fn} tests={len(pub[:3])}+{len(hidden)}", flush=True)
        return True
    return False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--cache", default="data/public/hf_cache")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    ds = load_dataset("evalplus/mbppplus", split="test", cache_dir=args.cache)
    mbpp_funcs = set()
    mbpp_wordsets = []
    for r in ds:
        for f in DEF_RE.findall(r.get("code") or ""):
            mbpp_funcs.add(f.lower())
        mbpp_wordsets.append(set(WORD_RE.findall((r.get("prompt") or "").lower())))
    print(f"Loaded {len(ds)} MBPP+ tasks ({len(mbpp_funcs)} forbidden func names) for strict contamination check.", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    used_funcs = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                used_funcs.add(json.loads(l)["func_name"].lower())

    jobs = [(cat, desc, lang, i) for cat, desc, lang, count in CATEGORIES for i in range(1, count + 1)]
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        with out.open("a") as fout:
            await asyncio.gather(*[
                gen_one(client, args.url, cat, desc, lang, i, mbpp_funcs, mbpp_wordsets, used_funcs, sem, lock, fout)
                for cat, desc, lang, i in jobs
            ])
    total = sum(1 for l in out.read_text().splitlines() if l.strip())
    print(f"SYNTH_PYTHON_DONE: {total} verified contamination-free Python tasks in {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
