"""Evaluate an OpenAI-compatible model server on a strictly held-out subset of MBPP / MBPP+.

Measures real execution pass@1 in isolated subprocesses on BOTH:
  - MBPP base tests (`test_imports` + `test_list`)
  - MBPP+ EvalPlus expanded edge-case stress tests (`test_imports` + `test`, when --full-plus)

Usage:
  eval_python.py --out runs/v4_base/mbpp_medium.jsonl --url http://127.0.0.1:8090 \
                 --n 150 --effort medium --temperature 0.0 --full-plus --concurrency 4
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

FENCE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.S | re.I)


def extract_python(text: str) -> str:
    blocks = FENCE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    return (text or "").strip()


def run_test(code: str, test_code: str, timeout: float = 8.0) -> tuple[bool, str]:
    script = f"{code}\n\n{test_code}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(script)
        path = Path(tf.name)
    try:
        p = subprocess.run(
            [sys.executable, str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if p.returncode == 0:
            return True, "ok"
        err = (p.stderr or p.stdout or "").strip().splitlines()
        return False, (err[-1] if err else f"exit={p.returncode}")[:200]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, repr(e)[:200]
    finally:
        path.unlink(missing_ok=True)


async def eval_one(client, url, row, args, sem, fout, lock):
    prompt = (
        f"{row['prompt'].strip()}\n\n"
        f"Your solution must pass these tests:\n```python\n"
        + "\n".join(row["test_list"][:3])
        + "\n```\nProvide only the Python implementation in a ```python code block."
    )
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }
    if args.effort != "none":
        body["chat_template_kwargs"] = {"reasoning_effort": args.effort}
    async with sem:
        rec = {"task_id": row["task_id"]}
        try:
            r = await client.post(f"{url}/v1/chat/completions", json=body)
            r.raise_for_status()
            d = r.json()
            msg = d["choices"][0]["message"]
            content = msg.get("content") or ""
            code = extract_python(content)
            imp = "\n".join(row.get("test_imports") or [])
            base_test = f"{imp}\n" + "\n".join(row["test_list"])
            passed_base, status_base = await asyncio.to_thread(run_test, code, base_test, 6.0)
            if args.full_plus and row.get("test"):
                if passed_base:
                    plus_test = f"{imp}\n{row['test']}"
                    passed_plus, status_plus = await asyncio.to_thread(run_test, code, plus_test, 10.0)
                else:
                    passed_plus, status_plus = False, status_base
            else:
                passed_plus, status_plus = passed_base, status_base
            rec.update(
                passed=passed_base,
                passed_plus=passed_plus,
                status=status_base,
                status_plus=status_plus,
                finish=d["choices"][0].get("finish_reason"),
                completion_tokens=d.get("usage", {}).get("completion_tokens"),
                code=code[:4000],
            )
        except Exception as e:
            rec.update(passed=False, passed_plus=False, status=repr(e)[:200], status_plus="error", finish="error")
        async with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
        return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    ap.add_argument("--top-k", dest="top_k", type=int, default=40)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=4096)
    ap.add_argument("--full-plus", action="store_true", help="Also run full EvalPlus MBPP+ stress test suite")
    ap.add_argument("--cache", default="data/public/hf_cache")
    args = ap.parse_args()

    ds = load_dataset("evalplus/mbppplus", split="test", cache_dir=args.cache)
    end = min(args.start + args.n, len(ds))
    rows = [ds[i] for i in range(args.start, end)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        with out.open("a") as fout:
            res = await asyncio.gather(*[eval_one(client, args.url, r, args, sem, fout, lock) for r in rows])
    n = max(1, len(res))
    n_base = sum(1 for r in res if r.get("passed"))
    n_plus = sum(1 for r in res if r.get("passed_plus"))
    n_trunc = sum(1 for r in res if r.get("finish") == "length")
    toks = [r["completion_tokens"] for r in res if r.get("completion_tokens")]
    avg_tok = sum(toks) / max(1, len(toks))
    if args.full_plus:
        print(
            f"MBPP ({args.effort}): base={n_base}/{len(res)} ({100*n_base/n:.1f}%) | "
            f"MBPP+={n_plus}/{len(res)} ({100*n_plus/n:.1f}%) | trunc={n_trunc} | avg_tokens={avg_tok:.0f}",
            flush=True,
        )
    else:
        print(
            f"MBPP+ ({args.effort}): {n_base}/{len(res)} ({100*n_base/n:.1f}%) | "
            f"trunc={n_trunc} | avg_tokens={avg_tok:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
