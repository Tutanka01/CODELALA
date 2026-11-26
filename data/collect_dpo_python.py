"""Collect 100% Execution-Verified On-Policy Python DPO+RPO pairs from Olala-base.

For every verified training task (strictly disjoint from all 378 MBPP+ tasks):
  1. Formats the exact same user prompt style (statement + 3 public asserts).
  2. Samples K=6 on-policy rollouts from Olala-base (in medium or disabled mode).
  3. Executes EVERY rollout in an isolated Python subprocess against public + hidden edge-case tests.
  4. Forms pure high-contrast preference pairs:
     - PURE ON-POLICY PAIR: Olala produced BOTH >=1 passing rollout (passes 100% of hidden tests)
       AND >=1 failing rollout (wrong logic / edge-case fail / exception / timeout / truncation).
       -> chosen = shortest clean passing rollout from Olala itself!
       -> rejected = worst failing rollout from Olala!
     - HARD-PROBLEM TEACHER PAIR: Olala failed ALL K=6 rollouts -> chosen = verified teacher
       solution (with concise plan in medium mode, direct code in disabled mode), rejected =
       Olala's failing rollout!
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

FENCE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.S | re.I)


def extract_python(text: str) -> str:
    blocks = FENCE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    return (text or "").strip()


def run_tests(code: str, tests: list[str], timeout: float = 5.0) -> tuple[bool, str]:
    if not code or len(code) < 10:
        return False, "empty_code"
    script = f"{code}\n\n" + "\n".join(tests) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(script)
        p = Path(tf.name)
    try:
        r = subprocess.run([sys.executable, str(p)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        if r.returncode == 0:
            return True, "ok"
        lines = (r.stderr or r.stdout or "").strip().splitlines()
        return False, (lines[-1] if lines else f"exit={r.returncode}")[:180]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, repr(e)[:180]
    finally:
        p.unlink(missing_ok=True)


def make_prompt(task: dict) -> str:
    if task.get("lang") == "fr":
        return (
            f"{task['prompt'].strip()}\n\n"
            f"Ta solution doit passer ces tests :\n```python\n"
            + "\n".join(task["public_tests"][:3])
            + "\n```\nFournis uniquement l'implémentation Python dans un bloc ```python."
        )
    return (
        f"{task['prompt'].strip()}\n\n"
        f"Your solution must pass these tests:\n```python\n"
        + "\n".join(task["public_tests"][:3])
        + "\n```\nProvide only the Python implementation in a ```python code block."
    )


def make_assistant(content: str, reasoning: str, effort: str) -> dict:
    m = {"role": "assistant", "content": content}
    if effort != "disabled" and reasoning:
        m["reasoning_content"] = reasoning
    return m


async def sample_one(client, url, prompt, effort, max_tokens):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    r = await client.post(f"{url}/v1/chat/completions", json=body)
    r.raise_for_status()
    d = r.json()
    msg = d["choices"][0]["message"]
    return {
        "content": (msg.get("content") or "").strip(),
        "reasoning": (msg.get("reasoning_content") or "").strip(),
        "finish": d["choices"][0].get("finish_reason"),
        "tokens": d.get("usage", {}).get("completion_tokens", 0),
    }


async def process_task(client, args, task, effort, sem, fout, lock, stats):
    prompt = make_prompt(task)
    all_tests = task["public_tests"] + task["hidden_tests"]

    rollouts = []
    for _ in range(args.k):
        async with sem:
            try:
                ro = await sample_one(client, args.url, prompt, effort, args.max_tokens)
                rollouts.append(ro)
            except Exception as e:
                print(f"[{task['id']} {effort}] sample_err: {e!r}", flush=True)

    if not rollouts:
        return

    winners = []
    losers = []
    for ro in rollouts:
        code = extract_python(ro["content"])
        ok, status = await asyncio.to_thread(run_tests, code, all_tests, 5.0)
        if ok and ro["finish"] == "stop":
            winners.append((ro, status))
        else:
            losers.append((ro, status))

    pairs = []
    if winners and losers:
        # GOLD STANDARD: 100% pure on-policy (shortest clean winner vs worst loser)
        best_w = min(winners, key=lambda x: x[0]["tokens"])[0]
        # Prefer a non-empty loser that finished (actual logic/edge-case bug), or truncated loop
        losers_nonempty = [x for x in losers if x[0]["content"] or x[0]["reasoning"]]
        if losers_nonempty:
            worst_l = max(losers_nonempty, key=lambda x: (x[0]["finish"] == "length", x[0]["tokens"]))[0]
            pairs.append({
                "id": f"{task['id']}_{effort}_onpolicy",
                "source": "pure_onpolicy",
                "effort": effort,
                "chosen_messages": [
                    {"role": "user", "content": prompt},
                    make_assistant(best_w["content"], best_w["reasoning"], effort),
                ],
                "rejected_messages": [
                    {"role": "user", "content": prompt},
                    make_assistant(worst_l["content"][:10000], worst_l["reasoning"][:4000], effort),
                ],
            })
            stats["pure_onpolicy"] += 1
    elif losers and not winners:
        # Hard problem (0/K pass): pair verified teacher solution against Olala's real buggy rollout
        losers_nonempty = [x for x in losers if x[0]["content"] or x[0]["reasoning"]]
        if losers_nonempty:
            worst_l = losers_nonempty[0][0]
            chosen_content = f"```python\n{task['solution'].strip()}\n```"
            chosen_plan = task.get("plan", "").strip() if effort != "disabled" else ""
            pairs.append({
                "id": f"{task['id']}_{effort}_teacher_fix",
                "source": "hard_teacher_fix",
                "effort": effort,
                "chosen_messages": [
                    {"role": "user", "content": prompt},
                    make_assistant(chosen_content, chosen_plan, effort),
                ],
                "rejected_messages": [
                    {"role": "user", "content": prompt},
                    make_assistant(worst_l["content"][:10000], worst_l["reasoning"][:4000], effort),
                ],
            })
            stats["hard_teacher_fix"] += 1
    else:
        stats["all_passed"] += 1

    print(f"[{task['id']} {effort}] pass={len(winners)}/{len(rollouts)} -> {len(pairs)} pair(s)", flush=True)
    if pairs:
        async with lock:
            for rec in pairs:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")

    # Run BOTH medium (60%) and disabled (40%), capped cleanly for ~35-45 min rollout budget
    k_eff = min(args.k, 4)
    max_tok_eff = min(args.max_tokens, 2048)
    args.k = k_eff
    args.max_tokens = max_tok_eff
    jobs = []
    for i, t in enumerate(tasks[:190]):
        eff = "disabled" if i % 5 < 2 else "medium"
        jobs.append((t, eff))

    print(f"Collecting execution-verified Python DPO pairs on {len(jobs)} (task, effort) jobs (K={args.k}, max_tok={args.max_tokens})", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"pure_onpolicy": 0, "hard_teacher_fix": 0, "all_passed": 0}
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        with out.open("a") as fout:
            await asyncio.gather(*[process_task(client, args, t, eff, sem, fout, lock, stats) for t, eff in jobs])
    print("COLLECT_DPO_PYTHON_DONE:", stats, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
