"""Sample completions from an OpenAI-compatible server (llama-server) for a task file.

Usage:
  gen.py --tasks eval/tasks_webgen.jsonl --out runs/base/gen.jsonl --url http://127.0.0.1:8090 \
         --samples 3 --effort medium --concurrency 4
Resumable: (task id, sample index) pairs already present in --out are skipped.
"""
import argparse
import asyncio
import json
from pathlib import Path

import httpx


async def one(client, url, task, idx, args, sem, fout, lock):
    body = {
        "messages": [{"role": "user", "content": task["prompt"]}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }
    if args.effort != "none":
        body["chat_template_kwargs"] = {"reasoning_effort": args.effort}
    if args.system:
        body["messages"].insert(0, {"role": "system", "content": args.system})
    async with sem:
        rec = {"id": task["id"], "sample": idx, "prompt": task["prompt"], "kind": task.get("kind")}
        try:
            r = await client.post(f"{url}/v1/chat/completions", json=body)
            r.raise_for_status()
            d = r.json()
            msg = d["choices"][0]["message"]
            rec.update(
                content=msg.get("content") or "",
                reasoning=msg.get("reasoning_content") or "",
                finish=d["choices"][0].get("finish_reason"),
                completion_tokens=d.get("usage", {}).get("completion_tokens"),
            )
        except Exception as e:  # keep going, record the failure
            rec.update(content="", reasoning="", finish="error", error=repr(e)[:300])
        async with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
        print(rec["id"], idx, rec["finish"], rec.get("completion_tokens"), flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    ap.add_argument("--top-k", dest="top_k", type=int, default=40)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=14000)
    ap.add_argument("--system", default="")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            r = json.loads(l)
            if r.get("finish") != "error":
                done.add((r["id"], r["sample"]))
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0)) as client:
        with out.open("a") as fout:
            jobs = [
                one(client, args.url, t, i, args, sem, fout, lock)
                for t in tasks
                for i in range(args.samples)
                if (t["id"], i) not in done
            ]
            await asyncio.gather(*jobs)


if __name__ == "__main__":
    asyncio.run(main())
