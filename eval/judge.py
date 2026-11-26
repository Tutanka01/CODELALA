"""LLM judge for generated web apps: a stronger local model reviews the code against the
request and the headless-browser report, and returns a strict verdict.

Usage: judge.py --gen runs/X/gen.jsonl --web runs/X/web.jsonl --out runs/X/judge.jsonl --url http://127.0.0.1:8091
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx

from webcheck import extract_html

SYSTEM = (
    "You are a meticulous senior front-end QA engineer. You review a single-file web app that was "
    "written for a user request. Mentally execute the code step by step: initial state, the main "
    "loop or event handlers, each core mechanic, win/lose/end states, restart. Be strict: plausible-"
    "looking code with broken logic is a failure. Typical failures: obstacles with no gap or drawn "
    "wrong, collisions that can never trigger (undefined/NaN fields), controls not wired, state "
    "never reset, game instantly over, score never increasing, features of the request missing."
)

USER = """## User request
{prompt}

## Headless browser report
{report}

## Code
```html
{code}
```

Decide whether a real user opening this file could actually use it as requested.
Reply with JSON only:
{{"bugs": ["short description of each real bug that affects use"], "core_mechanics_ok": true/false, "works": true/false, "score": 0-10}}
- works=true only if every core mechanic of the request functions correctly (minor cosmetic flaws allowed).
- score: 0-2 broken/unusable, 3-5 runs but core mechanics wrong, 6-7 usable with notable bugs, 8-10 solid."""

SCHEMA = {
    "type": "object",
    "properties": {
        "bugs": {"type": "array", "items": {"type": "string"}},
        "core_mechanics_ok": {"type": "boolean"},
        "works": {"type": "boolean"},
        "score": {"type": "integer"},
    },
    "required": ["bugs", "core_mechanics_ok", "works", "score"],
}


async def judge_one(client, url, rec, web, sem):
    html, _ = extract_html(rec.get("content") or "")
    out = {"id": rec["id"], "sample": rec["sample"]}
    if html is None:
        out.update(works=False, score=0, bugs=["no HTML document in the answer"], core_mechanics_ok=False)
        return out
    report = {k: web.get(k) for k in ("js_errors", "raf_frames", "nonblank", "idle_change", "input_change", "probe_error", "visible_text") if k in web}
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(prompt=rec["prompt"], report=json.dumps(report, ensure_ascii=False), code=html[:40000])},
        ],
        "temperature": 0.2,
        "max_tokens": 6000,
        "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "schema": SCHEMA}},
    }
    async with sem:
        try:
            r = await client.post(f"{url}/v1/chat/completions", json=body)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"].get("content") or ""
            m = re.search(r"\{.*\}", text, re.S)
            out.update(json.loads(m.group(0)))
        except Exception as e:
            out.update(works=None, score=None, error=repr(e)[:200])
    print(out["id"], out["sample"], out.get("works"), out.get("score"), flush=True)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--web", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    gen = [json.loads(l) for l in Path(args.gen).read_text().splitlines() if l.strip()]
    web = {(w["id"], w["sample"]): w for w in map(json.loads, Path(args.web).read_text().splitlines())}
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0)) as client:
        res = await asyncio.gather(*[judge_one(client, args.url, r, web.get((r["id"], r["sample"]), {}), sem) for r in gen])
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res))


if __name__ == "__main__":
    asyncio.run(main())
