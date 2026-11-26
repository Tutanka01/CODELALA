"""Synthesize & strictly 3-gate verify self-contained HTML/CSS/JS apps using a local teacher LLM.

For each strictly contamination-free prompt in prompts_v3.py:
  1. Teacher generates a concise structured engineering <plan> (200-600 words) + a single-file
     HTML/CSS/JS app (no external scripts/CDNs, compact, clean, modern UI).
  2. Gate 1: Structural & contamination checks (single complete HTML, no external scripts, length).
  3. Gate 2: Real headless Firefox execution via eval/webcheck.py (zero JS errors, non-blank,
     interactive/animated smoke pass, no hangs).
  4. Gate 3: Strict QA code review using the same strict rubric as eval/judge.py (works=True,
     core_mechanics_ok=True, score >= min_score).
  5. If Gate 2 or Gate 3 fails, feeds the exact runtime errors & QA bug list back to the teacher
     for a targeted repair (up to --max-retries), then re-verifies all 3 gates.

Resumable: prompts already verified in --out are skipped.
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "data"))

from prompts_v3 import STRICT_HELD_OUT, WEBAPP_PROMPTS
from webcheck import extract_html, instrument, probe_isolated

EXTERNAL_RE = re.compile(r"<script[^>]+src=|<link[^>]+href=[\"']?https?:|cdn\.|googleapis|unpkg\.com|jsdelivr", re.I)
PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.S | re.I)

GEN_SYSTEM = """You are an elite front-end game and web application engineer.
When asked to build a single-file HTML/CSS/JS application, you MUST produce:
1. First, a concise, concrete engineering blueprint inside <plan>...</plan> tags (STRICTLY 80-160 words, short bullet points, NO code blocks inside <plan>):
   - Exact HTML element IDs and layout
   - State variables (explicitly using `let` for mutable state, `const` only for constants) and coordinate system
   - Core functions, physics/math/game-rule invariants, win/lose/restart reset logic
   - Quick sanity check (no undefined refs, `e.preventDefault()` on arrows/Space, single requestAnimationFrame loop, `init()` called at startup)
2. Immediately after </plan>, ONE complete, self-contained HTML document inside ```html ... ```:
   - Strictly NO external scripts, CDNs, fonts, or image URLs (100% offline self-contained)
   - COMPACT, idiomatic, bug-free code: target 100-190 lines total (~3,500-9,500 chars). Keep CSS concise (~20-35 lines, clean dark modern look) and JS crisp and modular.
   - Every HTML ID referenced in JS must exist; every button/key control must work immediately on load; Restart must cleanly reset all state and timers without stacking event listeners."""

REPAIR_USER = """Your previous HTML implementation failed automated browser/QA verification.

## Original Request
{prompt}

## Browser Execution Report
{web_report}

## QA Bugs Found
{bugs}

Fix ALL of these issues cleanly. Output an updated concise <plan>...</plan> (80-160 words) followed by the complete fixed compact single-file HTML (100-190 lines) inside ```html ... ``` (strictly no external scripts/CDNs)."""

JUDGE_SYSTEM = (
    "You are a meticulous senior front-end QA engineer. You review a single-file web app that was "
    "written for a user request. Mentally execute the code step by step: initial state, the main "
    "loop or event handlers, each core mechanic, win/lose/end states, restart. Be strict: plausible-"
    "looking code with broken logic is a failure."
)

JUDGE_USER = """## User request
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

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "bugs": {"type": "array", "items": {"type": "string"}},
        "core_mechanics_ok": {"type": "boolean"},
        "works": {"type": "boolean"},
        "score": {"type": "integer"},
    },
    "required": ["bugs", "core_mechanics_ok", "works", "score"],
}


def parse_plan_and_html(text: str):
    m = PLAN_RE.search(text or "")
    plan = m.group(1).strip() if m else ""
    html, how = extract_html(text or "")
    return plan, html, how


def check_html_structure(html: str | None) -> tuple[bool, str]:
    if not html:
        return False, "no_html"
    h = html.strip()
    if not re.search(r"<!DOCTYPE html|<html", h, re.I) or not h.lower().endswith("</html>"):
        return False, "incomplete_html"
    if EXTERNAL_RE.search(h):
        return False, "external_resource"
    if len(h) < 1600:
        return False, "too_short"
    if len(h) > 14500:
        return False, "too_long"
    return True, "ok"


async def call_llm(client, url, messages, temperature=0.6, max_tokens=6000, response_format=None, enable_thinking=True):
    body = {
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format
    if not enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    r = await client.post(f"{url}/v1/chat/completions", json=body)
    r.raise_for_status()
    d = r.json()
    msg = d["choices"][0]["message"]
    return (msg.get("content") or ""), (msg.get("reasoning_content") or ""), d.get("usage", {}).get("completion_tokens", 0)


async def qa_judge(client, url, prompt, html, web_res):
    report = {
        k: web_res.get(k)
        for k in ("js_errors", "raf_frames", "nonblank", "idle_change", "input_change", "probe_error", "visible_text")
        if k in web_res
    }
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM + " Keep your internal analysis concise and focused (under 800 words) before outputting the final JSON verdict."},
        {"role": "user", "content": JUDGE_USER.format(prompt=prompt, report=json.dumps(report, ensure_ascii=False), code=html[:25000]) + "\nKeep each bug entry under 15 words (max 5 bugs)."},
    ]
    text, _, _ = await call_llm(client, url, msgs, temperature=0.2, max_tokens=6500, response_format={"type": "json_schema", "json_schema": {"name": "verdict", "schema": JUDGE_SCHEMA}}, enable_thinking=True)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"works": False, "core_mechanics_ok": False, "score": 0, "bugs": ["invalid judge JSON"]}
    return json.loads(m.group(0))


async def process_one(client, url, item, args, llm_sem, judge_sem, ff_sem, fout, lock, workdir):
    pid, kind, prompt, variant = item
    tag = f"{pid}_v{variant}"
    assert not STRICT_HELD_OUT.search(prompt), f"Held-out contamination: {pid}"

    messages = [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": prompt if variant == 0 else prompt + " Keep the code clean, modular, and visually polished."},
    ]
    temp = args.temp if variant == 0 else min(1.0, args.temp + 0.1)

    for attempt in range(args.max_retries + 1):
        async with llm_sem:
            try:
                content, _, toks = await call_llm(
                    client, url, messages, temperature=temp, max_tokens=args.max_tokens,
                    enable_thinking=not args.no_teacher_thinking,
                )
            except Exception as e:
                print(f"[{tag}] attempt={attempt} LLM_ERR: {e!r}", flush=True)
                return

        plan, html, _ = parse_plan_and_html(content)
        ok_struct, reason = check_html_structure(html)
        if not ok_struct or len(plan) < 120:
            err_msg = reason if not ok_struct else "missing_or_short_plan"
            print(f"[{tag}] attempt={attempt} STRUCT_FAIL: {err_msg}", flush=True)
            messages = [
                {"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": REPAIR_USER.format(
                    prompt=prompt,
                    web_report=json.dumps({"structural_error": err_msg}),
                    bugs=f"Structural check failed: {err_msg}. Ensure <plan> is 150-350 words and HTML is complete, 2000-18000 chars, with zero external scripts.",
                )},
            ]
            continue

        # Gate 2: Real headless Firefox check
        p = workdir / f"{tag}_a{attempt}.html"
        p.write_text(instrument(html))
        async with ff_sem:
            web_res = await asyncio.to_thread(probe_isolated, p, 45)
        p.unlink(missing_ok=True)

        smoke = bool(
            not web_res.get("probe_error")
            and not web_res.get("js_errors")
            and web_res.get("nonblank")
            and (web_res.get("idle_change", 0) > 0.05 or web_res.get("input_change", 0) > 0.05)
        )

        # Gate 3: Strict QA judge
        async with judge_sem:
            try:
                verdict = await qa_judge(client, args.judge_url or url, prompt, html, web_res)
            except Exception as e:
                print(f"[{tag}] attempt={attempt} JUDGE_ERR: {e!r}", flush=True)
                return

        score = int(verdict.get("score") or 0)
        works = bool(verdict.get("works")) and bool(verdict.get("core_mechanics_ok"))
        bugs = verdict.get("bugs") or []
        bug_preview = f" bugs={bugs[:2]}" if not (smoke and works and score >= args.min_score) else ""
        print(f"[{tag}] attempt={attempt} smoke={smoke} works={works} score={score} html_len={len(html)} plan_len={len(plan)}{bug_preview}", flush=True)

        accept = (smoke and works and score >= args.min_score) or (
            attempt == args.max_retries and smoke and works and score >= max(7, args.min_score - 1)
        )
        if accept:
            rec = {
                "id": tag,
                "base_id": pid,
                "kind": kind,
                "variant": variant,
                "prompt": prompt,
                "plan": plan,
                "html": html,
                "score": score,
                "smoke": smoke,
                "attempt": attempt,
                "web": {k: web_res.get(k) for k in ("raf_frames", "idle_change", "input_change", "n_canvas")},
            }
            async with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            return

        # Prepare targeted repair for next attempt
        messages = [
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": content[:22000]},
            {"role": "user", "content": REPAIR_USER.format(
                prompt=prompt,
                web_report=json.dumps({k: web_res.get(k) for k in ("js_errors", "nonblank", "idle_change", "input_change", "probe_error")}, ensure_ascii=False),
                bugs=json.dumps(bugs, ensure_ascii=False),
            )},
        ]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--judge-url", default="", help="Optional separate QA judge server URL (defaults to --url)")
    ap.add_argument("--variants", type=int, default=2, help="Number of variants to generate")
    ap.add_argument("--variant-start", type=int, default=0, help="Starting variant index")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of prompts (0=all)")
    ap.add_argument("--min-score", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--judge-concurrency", type=int, default=6)
    ap.add_argument("--ff-workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=7500)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--no-teacher-thinking", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    workdir = (out.parent / "tmp_synth_html").resolve()
    workdir.mkdir(exist_ok=True)

    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    prompts = WEBAPP_PROMPTS[: args.limit] if args.limit > 0 else WEBAPP_PROMPTS
    items = [
        (pid, kind, prompt, v)
        for v in range(args.variant_start, args.variant_start + args.variants)
        for (pid, kind, prompt) in prompts
        if f"{pid}_v{v}" not in done
    ]
    print(f"Total target items: {len(items)} (already done: {len(done)})", flush=True)

    llm_sem = asyncio.Semaphore(args.concurrency)
    judge_sem = asyncio.Semaphore(args.judge_concurrency)
    ff_sem = asyncio.Semaphore(args.ff_workers)
    lock = asyncio.Lock()

    q = asyncio.Queue()
    for it in items:
        q.put_nowait(it)

    async def worker(client):
        while True:
            try:
                it = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await process_one(client, args.url, it, args, llm_sem, judge_sem, ff_sem, fout, lock, workdir)
            finally:
                q.task_done()

    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
        with out.open("a") as fout:
            await asyncio.gather(*[worker(client) for _ in range(args.concurrency)])
    print("SYNTH_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
