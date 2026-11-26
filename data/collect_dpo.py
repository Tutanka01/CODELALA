"""Collect Stage-2 On-Policy Execution-Verified DPO Pairs from Olala-SFT rollouts.

For each training prompt (strictly contamination-free, disjoint from the 12 held-out tasks):
  1. Samples K=4 on-policy completions from Stage-1 Olala-SFT (in medium & disabled modes).
  2. Runs real headless Firefox execution (`webcheck.py`) + strict QA judge (`Qwen3.6` on 8091).
  3. Builds high-signal contrastive preference pairs:
     - Natural On-Policy pair: if Olala-SFT produced BOTH a strong passing rollout (smoke=True,
       works=True, score >= 7, not truncated) AND a failing rollout (JS error, broken mechanics,
       score <= 4, or truncated/looping) -> pair them directly!
     - Mistake-Targeted Teacher Repair pair: if ALL rollouts failed, pair Olala's failing rollout
       with a verified passing teacher solution (from synth_v3.jsonl or targeted repair) so DPO
       explicitly suppresses Olala's exact real failure modes!
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
from synth_teacher import qa_judge
from webcheck import extract_html, instrument, probe_isolated


async def sample_student(client, url, prompt, effort, max_tokens=6000):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.85,
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


def make_assistant_msg(content: str, reasoning: str, effort: str):
    m = {"role": "assistant", "content": content}
    if effort != "disabled" and reasoning:
        m["reasoning_content"] = reasoning
    return m


async def process_prompt(client, args, item, teacher_map, stu_sem, tea_sem, ff_sem, fout, lock, workdir):
    pid, kind, prompt = item
    effort = "medium" if hash(pid) % 3 != 0 else "disabled"

    # 1. Sample K on-policy rollouts from Stage-1 Olala-SFT
    rollouts = []
    for k in range(args.k):
        async with stu_sem:
            try:
                ro = await sample_student(client, args.student_url, prompt, effort, max_tokens=args.max_tokens)
                rollouts.append(ro)
            except Exception as e:
                print(f"[{pid} k={k}] student_err: {e!r}", flush=True)

    if not rollouts:
        return

    # 2. Execute in headless Firefox + strict QA judge
    scored = []
    for k, ro in enumerate(rollouts):
        html, _ = extract_html(ro["content"])
        if html is None or ro["finish"] == "length":
            scored.append((ro, False, False, 0, ["truncated_or_no_html"]))
            continue
        p = workdir / f"dpo_{pid}_{k}.html"
        p.write_text(instrument(html))
        async with ff_sem:
            web_res = await asyncio.to_thread(probe_isolated, p, 40)
        p.unlink(missing_ok=True)
        smoke = bool(
            not web_res.get("probe_error")
            and not web_res.get("js_errors")
            and web_res.get("nonblank")
            and (web_res.get("idle_change", 0) > 0.05 or web_res.get("input_change", 0) > 0.05)
        )
        async with tea_sem:
            try:
                v = await qa_judge(client, args.teacher_url, prompt, html, web_res)
            except Exception:
                v = {"works": False, "core_mechanics_ok": False, "score": 0, "bugs": ["judge_err"]}
        works = bool(v.get("works")) and bool(v.get("core_mechanics_ok"))
        score = int(v.get("score") or 0)
        scored.append((ro, smoke, works, score, v.get("bugs") or []))

    # 3. Form high-contrast DPO pairs!
    winners = [s for s in scored if s[1] and s[2] and s[3] >= 7 and s[0]["finish"] == "stop"]
    losers = [s for s in scored if (not s[1] or not s[2] or s[3] <= 4 or s[0]["finish"] == "length")]

    pairs_to_write = []
    if winners and losers:
        best_w = max(winners, key=lambda s: (s[3], -s[0]["tokens"]))
        worst_l = min(losers, key=lambda s: s[3])
        if worst_l[0]["content"]:
            pairs_to_write.append({
                "id": f"{pid}_onpolicy",
                "source": "onpolicy_win_loss",
                "effort": effort,
                "chosen_score": best_w[3],
                "rejected_score": worst_l[3],
                "chosen_messages": [
                    {"role": "user", "content": prompt},
                    make_assistant_msg(best_w[0]["content"], best_w[0]["reasoning"], effort),
                ],
                "rejected_messages": [
                    {"role": "user", "content": prompt},
                    make_assistant_msg(worst_l[0]["content"][:16000], worst_l[0]["reasoning"][:4000], effort),
                ],
            })
    elif losers and pid in teacher_map:
        # All rollouts failed -> pair Olala's real buggy rollout against verified teacher solution!
        t = teacher_map[pid]
        chosen_content = f"```html\n{t['html'].strip()}\n```"
        chosen_plan = (t.get("plan") or "").strip() if effort != "disabled" else ""
        # Pick up to 2 distinct failing rollouts (e.g., worst bug + any truncation/loop)
        losers_sorted = sorted(losers, key=lambda s: s[3])
        for j, litem in enumerate(losers_sorted[:2]):
            if not litem[0]["content"]:
                continue
            pairs_to_write.append({
                "id": f"{pid}_teacher_fix_{j}",
                "source": "onpolicy_rejected_vs_verified_teacher",
                "effort": effort,
                "chosen_score": t.get("score", 9),
                "rejected_score": litem[3],
                "chosen_messages": [
                    {"role": "user", "content": prompt},
                    make_assistant_msg(chosen_content, chosen_plan, effort),
                ],
                "rejected_messages": [
                    {"role": "user", "content": prompt},
                    make_assistant_msg(litem[0]["content"][:16000], litem[0]["reasoning"][:4000], effort),
                ],
            })

    print(f"[{pid}] eff={effort} scores={[s[3] for s in scored]} -> {len(pairs_to_write)} DPO pair(s)", flush=True)
    if pairs_to_write:
        async with lock:
            for rec in pairs_to_write:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--synth", required=True, help="Verified teacher jsonl for fallback chosen")
    ap.add_argument("--student-url", default="http://127.0.0.1:8090")
    ap.add_argument("--teacher-url", default="http://127.0.0.1:8091")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=160)
    ap.add_argument("--max-tokens", type=int, default=5500)
    ap.add_argument("--student-conc", type=int, default=4)
    ap.add_argument("--teacher-conc", type=int, default=6)
    ap.add_argument("--ff-workers", type=int, default=4)
    args = ap.parse_args()

    teacher_map = {}
    for l in Path(args.synth).read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        bid = r["base_id"]
        if bid not in teacher_map or r.get("score", 0) > teacher_map[bid].get("score", 0):
            teacher_map[bid] = r

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    workdir = (out.parent / "tmp_dpo_html").resolve()
    workdir.mkdir(exist_ok=True)

    done_pids = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                done_pids.add(json.loads(l)["id"].rsplit("_", 2)[0])

    prompts = [p for p in WEBAPP_PROMPTS[: args.limit] if p[0] not in done_pids and not STRICT_HELD_OUT.search(p[2])]
    print(f"Collecting DPO pairs on {len(prompts)} prompts (k={args.k})", flush=True)

    stu_sem = asyncio.Semaphore(args.student_conc)
    tea_sem = asyncio.Semaphore(args.teacher_conc)
    ff_sem = asyncio.Semaphore(args.ff_workers)
    lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
        with out.open("a") as fout:
            await asyncio.gather(*[
                process_prompt(client, args, it, teacher_map, stu_sem, tea_sem, ff_sem, fout, lock, workdir)
                for it in prompts
            ])
    print("COLLECT_DPO_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
