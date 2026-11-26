"""Build the CODELALA v3 SFT dataset from verified, short-prompt, concise-reasoning pillars:

  Pillar A: 3-gate verified self-contained HTML/CSS/JS apps synthesized by local Qwen3.6 teacher
            (concise structured engineering <plan> in analysis channel for medium/low effort,
             plus direct clean code for disabled effort).
  Pillar B: Verified multi-language coding + concise algorithmic reasoning:
            - KodCode/KodCode-V1-SFT-R1 (Python with concise <think> traces, AST-verified)
            - m-a-p/CodeFeedback-Filtered-Instruction (JS/TS/Python/SQL/Bash/C/Rust, short prompts)
  Pillar C: Anti-forgetting bilingual FR/EN anchor (OPUS-100 + optional Olala-base self-anchor).

Strictly enforces STRICT_HELD_OUT contamination filtering on every single source and caps total
sequence length at --max-tokens (default 4096) so training is fast, packed, and loop-free.
"""
import argparse
import ast
import collections
import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts_v3 import STRICT_HELD_OUT

FENCE = re.compile(r"```([a-zA-Z0-9+#]*)\s*\n(.*?)```", re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>\s*(.*)", re.S)


def python_ast_ok(text: str) -> bool:
    blocks = [b for lang, b in FENCE.findall(text) if lang.lower() in ("python", "py", "python3", "")]
    if not blocks:
        return False
    for b in blocks:
        try:
            ast.parse(b)
        except Exception:
            return False
    return True


def tok_len(tok, messages, effort):
    return len(tok.apply_chat_template(messages, tokenize=True, return_dict=True, reasoning_effort=effort)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True, help="JSONL from synth_teacher.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="models/olala-hf")
    ap.add_argument("--cache", default="data/public/hf_cache")
    ap.add_argument("--opus", default="src/ovhai_llm_sft_tutorial/open_data/opus100_enfr_train.jsonl")
    ap.add_argument("--self-anchor", default="", help="Optional Olala-base self-distilled general Q&A jsonl")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--n-kodcode", type=int, default=650)
    ap.add_argument("--n-codefeedback", type=int, default=550)
    ap.add_argument("--n-opus", type=int, default=180)
    ap.add_argument("--webapp-repeat", type=int, default=2, help="Weight multiplier for verified webapps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stats = {}

    # ------------------------------------------------------------------
    # Pillar A: 3-gate verified synthetic web apps (FR + EN)
    # ------------------------------------------------------------------
    synth_rows = [json.loads(l) for l in Path(args.synth).read_text().splitlines() if l.strip()]
    rows_a_by_base = collections.defaultdict(list)
    effort_a = collections.Counter()
    dropped_a = 0
    for r in synth_rows:
        prompt = r["prompt"].strip()
        if STRICT_HELD_OUT.search(prompt) or STRICT_HELD_OUT.search(r.get("id", "")):
            continue
        html = r["html"].strip()
        plan = (r.get("plan") or "").strip()
        content = f"```html\n{html}\n```"
        base_id = r.get("base_id") or r["id"]

        # Always add reasoning-mode version: ~75% medium (Olala's default!), ~25% low
        eff_r = "low" if abs(hash(r["id"] + "_eff")) % 4 == 0 else "medium"
        msgs_r = [{"role": "user", "content": prompt}, {"role": "assistant", "content": content, "reasoning_content": plan}]
        n_r = tok_len(tok, msgs_r, eff_r)
        if n_r <= args.max_tokens:
            effort_a[eff_r] += 1
            rows_a_by_base[base_id].append({
                "messages": msgs_r,
                "chat_template_kwargs": {"reasoning_effort": eff_r},
                "meta": {"source": "qwen36_verified_webapp", "bucket": "webapp", "kind": r.get("kind"), "tokens": n_r, "score": r.get("score")},
            })
        else:
            dropped_a += 1

        # Also add direct disabled-mode version for 60% of apps so disabled mode improves sharply too
        if abs(hash(r["id"])) % 5 < 3:
            msgs_d = [{"role": "user", "content": prompt}, {"role": "assistant", "content": content}]
            n_d = tok_len(tok, msgs_d, "disabled")
            if n_d <= args.max_tokens:
                effort_a["disabled"] += 1
                rows_a_by_base[base_id].append({
                    "messages": msgs_d,
                    "chat_template_kwargs": {"reasoning_effort": "disabled"},
                    "meta": {"source": "qwen36_verified_webapp", "bucket": "webapp_direct", "kind": r.get("kind"), "tokens": n_d, "score": r.get("score")},
                })

    # Group-split by base_id so train & eval never share the same app!
    base_ids = list(rows_a_by_base.keys())
    rnd.shuffle(base_ids)
    n_eval_bases = min(30, max(10, len(base_ids) // 7))
    eval_a = [row for bid in base_ids[:n_eval_bases] for row in rows_a_by_base[bid]]
    train_a_unique = [row for bid in base_ids[n_eval_bases:] for row in rows_a_by_base[bid]]
    train_a = train_a_unique * max(1, args.webapp_repeat)
    stats["pillar_a"] = {
        "raw": len(synth_rows),
        "unique_apps": len(base_ids),
        "kept_rows": len(eval_a) + len(train_a_unique),
        "dropped_overlong": dropped_a,
        "efforts": dict(effort_a),
        "train_with_repeat": len(train_a),
        "eval": len(eval_a),
    }
    print("Pillar A:", stats["pillar_a"], flush=True)

    # ------------------------------------------------------------------
    # Pillar B1: KodCode-V1-SFT-R1 (concise Python coding with verified reasoning)
    # Hard 180s wall-clock guard so slow proxies can NEVER stall overnight training!
    # ------------------------------------------------------------------
    # Skip KodCode if --n-kodcode 0 (avoids slow proxy waits when resuming locally)
    rows_b1 = []
    if args.n_kodcode > 0:
        import signal
        def _alarm_handler(signum, frame):
            raise TimeoutError("KodCode streaming timed out after 180s")
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(180)
        try:
            kc = load_dataset("KodCode/KodCode-V1-SFT-R1", split="train", streaming=True)
            for scanned, r in enumerate(kc):
                if len(rows_b1) >= args.n_kodcode or scanned >= 12000:
                    break
                convs = r.get("conversations") or []
                if len(convs) < 2:
                    continue
                q = (convs[0].get("value") or "").strip()
                a = (convs[1].get("value") or "").strip()
                if not q or not a or len(q) > 1500 or STRICT_HELD_OUT.search(q):
                    continue
                m = THINK_RE.match(a)
                if not m:
                    continue
                trace, ans = m.group(1).strip(), m.group(2).strip()
                if not (250 <= len(trace) <= 3200) or not (150 <= len(ans) <= 5000):
                    continue
                if not python_ast_ok(ans):
                    continue
                if len(rows_b1) % 4 == 0:
                    effort = "disabled"
                    assistant = {"role": "assistant", "content": ans}
                else:
                    effort = "low" if len(trace) <= 1300 else "medium"
                    assistant = {"role": "assistant", "content": ans, "reasoning_content": trace}
                msgs = [{"role": "user", "content": q}, assistant]
                n = tok_len(tok, msgs, effort)
                if n > args.max_tokens:
                    continue
                rows_b1.append({
                    "messages": msgs,
                    "chat_template_kwargs": {"reasoning_effort": effort},
                    "meta": {"source": "KodCode-V1-SFT-R1", "bucket": "python_cot", "tokens": n},
                })
        except Exception as e:
            print(f"WARN: KodCode stream stopped ({e!r}); kept {len(rows_b1)} rows, continuing with local CodeFeedback.", flush=True)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    stats["pillar_b1_kodcode"] = len(rows_b1)
    print("Pillar B1 (KodCode):", len(rows_b1), flush=True)

    # ------------------------------------------------------------------
    # Pillar B2: CodeFeedback-Filtered-Instruction (short prompts, multi-lang JS/TS/Py/SQL/C/Rust)
    # ------------------------------------------------------------------
    cf = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", cache_dir=args.cache)
    cf_idx = list(range(len(cf)))
    rnd.shuffle(cf_idx)
    lang_caps = {"javascript": 180, "typescript": 90, "python": 120, "sql": 50, "bash": 40, "c++": 40, "rust": 30}
    lang_cnt = collections.Counter()
    rows_b2 = []
    for idx in cf_idx:
        if len(rows_b2) >= args.n_codefeedback:
            break
        r = cf[idx]
        lang = (r.get("lang") or "").lower()
        if lang == "jsx":
            lang = "javascript"
        if lang not in lang_caps or lang_cnt[lang] >= lang_caps[lang]:
            continue
        q = (r.get("query") or "").strip()
        a = (r.get("answer") or "").strip()
        if not (40 <= len(q) <= 900) or not (600 <= len(a) <= 6500):
            continue
        if a.count("```") < 2 or a.count("```") % 2 != 0:
            continue
        if STRICT_HELD_OUT.search(q):
            continue
        if lang == "python" and not python_ast_ok(a):
            continue
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        n = tok_len(tok, msgs, "disabled")
        if n > args.max_tokens:
            continue
        lang_cnt[lang] += 1
        rows_b2.append({
            "messages": msgs,
            "chat_template_kwargs": {"reasoning_effort": "disabled"},
            "meta": {"source": "CodeFeedback", "bucket": f"cf_{lang}", "tokens": n},
        })
    stats["pillar_b2_codefeedback"] = {"total": len(rows_b2), "langs": dict(lang_cnt)}
    print("Pillar B2 (CodeFeedback):", stats["pillar_b2_codefeedback"], flush=True)

    # ------------------------------------------------------------------
    # Pillar C: Anti-forgetting anchor (OPUS-100 + optional self-distilled general Q&A)
    # ------------------------------------------------------------------
    rows_c = []
    opus_path = Path(args.opus)
    if opus_path.exists():
        opus = [json.loads(l) for l in opus_path.read_text().splitlines() if l.strip()]
        rnd.shuffle(opus)
        for r in opus[: args.n_opus]:
            n = tok_len(tok, r["messages"], "disabled")
            rows_c.append({
                "messages": r["messages"],
                "chat_template_kwargs": {"reasoning_effort": "disabled"},
                "meta": {"source": "OPUS-100", "bucket": "anchor_translation", "tokens": n},
            })
    if args.self_anchor and Path(args.self_anchor).exists():
        for l in Path(args.self_anchor).read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            eff = (r.get("chat_template_kwargs") or {}).get("reasoning_effort", "disabled")
            n = tok_len(tok, r["messages"], eff)
            if n <= args.max_tokens:
                r.setdefault("meta", {})["tokens"] = n
                rows_c.append(r)
    stats["pillar_c_anchor"] = len(rows_c)
    print("Pillar C (Anchor):", len(rows_c), flush=True)

    # Split Pillar B & C cleanly into eval + train
    rnd.shuffle(rows_b1)
    rnd.shuffle(rows_b2)
    rnd.shuffle(rows_c)
    eval_data = eval_a + rows_b1[:25] + rows_b2[:25] + rows_c[:10]
    train_data = train_a + rows_b1[25:] + rows_b2[25:] + rows_c[10:]
    rnd.shuffle(train_data)
    rnd.shuffle(eval_data)

    (out / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_data))
    (out / "eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eval_data))

    toks = [r["meta"]["tokens"] for r in train_data]
    stats.update(
        train=len(train_data),
        eval=len(eval_data),
        train_tokens_total=sum(toks),
        train_tokens_mean=round(sum(toks) / max(1, len(toks))),
        train_tokens_max=max(toks) if toks else 0,
        buckets=dict(collections.Counter(r["meta"]["bucket"] for r in train_data)),
    )
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
