"""Build fine-tuning set v2 from existing public datasets only:

  A. AdhyanshVerma/html-css-js-cot (GPL-3.0): complete single-file HTML/CSS/JS apps.
     Kept only if the page loads in headless Firefox without JS errors and renders, uses no
     external script, fits the token budget, and is not one of the 12 held-out eval concepts.
     Short reasoning traces are kept as the model's analysis channel; long ones are dropped.
  B. A subset of data/v1 (m-a-p/CodeFeedback-Filtered-Instruction, Apache-2.0), already filtered.
  C. A small French<->English anchor from OVH's tutorial data (OPUS-100).

Output: <out-dir>/train.jsonl, eval.jsonl, stats.json (OVH `messages` format).
Usage: build_v2.py --out-dir data/v2 [--workers 4]
"""
import argparse
import collections
import json
import random
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

HELD_OUT = re.compile(
    r"flap|helicopt|copter|snake|serpent|pong|breakout|brick|memory game|matching pairs|card.?matching|memory card|"
    r"tic.?tac|noughts|to.?do|task list|calculat|pomodoro|2048|minesweeper|paint|drawing app|draw on|whiteboard|sketch",
    re.I,
)
EXTERNAL_SCRIPT = re.compile(r"<script[^>]+src=", re.I)
MAX_TOKENS = 8192
COT_TOKEN_BUDGET = 4_500_000
V1_PICK = {"html": 400, "python": 350, "js": 250, "other": 200}
OPUS_PICK = 200
SHORT_TRACE = 4000  # chars: longer reasoning traces are not used


def check_chunk(args):
    """Worker: load each page in its own headless Firefox, return indices that render cleanly."""
    wid, items, workdir = args
    from PIL import ImageStat
    from webcheck import instrument, new_driver, shot

    ok, reasons = [], collections.Counter()
    d = new_driver()
    for idx, html in items:
        p = Path(workdir) / f"w{wid}_{idx}.html"
        p.write_text(instrument(html))
        try:
            d.get(p.resolve().as_uri())
            time.sleep(0.8)
            errs = d.execute_script("return window.__errs||[]")
            blank = ImageStat.Stat(shot(d)).stddev[0] <= 1.0
            if errs:
                reasons["js_error"] += 1
            elif blank:
                reasons["blank"] += 1
            else:
                reasons["ok"] += 1
                ok.append(idx)
        except Exception:
            reasons["probe_error"] += 1
            try:
                d.quit()
            except Exception:
                pass
            d = new_driver()
        finally:
            p.unlink(missing_ok=True)
    try:
        d.quit()
    except Exception:
        pass
    return ok, dict(reasons)


FENCE = re.compile(r"```([a-zA-Z0-9]*)\s*\n(.*?)```", re.S)


def extract_doc(code: str):
    """The single complete HTML document of a solution, or None (empty, truncated, multi-file)."""
    docs = [b.strip() for _, b in FENCE.findall(code) if re.search(r"<!DOCTYPE html|<html", b, re.I)]
    if len(docs) != 1 or not docs[0].lower().rstrip().endswith("</html>"):
        return None
    return docs[0]


def n_tokens(tok, messages, effort):
    return len(tok.apply_chat_template(messages, tokenize=True, return_dict=True, reasoning_effort=effort)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache", default="data/public/hf_cache")
    ap.add_argument("--v1-dir", default="data/v1")
    ap.add_argument("--opus", default="src/ovhai_llm_sft_tutorial/open_data/opus100_enfr_train.jsonl")
    ap.add_argument("--model", default="models/olala-hf")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stats = {}

    # ---- A. single-file web apps
    ds = load_dataset("AdhyanshVerma/html-css-js-cot", split="train", cache_dir=args.cache)
    cands, why = [], collections.Counter()
    for i, r in enumerate(ds):
        q, code = (r["question_text"] or "").strip(), (r["solution_code"] or "").strip()
        doc = extract_doc(code)
        if doc is None or len(doc) < 4000:
            why["no_or_short_code"] += 1
        elif HELD_OUT.search(q[:1500]):
            why["held_out_concept"] += 1
        elif EXTERNAL_SCRIPT.search(doc):
            why["external_script"] += 1
        elif len(doc) > 26000:
            why["too_long"] += 1
        else:
            cands.append((i, doc))
    docs = dict(cands)
    stats["webapps_total"] = len(ds)
    stats["webapps_prefilter"] = dict(why)
    stats["webapps_candidates"] = len(cands)

    work = out / "tmp_html"
    work.mkdir(exist_ok=True)
    chunks = [(w, cands[w :: args.workers], str(work)) for w in range(args.workers)]
    ok_idx, reasons = set(), collections.Counter()
    with Pool(args.workers) as pool:
        for ok, rs in pool.imap_unordered(check_chunk, chunks):
            ok_idx.update(ok)
            reasons.update(rs)
    stats["webapps_browser"] = dict(reasons)

    rows_a, used_tokens, efforts = [], 0, collections.Counter()
    order = [i for i, _ in cands if i in ok_idx]
    rnd.shuffle(order)
    for i in order:
        r = ds[i]
        trace = (r["solution_thinking_trace"] or "").strip()
        assistant = {"role": "assistant", "content": "```html\n" + docs[i] + "\n```"}
        if 200 <= len(trace) <= SHORT_TRACE:
            effort = "low" if len(trace) <= 1500 else "medium"
            assistant["reasoning_content"] = trace
        else:
            effort = "disabled"
        messages = [{"role": "user", "content": r["question_text"].strip()}, assistant]
        n = n_tokens(tok, messages, effort)
        if n > MAX_TOKENS:
            continue
        if used_tokens + n > COT_TOKEN_BUDGET:
            break
        used_tokens += n
        efforts[effort] += 1
        rows_a.append({"messages": messages, "chat_template_kwargs": {"reasoning_effort": effort},
                       "meta": {"source": "AdhyanshVerma/html-css-js-cot", "lang": "html", "bucket": "webapp"}})
    stats["webapps_kept"] = len(rows_a)
    stats["webapps_tokens"] = used_tokens
    stats["webapps_efforts"] = dict(efforts)

    # ---- B. subset of v1
    v1 = [json.loads(l) for f in ("train.jsonl", "eval.jsonl") for l in (Path(args.v1_dir) / f).read_text().splitlines()]
    rnd.shuffle(v1)
    taken, rows_b = collections.Counter(), []
    for r in v1:
        b = r["meta"]["bucket"]
        if taken[b] < V1_PICK.get(b, 0):
            taken[b] += 1
            rows_b.append(r)
    stats["v1_subset"] = dict(taken)

    # ---- C. French/English anchor
    opus = [json.loads(l) for l in Path(args.opus).read_text().splitlines() if l.strip()]
    rnd.shuffle(opus)
    rows_c = [{"messages": r["messages"], "chat_template_kwargs": {"reasoning_effort": "disabled"},
               "meta": {"source": "OPUS-100 (ovhai_llm_sft_tutorial)", "lang": "fr-en", "bucket": "translation"}}
              for r in opus[:OPUS_PICK]]

    data = rows_a + rows_b + rows_c
    rnd.shuffle(data)
    n_eval = 100
    (out / "eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data[:n_eval]))
    (out / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data[n_eval:]))
    stats.update(train=len(data) - n_eval, eval=n_eval,
                 buckets=dict(collections.Counter(r["meta"]["bucket"] for r in data)))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
