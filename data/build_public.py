"""Build a fine-tuning set from an existing public dataset (m-a-p/CodeFeedback-Filtered-Instruction,
Apache-2.0), keeping only long, complete answers that pass cheap objective checks:

  - HTML documents must load in headless Firefox without JS errors and render something;
  - Python code blocks must parse;
  - requests about the 12 held-out eval concepts are removed.

Output: <out-dir>/train.jsonl, eval.jsonl (OVH `messages` format) and stats.json.
Usage: build_public.py --out-dir data/v1 [--skip-browser]
"""
import argparse
import ast
import collections
import json
import random
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

HELD_OUT = re.compile(
    r"flap|helicopt|copter|snake|serpent|pong|breakout|brick|memory game|matching pairs|card.?matching|"
    r"tic.?tac|noughts|to.?do|task list|calculat|pomodoro|2048|minesweeper|paint|drawing app|draw on",
    re.I,
)
FENCE = re.compile(r"```([a-zA-Z0-9+#]*)\s*\n(.*?)```", re.S)
CAPS = {"html": 1200, "python": 900, "js": 500, "other": 500}
PER_LANG_CAP_OTHER = 110


def python_ok(answer: str) -> bool:
    blocks = [b for lang, b in FENCE.findall(answer) if lang.lower() in ("python", "py", "python3")]
    if not blocks:
        return False
    try:
        for b in blocks:
            ast.parse(b)
        return True
    except (SyntaxError, ValueError, RecursionError):
        return False


def html_candidates(rows):
    out = []
    for r in rows:
        a = r["answer"]
        if len(a) >= 1800 and re.search(r"<!DOCTYPE html|<html", a, re.I) and "<script" in a.lower():
            out.append(r)
    return out


def browser_filter(rows, workdir: Path):
    """Keep rows whose HTML document loads cleanly. Light probe: ~1.5 s per page."""
    from PIL import ImageStat
    from webcheck import extract_html, instrument, new_driver, shot

    workdir.mkdir(parents=True, exist_ok=True)
    keep, reasons = [], collections.Counter()
    d = new_driver()
    for n, r in enumerate(rows):
        a = r["answer"]
        html, how = extract_html(a)
        if html is None:
            html, how = extract_html("```html\n" + a + "\n```")
        if html is None or re.search(r"<script[^>]+src=|<link[^>]+href=[\"']?http|cdn\.|googleapis", html, re.I):
            reasons["no_doc_or_external"] += 1
            continue
        p = workdir / f"{n}.html"
        p.write_text(instrument(html))
        try:
            d.get(p.resolve().as_uri())
            time.sleep(0.8)
            errs = d.execute_script("return window.__errs||[]")
            blank = ImageStat.Stat(shot(d)).stddev[0] <= 1.0
        except Exception:
            reasons["probe_error"] += 1
            try:
                d.quit()
            except Exception:
                pass
            d = new_driver()
            continue
        finally:
            p.unlink(missing_ok=True)
        if errs:
            reasons["js_error"] += 1
        elif blank:
            reasons["blank"] += 1
        else:
            reasons["ok"] += 1
            keep.append(r)
        if n % 100 == 0:
            print(f"browser {n}/{len(rows)} {dict(reasons)}", flush=True)
    d.quit()
    return keep, dict(reasons)


def to_row(r):
    a = r["answer"].strip()
    if re.match(r"<!DOCTYPE html|<html", a, re.I) and "```" not in a:
        a = "```html\n" + a + "\n```"
    return {
        "messages": [{"role": "user", "content": r["query"].strip()}, {"role": "assistant", "content": a}],
        "chat_template_kwargs": {"reasoning_effort": "disabled"},
        "meta": {"source": "m-a-p/CodeFeedback-Filtered-Instruction", "lang": r["lang"], "bucket": r["bucket"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache", default="data/public/hf_cache")
    ap.add_argument("--skip-browser", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", cache_dir=args.cache)
    rows = [dict(r) for r in ds if not HELD_OUT.search(r["query"]) and r["answer"].count("```") % 2 == 0]
    stats = {"total": len(ds), "after_held_out_filter": len(rows)}

    html = html_candidates(rows)
    stats["html_candidates"] = len(html)
    if not args.skip_browser:
        html, stats["html_browser"] = browser_filter(html, out / "tmp_html")
    html_ids = {id(r) for r in html}
    for r in html:
        r["bucket"] = "html"

    py = [r for r in rows if r["lang"] == "python" and len(r["answer"]) >= 3000 and id(r) not in html_ids and python_ok(r["answer"])]
    long_py = [r for r in py if len(r["answer"]) >= 5000]
    rest_py = [r for r in py if len(r["answer"]) < 5000]
    rnd.shuffle(rest_py)
    py = (long_py + rest_py)[: CAPS["python"]]
    for r in py:
        r["bucket"] = "python"

    js = [r for r in rows if r["lang"] in ("javascript", "jsx", "typescript") and len(r["answer"]) >= 2500 and id(r) not in html_ids]
    rnd.shuffle(js)
    js = js[: CAPS["js"]]
    for r in js:
        r["bucket"] = "js"

    other, per = [], collections.Counter()
    pool = [r for r in rows if r["lang"] not in ("python", "javascript", "jsx", "typescript", "html", "css", "json", "xml", "yaml", "http")
            and len(r["answer"]) >= 2500]
    rnd.shuffle(pool)
    for r in pool:
        if per[r["lang"]] < PER_LANG_CAP_OTHER and len(other) < CAPS["other"]:
            per[r["lang"]] += 1
            r["bucket"] = "other"
            other.append(r)

    rnd.shuffle(html)
    picked = html[: CAPS["html"]] + py + js + other
    rnd.shuffle(picked)
    data = [to_row(r) for r in picked]
    n_eval = min(150, len(data) // 20)
    (out / "eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data[:n_eval]))
    (out / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data[n_eval:]))
    stats.update(
        buckets=dict(collections.Counter(r["meta"]["bucket"] for r in data)),
        other_langs=dict(per),
        train=len(data) - n_eval,
        eval=n_eval,
        answer_chars_mean=round(sum(len(r["messages"][1]["content"]) for r in data) / max(1, len(data))),
    )
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
