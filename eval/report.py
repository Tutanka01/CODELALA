"""Aggregate web.jsonl + judge.jsonl of one or more runs. Usage: report.py runs/base_medium [runs/other ...]"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


for run in sys.argv[1:]:
    web = load(f"{run}/web.jsonl")
    judge = {(j["id"], j["sample"]): j for j in load(f"{run}/judge.jsonl")}
    per = defaultdict(list)
    for w in web:
        j = judge.get((w["id"], w["sample"]), {})
        per[w["id"]].append((w, j))
    print(f"\n== {run}  ({len(web)} samples)")
    print(f"{'task':12} {'smoke':>6} {'works':>6} {'score':>6} {'tokens':>7}")
    tot = []
    for tid, rows in per.items():
        tot += rows
        print(f"{tid:12} {mean([r[0]['smoke_pass'] for r in rows]):6.2f} {mean([r[1].get('works') for r in rows]):6.2f} "
              f"{mean([r[1].get('score') for r in rows]):6.1f} {mean([r[0].get('completion_tokens') for r in rows]):7.0f}")
    print(f"{'ALL':12} {mean([r[0]['smoke_pass'] for r in tot]):6.2f} {mean([r[1].get('works') for r in tot]):6.2f} "
          f"{mean([r[1].get('score') for r in tot]):6.1f} {mean([r[0].get('completion_tokens') for r in tot]):7.0f}")
    both = mean([bool(r[0]['smoke_pass'] and r[1].get('works')) for r in tot])
    print(f"smoke AND judged working: {both:.2f}   truncated (finish=length): {mean([r[0].get('finish') == 'length' for r in tot]):.2f}")
