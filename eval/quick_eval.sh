#!/usr/bin/env bash
# Fast 3-minute sanity check for a newly merged GGUF checkpoint BEFORE full eval & DPO:
#   1. gguf_diff.py weight-perturbation check vs base Olala (target max_rel ~0.01-0.06)
#   2. 4 held-out web tasks (flappy, snake, todo, calculator) x 2 samples in medium & disabled
#      -> Firefox headless smoke test (webcheck.py) + token/plan length inspection
#   3. 20 MBPP+ Python execution tasks in medium & disabled
# Usage: eval/quick_eval.sh <tag> <gguf_path>
set -euo pipefail
TAG="$1"
GGUF="$2"
cd ~/codelala
export NO_PROXY=127.0.0.1,localhost
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PY=.venv/bin/python
BIN=src/llama.cpp-olala/build-cuda/bin
BASE_GGUF=models/gguf/olala-base-bf16.gguf

echo "=== 1. GGUF weight diff vs base ==="
if [ -f "$BASE_GGUF" ]; then
  $PY diag/gguf_diff.py "$BASE_GGUF" "$GGUF" | head -n 35
fi

# Build 4-task subset of held-out tasks
mkdir -p "runs/${TAG}_quick"
$PY -c '
import json
from pathlib import Path
keep = {"flappy", "snake", "todo", "calculator"}
rows = [l for l in Path("eval/tasks_webgen.jsonl").read_text().splitlines() if l.strip() and json.loads(l)["id"] in keep]
Path("runs/'"${TAG}_quick"'/tasks4.jsonl").write_text("\n".join(rows) + "\n")
'

tmux kill-session -t srv 2>/dev/null || true
tmux kill-session -t teacher 2>/dev/null || true
sleep 2

tmux new -d -s srv "cd ~/codelala; $BIN/llama-server -m $GGUF -ngl 99 -fa 1 -c 65536 -np 4 --host 127.0.0.1 --port 8090 > logs/server_${TAG}_quick.log 2>&1"
for _ in $(seq 1 90); do
  curl -s -m 3 http://127.0.0.1:8090/health | grep -q ok && break
  sleep 2
done

for EFF in medium disabled; do
  OUT="runs/${TAG}_quick/${EFF}"
  mkdir -p "$OUT"
  echo "=== 2. Quick webgen ($EFF): 4 held-out apps x 2 samples ==="
  $PY eval/gen.py --tasks "runs/${TAG}_quick/tasks4.jsonl" --out "$OUT/gen.jsonl" --samples 2 --effort "$EFF" --concurrency 4
  $PY eval/webcheck.py --gen "$OUT/gen.jsonl" --out "$OUT/web.jsonl"
  $PY -c '
import json
from pathlib import Path
gen = [json.loads(l) for l in Path("'"$OUT"'/gen.jsonl").read_text().splitlines() if l.strip()]
web = [json.loads(l) for l in Path("'"$OUT"'/web.jsonl").read_text().splitlines() if l.strip()]
smoke = sum(1 for w in web if not w.get("probe_error") and not w.get("js_errors") and w.get("nonblank") and (w.get("idle_change",0)>0.05 or w.get("input_change",0)>0.05))
js_err = sum(1 for w in web if w.get("js_errors"))
avg_tok = sum(g.get("tokens",0) for g in gen) / max(1, len(gen))
avg_plan = sum(len(g.get("reasoning") or "") for g in gen) / max(1, len(gen))
trunc = sum(1 for g in gen if g.get("finish") == "length")
print(f"QUICK_WEB ['"$EFF"']: smoke={smoke}/{len(web)} ({100*smoke/max(1,len(web)):.0f}%) js_err={js_err} trunc={trunc} avg_tok={avg_tok:.0f} avg_plan_chars={avg_plan:.0f}")
for w in web:
    wid = w.get("id")
    ws = w.get("sample")
    nb = w.get("nonblank")
    ic = w.get("idle_change", 0)
    inc = w.get("input_change", 0)
    errs = (w.get("js_errors") or [])[:2]
    print(f"  {wid}_{ws}: nonblank={nb} idle={ic:.2f} inp={inc:.2f} errs={errs}")
'
  echo "=== 3. Quick MBPP+ ($EFF): 20 Python tasks ==="
  $PY eval/eval_python.py --out "$OUT/mbpp20.jsonl" --url http://127.0.0.1:8090 --n 20 --effort "$EFF" --concurrency 4
done

tmux kill-session -t srv 2>/dev/null || true
echo "QUICK_EVAL_DONE $TAG"
