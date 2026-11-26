#!/usr/bin/env bash
# Full reproducible evaluation suite for a GGUF checkpoint:
#   1. 12 held-out single-file web apps (3 samples each = 36) in disabled & medium modes
#      -> Headless Firefox runtime probe (webcheck.py) + strict Qwen3.6 QA judge (judge.py)
#   2. 60 MBPP+ Python execution tasks in disabled & medium modes
# Usage: eval/eval_all.sh <tag> <gguf_path>
set -euo pipefail
TAG="$1"
GGUF="$2"
cd "${CODELALA_DIR:-$(pwd)}"
export NO_PROXY=127.0.0.1,localhost
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
BIN=src/llama.cpp-olala/build-cuda/bin
UBIN=src/llama.cpp-upstream/build-cuda/bin
# Keep the EXACT same judge model as RAPPORT.md for 100% apples-to-apples comparison!
JUDGE=$HOME/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf

wait_health() {
  for _ in $(seq 1 120); do
    curl -s -m 3 "http://127.0.0.1:$1/health" | grep -q ok && return 0
    sleep 2
  done
  echo "Server on port $1 failed to start!" >&2
  return 1
}

tmux kill-session -t srv 2>/dev/null || true
tmux kill-session -t teacher 2>/dev/null || true
sleep 2

tmux new -d -s srv "cd ~/codelala; $BIN/llama-server -m $GGUF -ngl 99 -fa 1 -c 65536 -np 4 --host 127.0.0.1 --port 8090 > logs/server_${TAG}.log 2>&1"
tmux new -d -s teacher "cd ~/codelala; $UBIN/llama-server -m $JUDGE -ngl 99 -fa 1 -c 131072 -np 6 --reasoning-budget 2048 --no-reasoning-preserve --host 127.0.0.1 --port 8091 > logs/server_judge_${TAG}.log 2>&1"
wait_health 8090
wait_health 8091

for EFF in disabled medium; do
  RUN="${TAG}_${EFF}"
  echo "=== Running webgen eval: $RUN ==="
  $PY eval/gen.py --tasks eval/tasks_webgen.jsonl --out "runs/${RUN}/gen.jsonl" --samples 3 --effort "$EFF" --concurrency 4
  $PY eval/webcheck.py --gen "runs/${RUN}/gen.jsonl" --out "runs/${RUN}/web.jsonl"
  $PY eval/judge.py --gen "runs/${RUN}/gen.jsonl" --web "runs/${RUN}/web.jsonl" --out "runs/${RUN}/judge.jsonl" --url http://127.0.0.1:8091 --concurrency 6
  $PY eval/report.py "runs/${RUN}" | tee "runs/${RUN}/report.txt"
  echo "=== Running MBPP+ eval: $RUN ==="
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 $PY eval/eval_python.py --out "runs/${RUN}/mbpp.jsonl" --url http://127.0.0.1:8090 --n 60 --effort "$EFF" --concurrency 4 | tee "runs/${RUN}/mbpp_report.txt"
done

tmux kill-session -t srv 2>/dev/null || true
tmux kill-session -t teacher 2>/dev/null || true
echo "EVAL_ALL_DONE $TAG"
