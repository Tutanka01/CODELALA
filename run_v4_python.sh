#!/usr/bin/env bash
# CODELALA v4 — 100% Execution-Verified On-Policy Python DPO+RPO (ZERO in_proj_dyn!)
#
# Strict zero-contamination design:
#   - ALL 378 evalplus/mbppplus tasks are 100% strictly held out (never touched in training!)
#   - Stage 0: Olala-base deterministic greedy baseline on 150 MBPP & MBPP+ tasks (medium & disabled)
#   - Stage 1: Synthesize 260 original unit-test-verified Python problems (Qwen3.6, zero overlap)
#   - Stage 2: Collect K=6 on-policy Olala-base rollouts -> real Python subprocess execution -> (PASS, FAIL) pairs
#   - Stage 3: Train calibrated DPO+RPO from Olala-base with --targets safe (in_proj_dyn FROZEN!) + ExpertLoRA3D
#   - Stage 4: Merge -> GGUF bf16 + Q6_K -> gguf_diff check -> 150-task MBPP & MBPP+ eval + stability/loop check
set -euo pipefail
cd ~/codelala

export NO_PROXY=127.0.0.1,localhost
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PY=.venv/bin/python
BIN=src/llama.cpp-olala/build-cuda/bin
UBIN=src/llama.cpp-upstream/build-cuda/bin
CONV=src/llama.cpp-olala/convert_hf_to_gguf.py
BASE_GGUF=models/gguf/Olala-v1.0-mini-7A1B-bf16.gguf
JUDGE=$HOME/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf

mkdir -p logs data/v4 runs/v4_base runs/v4_py_dpo runs/v4_eval models/gguf

wait_health() {
  for _ in $(seq 1 120); do
    curl -s -m 3 "http://127.0.0.1:$1/health" | grep -q ok && return 0
    sleep 2
  done
  echo "Server on port $1 failed to start!" >&2
  return 1
}

stop_servers() {
  tmux kill-session -t srv 2>/dev/null || true
  tmux kill-session -t teacher 2>/dev/null || true
  sleep 2
}

echo "========== [STAGE 0] Olala-base 150-task MBPP & MBPP+ Baseline (greedy T=0.0) =========="
if [ -f runs/v4_base/report_medium.txt ] && [ -f runs/v4_base/report_disabled.txt ] && [ -f runs/v4_base/stability.txt ]; then
  echo "Stage 0 baseline already completed — skipping straight to Stage 1!"
  cat runs/v4_base/report_medium.txt runs/v4_base/report_disabled.txt
else
  stop_servers
  tmux new -d -s srv "cd ~/codelala; $BIN/llama-server -m $BASE_GGUF -ngl 99 -fa 1 -c 32768 -np 4 --host 127.0.0.1 --port 8090 > logs/server_v4_base.log 2>&1"
  wait_health 8090

  for EFF in medium disabled; do
    $PY eval/eval_python.py \
      --out "runs/v4_base/mbpp150_${EFF}.jsonl" \
      --url http://127.0.0.1:8090 \
      --start 0 --n 150 --effort "$EFF" --temperature 0.0 --full-plus --concurrency 4 \
      | tee "runs/v4_base/report_${EFF}.txt"
  done
  $PY eval/chat_loop_check.py --url http://127.0.0.1:8090 --out runs/v4_base/stability.jsonl | tee runs/v4_base/stability.txt
fi

echo "========== [STAGE 1] Synthesize 260 Execution-Verified Python Tasks (Qwen3.6) =========="
stop_servers
tmux new -d -s teacher "cd ~/codelala; $UBIN/llama-server -m $JUDGE -ngl 99 -fa 1 -c 65536 -np 6 --reasoning-budget 1536 --no-reasoning-preserve --host 127.0.0.1 --port 8091 > logs/server_v4_synth.log 2>&1"
wait_health 8091

$PY data/synth_python_v4.py \
  --out data/v4/python_tasks.jsonl \
  --url http://127.0.0.1:8091 \
  --concurrency 6

stop_servers
N_TASKS=$(wc -l < data/v4/python_tasks.jsonl || echo 0)
echo "Synthesized $N_TASKS verified Python tasks."
if [ "$N_TASKS" -lt 80 ]; then
  echo "ERROR: Too few verified Python tasks ($N_TASKS < 80)!" >&2
  exit 1
fi

echo "========== [STAGE 2] Collect On-Policy Execution-Verified DPO Pairs from Olala-base =========="
tmux new -d -s srv "cd ~/codelala; $BIN/llama-server -m $BASE_GGUF -ngl 99 -fa 1 -c 32768 -np 4 --host 127.0.0.1 --port 8090 > logs/server_v4_rollout.log 2>&1"
wait_health 8090

$PY data/collect_dpo_python.py \
  --tasks data/v4/python_tasks.jsonl \
  --out data/v4/dpo_pairs.jsonl \
  --url http://127.0.0.1:8090 \
  --k 6 \
  --max-tokens 3072 \
  --concurrency 4

stop_servers
N_PAIRS=$(wc -l < data/v4/dpo_pairs.jsonl || echo 0)
echo "Collected $N_PAIRS execution-verified Python DPO pairs."
if [ "$N_PAIRS" -lt 40 ]; then
  echo "ERROR: Too few DPO pairs ($N_PAIRS < 40)!" >&2
  exit 1
fi

echo "========== [STAGE 3] Train Calibrated DPO+RPO (NO in_proj_dyn, scale=2.0) =========="
$PY train/dpo_v3.py \
  --model-path models/olala-hf \
  --pairs-file data/v4/dpo_pairs.jsonl \
  --output-dir runs/v4_py_dpo \
  --targets safe \
  --epochs 2.0 \
  --grad-accum 8 \
  --lr 3.0e-5 \
  --expert-lr 3.0e-5 \
  --weight-decay 0.05 \
  --beta 2.5 \
  --rpo-alpha 0.35 \
  --lora-r 32 \
  --lora-alpha 64 \
  --expert-r 8 \
  --expert-alpha 16 \
  --max-length 3072

echo "========== [STAGE 4] Convert GGUF + gguf_diff + 150-task Held-Out Eval =========="
$PY $CONV runs/v4_py_dpo/merged --outfile models/gguf/olala-v4-py-dpo-bf16.gguf --outtype bf16
$BIN/llama-quantize models/gguf/olala-v4-py-dpo-bf16.gguf models/gguf/olala-v4-py-dpo-q6_k.gguf Q6_K
$PY runs/kernel_test/gguf_diff.py "$BASE_GGUF" models/gguf/olala-v4-py-dpo-bf16.gguf | tee runs/v4_eval/gguf_diff.txt

tmux new -d -s srv "cd ~/codelala; $BIN/llama-server -m models/gguf/olala-v4-py-dpo-bf16.gguf -ngl 99 -fa 1 -c 32768 -np 4 --host 127.0.0.1 --port 8090 > logs/server_v4_eval.log 2>&1"
wait_health 8090

for EFF in medium disabled; do
  $PY eval/eval_python.py \
    --out "runs/v4_eval/mbpp150_${EFF}.jsonl" \
    --url http://127.0.0.1:8090 \
    --start 0 --n 150 --effort "$EFF" --temperature 0.0 --full-plus --concurrency 4 \
    | tee "runs/v4_eval/report_${EFF}.txt"
done
$PY eval/chat_loop_check.py --url http://127.0.0.1:8090 --out runs/v4_eval/stability.jsonl | tee runs/v4_eval/stability.txt

stop_servers

echo "==================== FINAL SUMMARY (BASE vs CODELALA V4 PYTHON DPO) ===================="
echo "--- BASE ---"
cat runs/v4_base/report_medium.txt runs/v4_base/report_disabled.txt
grep STABILITY_CHECK runs/v4_base/stability.txt
echo "--- CODELALA V4 ---"
cat runs/v4_eval/report_medium.txt runs/v4_eval/report_disabled.txt
grep STABILITY_CHECK runs/v4_eval/stability.txt
echo "ALL_V4_PYTHON_COMPLETE!"
