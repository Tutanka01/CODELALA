#!/usr/bin/env bash
# Calibrated CODELALA v3b autonomous pipeline:
#   Stage 0: Build clean offline dataset data/v3b (webapp_repeat=1, zero memorization)
#   Stage 1: Calibrated SFT (use_rslora=False, scale=2.0, lr=4e-5, expert_lr=5e-5, wd=0.05)
#   Gate 1:  gguf_diff + 3-min quick_eval (4 held-out web apps + 20 MBPP+)
#   Stage 1b: Full apples-to-apples eval (12 held-out web apps + 60 MBPP+, disabled & medium)
#   Stage 2: On-Policy Execution-Verified DPO+RPO (collect_dpo + dpo_v3 scale=2.0) + full eval
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
JUDGE=$HOME/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf

mkdir -p logs models/gguf runs

echo "========== [STAGE 0] Build data/v3b (offline, webapp_repeat=1) =========="
$PY data/build_v3.py \
  --synth data/v3/synth_v3.jsonl \
  --self-anchor data/v3/self_anchor.jsonl \
  --out-dir data/v3b \
  --n-kodcode 0 \
  --n-codefeedback 550 \
  --n-opus 180 \
  --webapp-repeat 1

echo "========== [STAGE 1] Calibrated SFT v3b =========="
tmux kill-session -t srv 2>/dev/null || true
tmux kill-session -t teacher 2>/dev/null || true
sleep 2

$PY train/sft_v3.py \
  --model-path models/olala-hf \
  --train-file data/v3b/train.jsonl \
  --eval-file data/v3b/eval.jsonl \
  --output-dir runs/v3b_sft \
  --epochs 2.0 \
  --grad-accum 16 \
  --lr 4.0e-5 \
  --expert-lr 5.0e-5 \
  --weight-decay 0.05 \
  --warmup-ratio 0.08 \
  --lora-r 32 \
  --lora-alpha 64 \
  --expert-r 16 \
  --expert-alpha 32 \
  --eval-steps 20 \
  --save-steps 30

echo "========== [STAGE 1 CONVERT] Merged HF -> GGUF bf16 + Q6_K =========="
$PY $CONV runs/v3b_sft/merged --outfile models/gguf/olala-v3b-sft-bf16.gguf --outtype bf16
$BIN/llama-quantize models/gguf/olala-v3b-sft-bf16.gguf models/gguf/olala-v3b-sft-q6_k.gguf Q6_K

echo "========== [GATE 1] Quick Sanity Eval (gguf_diff + 4 apps + 20 MBPP+) =========="
bash eval/quick_eval.sh v3b_sft models/gguf/olala-v3b-sft-bf16.gguf | tee runs/v3b_sft/quick_eval.log

# Verify gate: smoke >= 4/8 in either medium or disabled, AND no catastrophic JS collapse
GATE_OK=$($PY -c '
import json
from pathlib import Path
ok = False
for eff in ("medium", "disabled"):
    p = Path(f"runs/v3b_sft_quick/{eff}/web.jsonl")
    if not p.exists(): continue
    web = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    smoke = sum(1 for w in web if not w.get("probe_error") and not w.get("js_errors") and w.get("nonblank") and (w.get("idle_change",0)>0.05 or w.get("input_change",0)>0.05))
    if smoke >= 4:
        ok = True
print("1" if ok else "0")
')

if [ "$GATE_OK" != "1" ]; then
  echo "GATE_1_FAILED: stopping before full eval/DPO so we can inspect checkpoints!"
  exit 0
fi
echo "GATE_1_PASSED! Proceeding to full apples-to-apples eval and Stage-2 DPO+RPO!"

echo "========== [STAGE 1 EVAL] Full Apples-to-Apples Eval (v3b_sft) =========="
bash eval/eval_all.sh v3b_sft models/gguf/olala-v3b-sft-bf16.gguf

echo "========== [STAGE 2 DATA] On-Policy Execution-Verified DPO Rollouts =========="
tmux kill-session -t srv 2>/dev/null || true
tmux kill-session -t teacher 2>/dev/null || true
sleep 2

tmux new -d -s srv "cd ~/codelala; $BIN/llama-server -m models/gguf/olala-v3b-sft-bf16.gguf -ngl 99 -fa 1 -c 65536 -np 4 --host 127.0.0.1 --port 8090 > logs/server_dpo_stu.log 2>&1"
tmux new -d -s teacher "cd ~/codelala; $UBIN/llama-server -m $JUDGE -ngl 99 -fa 1 -c 131072 -np 6 --reasoning-budget 1536 --no-reasoning-preserve --host 127.0.0.1 --port 8091 > logs/server_dpo_tea.log 2>&1"

for PORT in 8090 8091; do
  for _ in $(seq 1 120); do
    curl -s -m 3 "http://127.0.0.1:${PORT}/health" | grep -q ok && break
    sleep 2
  done
done

$PY data/collect_dpo.py \
  --out data/v3b/dpo_pairs.jsonl \
  --synth data/v3/synth_v3.jsonl \
  --student-url http://127.0.0.1:8090 \
  --teacher-url http://127.0.0.1:8091 \
  --k 4 \
  --limit 140 \
  --max-tokens 5000 \
  --student-conc 4 \
  --teacher-conc 6 \
  --ff-workers 4

tmux kill-session -t srv 2>/dev/null || true
tmux kill-session -t teacher 2>/dev/null || true
sleep 3

N_PAIRS=$(wc -l < data/v3b/dpo_pairs.jsonl || echo 0)
echo "Collected $N_PAIRS DPO pairs."
if [ "$N_PAIRS" -lt 20 ]; then
  echo "Not enough DPO pairs ($N_PAIRS < 20), stopping cleanly after Stage 1!"
  exit 0
fi

echo "========== [STAGE 2 TRAIN] Calibrated Length-Normalized DPO + RPO =========="
$PY train/dpo_v3.py \
  --model-path runs/v3b_sft/merged \
  --pairs-file data/v3b/dpo_pairs.jsonl \
  --output-dir runs/v3b_dpo \
  --epochs 1.5 \
  --grad-accum 8 \
  --lr 2.0e-5 \
  --expert-lr 2.0e-5 \
  --weight-decay 0.05 \
  --beta 2.5 \
  --rpo-alpha 0.35 \
  --lora-r 32 \
  --lora-alpha 64 \
  --expert-r 8 \
  --expert-alpha 16

echo "========== [STAGE 2 CONVERT] DPO Merged HF -> GGUF bf16 + Q6_K =========="
$PY $CONV runs/v3b_dpo/merged --outfile models/gguf/olala-v3b-dpo-bf16.gguf --outtype bf16
$BIN/llama-quantize models/gguf/olala-v3b-dpo-bf16.gguf models/gguf/olala-v3b-dpo-q6_k.gguf Q6_K
$PY diag/gguf_diff.py models/gguf/olala-v3b-sft-bf16.gguf models/gguf/olala-v3b-dpo-bf16.gguf | head -n 30

echo "========== [STAGE 2 EVAL] Full Apples-to-Apples Eval (v3b_dpo) =========="
bash eval/eval_all.sh v3b_dpo models/gguf/olala-v3b-dpo-bf16.gguf

echo "ALL_V3B_PIPELINE_COMPLETE!"
