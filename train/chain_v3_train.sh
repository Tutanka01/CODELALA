#!/usr/bin/env bash
# CODELALA v3 Training & Evaluation Pipeline (Phases 1C -> 2 -> 3) on DGX Spark:
# Assumes Phase 1A (data/v3/synth_v3.jsonl = 153 verified 8-10/10 webapps) and
# Phase 1B (self_anchor + MBPP+ baseline) are already complete.
set -euo pipefail
cd "${CODELALA_DIR:-$(pwd)}"
[ -f train/env.sh ] && . train/env.sh
export NO_PROXY=127.0.0.1,localhost
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
BIN_OLALA=src/llama.cpp-olala/build-cuda/bin
BIN_UPSTREAM=src/llama.cpp-upstream/build-cuda/bin
TEACHER=$HOME/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
LOG=logs/chain_v3_train.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

stop_servers() {
  tmux kill-session -t srv 2>/dev/null || true
  tmux kill-session -t teacher 2>/dev/null || true
  sleep 3
}

wait_health() {
  for _ in $(seq 1 120); do
    curl -s -m 3 "http://127.0.0.1:$1/health" | grep -q ok && return 0
    sleep 2
  done
  log "ERROR: server on port $1 failed health check"
  return 1
}

stop_servers

# =====================================================================
# PHASE 1C: Rebuild data/v3 cleanly (~75% medium, ~25% low + disabled direct)
# =====================================================================
log "PHASE 1C: Rebuilding data/v3 (medium/low/disabled balanced, webapp-repeat=3, offline cache)"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 $PY data/build_v3.py \
  --synth data/v3/synth_v3.jsonl \
  --out-dir data/v3 \
  --self-anchor data/v3/self_anchor.jsonl \
  --max-tokens 4096 \
  --n-kodcode 0 \
  --n-codefeedback 550 \
  --n-opus 180 \
  --webapp-repeat 3 \
  > logs/build_v3.log 2>&1
cat data/v3/stats.json | tee -a "$LOG"

# =====================================================================
# PHASE 2: Stage-1 SFT + Merge + GGUF + Full Apples-to-Apples Eval
# =====================================================================
log "PHASE 2: Starting Stage-1 SFT (rsLoRA r=64 + 3D Expert-LoRA r=16, 2.5 epochs)"
rm -rf runs/v3_sft
$PY train/sft_v3.py \
  --model-path models/olala-hf \
  --train-file data/v3/train.jsonl \
  --eval-file data/v3/eval.jsonl \
  --output-dir runs/v3_sft \
  --epochs 2.5 \
  --grad-accum 16 \
  --lr 1.2e-4 \
  --expert-lr 6.0e-5 \
  --lora-r 64 --lora-alpha 128 \
  --expert-r 16 --expert-alpha 32 \
  --max-length 4096 \
  --eval-steps 30 \
  --save-steps 60 \
  > logs/train_v3_sft.log 2>&1
log "PHASE 2 SFT DONE; converting to GGUF bf16 + q6_k"
$PY src/llama.cpp-olala/convert_hf_to_gguf.py runs/v3_sft/merged --outtype bf16 --outfile models/gguf/olala-v3-sft-bf16.gguf > logs/convert_v3_sft.log 2>&1
$BIN_OLALA/llama-quantize models/gguf/olala-v3-sft-bf16.gguf models/gguf/olala-v3-sft-q6_k.gguf Q6_K > logs/quant_v3_sft.log 2>&1
log "PHASE 2 EVALUATING v3_sft on 12 held-out web apps + 60 MBPP+"
bash eval/eval_all.sh v3_sft models/gguf/olala-v3-sft-bf16.gguf > logs/eval_v3_sft.log 2>&1
cat runs/v3_sft_disabled/report.txt runs/v3_sft_medium/report.txt runs/v3_sft_disabled/mbpp_report.txt runs/v3_sft_medium/mbpp_report.txt | tee -a "$LOG"

# =====================================================================
# PHASE 3: Stage-2 On-Policy Execution-Verified DPO+RPO + Eval + Q6_K
# =====================================================================
log "PHASE 3A: Collecting on-policy DPO pairs from v3_sft"
stop_servers
tmux new -d -s srv "cd ~/codelala; $BIN_OLALA/llama-server -m models/gguf/olala-v3-sft-bf16.gguf -ngl 99 -fa 1 -c 65536 -np 4 --host 127.0.0.1 --port 8090 > logs/server_v3_sft_dpo.log 2>&1"
tmux new -d -s teacher "cd ~/codelala; $BIN_UPSTREAM/llama-server -m $TEACHER --jinja --reasoning-budget 1536 --no-reasoning-preserve -ngl 99 -fa 1 -c 65536 -np 4 --host 127.0.0.1 --port 8091 > logs/server_judge_dpo.log 2>&1"
wait_health 8090
wait_health 8091
rm -f data/v3/dpo_pairs.jsonl
$PY data/collect_dpo.py \
  --out data/v3/dpo_pairs.jsonl \
  --synth data/v3/synth_v3.jsonl \
  --student-url http://127.0.0.1:8090 \
  --teacher-url http://127.0.0.1:8091 \
  --k 4 --limit 160 --student-conc 4 --teacher-conc 4 --ff-workers 4 \
  > logs/collect_dpo.log 2>&1
stop_servers
log "PHASE 3A DONE: $(wc -l < data/v3/dpo_pairs.jsonl) DPO pairs collected"

log "PHASE 3B: Running Stage-2 DPO+RPO training"
rm -rf runs/v3_dpo
$PY train/dpo_v3.py \
  --model-path runs/v3_sft/merged \
  --pairs-file data/v3/dpo_pairs.jsonl \
  --output-dir runs/v3_dpo \
  --epochs 1.5 --grad-accum 8 --lr 3.0e-5 --expert-lr 1.5e-5 --beta 2.5 --rpo-alpha 0.35 \
  --lora-r 32 --lora-alpha 64 --expert-r 8 --expert-alpha 16 --max-length 4096 \
  > logs/train_v3_dpo.log 2>&1
log "PHASE 3B DPO DONE; converting to GGUF bf16 + q6_k"
$PY src/llama.cpp-olala/convert_hf_to_gguf.py runs/v3_dpo/merged --outtype bf16 --outfile models/gguf/olala-v3-dpo-bf16.gguf > logs/convert_v3_dpo.log 2>&1
$BIN_OLALA/llama-quantize models/gguf/olala-v3-dpo-bf16.gguf models/gguf/olala-v3-dpo-q6_k.gguf Q6_K > logs/quant_v3_dpo.log 2>&1
log "PHASE 3C EVALUATING v3_dpo on 12 held-out web apps + 60 MBPP+"
bash eval/eval_all.sh v3_dpo models/gguf/olala-v3-dpo-bf16.gguf > logs/eval_v3_dpo.log 2>&1
cat runs/v3_dpo_disabled/report.txt runs/v3_dpo_medium/report.txt runs/v3_dpo_disabled/mbpp_report.txt runs/v3_dpo_medium/mbpp_report.txt | tee -a "$LOG"
log "CHAIN_V3_ALL_DONE"
