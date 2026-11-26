#!/usr/bin/env bash
# CODELALA v3 End-to-End Post-Training Pipeline — 100% on DGX Spark (GB10, 119 GB):
#   Phase 1A: 3-gate verified teacher distillation (Qwen3.6-35B-A3B-UD-Q5_K_XL, --reasoning-budget 2048)
#   Phase 1B: Olala-base self-anchor (79 rows) + MBPP+ Python baseline (skipped if already present)
#   Phase 1C: Build data/v3 (WebApps + KodCode-R1 + CodeFeedback + OPUS-100 + Self-Anchor)
#   Phase 2 : Stage-1 SFT (compiled Mamba-3 + rsLoRA r=64 + 3D Expert-LoRA r=16) -> GGUF -> full eval
#   Phase 3 : Stage-2 On-Policy Execution-Verified DPO+RPO -> GGUF (bf16 + q6_k) -> full eval
#
# Memory safety rule: NEVER run training alongside any llama-server.
set -euo pipefail
cd "${CODELALA_DIR:-$(pwd)}"
[ -f train/env.sh ] && . train/env.sh
export NO_PROXY=127.0.0.1,localhost
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
BIN_OLALA=src/llama.cpp-olala/build-cuda/bin
BIN_UPSTREAM=src/llama.cpp-upstream/build-cuda/bin
TEACHER=$HOME/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
BASE_GGUF=models/gguf/Olala-v1.0-mini-7A1B-bf16.gguf
LOG=logs/chain_v3.log

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

# =====================================================================
# PHASE 1A: Full 3-gate verified synthesis with Qwen3.6-35B-A3B-UD-Q5_K_XL
#           using --reasoning-budget 2048 so thinking NEVER truncates HTML/JSON!
# =====================================================================
log "PHASE 1A: Starting Qwen3.6-Q5 synthesis (231 unique prompts, max-retries=2, reasoning-budget=2560, resumable)"
stop_servers
tmux new -d -s teacher "cd ~/codelala; $BIN_UPSTREAM/llama-server -m $TEACHER --jinja --reasoning-budget 2560 --no-reasoning-preserve -ngl 99 -fa 1 -c 65536 -np 6 --host 127.0.0.1 --port 8091 > logs/server_teacher_v3.log 2>&1"
wait_health 8091
$PY data/synth_teacher.py \
  --out data/v3/synth_v3.jsonl \
  --url http://127.0.0.1:8091 \
  --variants 1 \
  --min-score 8 \
  --max-retries 2 \
  --concurrency 6 \
  --judge-concurrency 6 \
  --ff-workers 4 \
  --max-tokens 7000 \
  --temp 0.7 \
  > logs/synth_v3.log 2>&1
stop_servers
log "PHASE 1A DONE: $(wc -l < data/v3/synth_v3.jsonl) verified apps"

# =====================================================================
# PHASE 1B: Olala-base self-anchor + MBPP+ Python baseline (skip if done)
# =====================================================================
if [ ! -s data/v3/self_anchor.jsonl ] || [ ! -s runs/base_py/mbpp_medium.jsonl ]; then
  log "PHASE 1B: Collecting Olala-base self-anchor + MBPP+ baseline"
  tmux new -d -s srv "cd ~/codelala; $BIN_OLALA/llama-server -m $BASE_GGUF -ngl 99 -fa 1 -c 32768 -np 4 --host 127.0.0.1 --port 8090 > logs/server_base_anchor.log 2>&1"
  wait_health 8090
  [ -s data/v3/self_anchor.jsonl ] || $PY data/build_self_anchor.py --out data/v3/self_anchor.jsonl --url http://127.0.0.1:8090 --concurrency 4
  mkdir -p runs/base_py
  [ -s runs/base_py/mbpp_disabled.jsonl ] || $PY eval/eval_python.py --out runs/base_py/mbpp_disabled.jsonl --url http://127.0.0.1:8090 --n 60 --effort disabled --concurrency 4 | tee runs/base_py/mbpp_disabled_report.txt
  [ -s runs/base_py/mbpp_medium.jsonl ] || $PY eval/eval_python.py --out runs/base_py/mbpp_medium.jsonl --url http://127.0.0.1:8090 --n 60 --effort medium --concurrency 4 | tee runs/base_py/mbpp_medium_report.txt
  stop_servers
else
  log "PHASE 1B: Already complete (self_anchor=$(wc -l < data/v3/self_anchor.jsonl) rows, MBPP+ baseline done)"
fi

# =====================================================================
# PHASE 1C: Build CODELALA v3 SFT dataset
# =====================================================================
log "PHASE 1C: Building data/v3 train/eval datasets"
$PY data/build_v3.py \
  --synth data/v3/synth_v3.jsonl \
  --out-dir data/v3 \
  --self-anchor data/v3/self_anchor.jsonl \
  --max-tokens 4096 \
  --n-kodcode 650 \
  --n-codefeedback 550 \
  --n-opus 180 \
  --webapp-repeat 2 \
  > logs/build_v3.log 2>&1
cat data/v3/stats.json >> "$LOG"

# =====================================================================
# PHASE 2: Stage-1 SFT + Merge + GGUF + Full Apples-to-Apples Eval
# =====================================================================
log "PHASE 2: Starting Stage-1 SFT (rsLoRA r=64 + 3D Expert-LoRA r=16, 403M trainable params)"
$PY train/sft_v3.py \
  --model-path models/olala-hf \
  --train-file data/v3/train.jsonl \
  --eval-file data/v3/eval.jsonl \
  --output-dir runs/v3_sft \
  --epochs 2.0 \
  --grad-accum 16 \
  --lr 1.2e-4 \
  --expert-lr 6.0e-5 \
  --lora-r 64 --lora-alpha 128 \
  --expert-r 16 --expert-alpha 32 \
  --max-length 4096 \
  --eval-steps 35 \
  --save-steps 70 \
  > logs/train_v3_sft.log 2>&1
log "PHASE 2 SFT DONE; converting to GGUF bf16 + q6_k"
$PY src/llama.cpp-olala/convert_hf_to_gguf.py runs/v3_sft/merged --outtype bf16 --outfile models/gguf/olala-v3-sft-bf16.gguf > logs/convert_v3_sft.log 2>&1
$BIN_OLALA/llama-quantize models/gguf/olala-v3-sft-bf16.gguf models/gguf/olala-v3-sft-q6_k.gguf Q6_K > logs/quant_v3_sft.log 2>&1
log "PHASE 2 EVALUATING v3_sft on 12 held-out web apps + 60 MBPP+"
bash eval/eval_all.sh v3_sft models/gguf/olala-v3-sft-bf16.gguf > logs/eval_v3_sft.log 2>&1
cat runs/v3_sft_disabled/report.txt runs/v3_sft_medium/report.txt >> "$LOG"

# =====================================================================
# PHASE 3: Stage-2 On-Policy Execution-Verified DPO+RPO + Eval + Q6_K
# =====================================================================
log "PHASE 3A: Collecting on-policy DPO pairs from v3_sft"
stop_servers
tmux new -d -s srv "cd ~/codelala; $BIN_OLALA/llama-server -m models/gguf/olala-v3-sft-bf16.gguf -ngl 99 -fa 1 -c 65536 -np 4 --host 127.0.0.1 --port 8090 > logs/server_v3_sft_dpo.log 2>&1"
tmux new -d -s teacher "cd ~/codelala; $BIN_UPSTREAM/llama-server -m $TEACHER --jinja --reasoning-budget 1536 --no-reasoning-preserve -ngl 99 -fa 1 -c 65536 -np 4 --host 127.0.0.1 --port 8091 > logs/server_judge_dpo.log 2>&1"
wait_health 8090
wait_health 8091
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
cat runs/v3_dpo_disabled/report.txt runs/v3_dpo_medium/report.txt >> "$LOG"
log "CHAIN_V3_ALL_DONE"
