#!/usr/bin/env bash
# CODELALA v3 Post-SFT Pipeline (from early-stopped adapter_best at Step 60, eval=0.6673):
#   1. Surgical fp32 merge of linear rsLoRA + 3D Expert-LoRA -> runs/v3_sft/merged
#   2. Convert to GGUF bf16 + quantize Q6_K + run gguf_diff sanity check
#   3. Full apples-to-apples eval of v3_sft (12 held-out web apps + 60 MBPP+, disabled & medium)
#   4. Stage-2 On-Policy Execution-Verified DPO+RPO -> merge -> GGUF bf16 + Q6_K -> full eval
set -euo pipefail
cd "${CODELALA_DIR:-$(pwd)}"
[ -f train/env.sh ] && . train/env.sh
export NO_PROXY=127.0.0.1,localhost
export PYTHONUNBUFFERED=1

PY=.venv/bin/python
BIN_OLALA=src/llama.cpp-olala/build-cuda/bin
BIN_UPSTREAM=src/llama.cpp-upstream/build-cuda/bin
TEACHER=$HOME/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
LOG=logs/chain_v3_post_sft.log

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
# STEP 1: Surgical fp32 merge of adapter_best (Step 60, eval=0.6673)
# =====================================================================
log "STEP 1: Merging runs/v3_sft/adapter_best -> runs/v3_sft/merged"
rm -rf runs/v3_sft/merged
$PY -c "from pathlib import Path; import sys; sys.path.insert(0, 'train'); from olala_fast import merge_all_to_safetensors; merge_all_to_safetensors(Path('models/olala-hf'), Path('runs/v3_sft/adapter_best'), Path('runs/v3_sft/merged'))" | tee -a "$LOG"

# =====================================================================
# STEP 2: Convert to GGUF bf16 + Q6_K + gguf_diff check
# =====================================================================
log "STEP 2: Converting runs/v3_sft/merged -> GGUF bf16 + Q6_K"
$PY src/llama.cpp-olala/convert_hf_to_gguf.py runs/v3_sft/merged --outtype bf16 --outfile models/gguf/olala-v3-sft-bf16.gguf > logs/convert_v3_sft.log 2>&1
$BIN_OLALA/llama-quantize models/gguf/olala-v3-sft-bf16.gguf models/gguf/olala-v3-sft-q6_k.gguf Q6_K > logs/quant_v3_sft.log 2>&1
log "STEP 2 GGUF DIFF CHECK:"
$PY runs/kernel_test/gguf_diff.py models/gguf/Olala-v1.0-mini-7A1B-bf16.gguf models/gguf/olala-v3-sft-bf16.gguf | tee -a "$LOG"

# =====================================================================
# STEP 3: Full apples-to-apples evaluation of v3_sft
# =====================================================================
log "STEP 3: Evaluating v3_sft on 12 held-out web apps + 60 MBPP+ (disabled & medium)"
rm -rf runs/v3_sft_disabled runs/v3_sft_medium
bash eval/eval_all.sh v3_sft models/gguf/olala-v3-sft-bf16.gguf > logs/eval_v3_sft.log 2>&1
log "=== V3_SFT RESULTS ==="
cat runs/v3_sft_disabled/report.txt runs/v3_sft_disabled/mbpp_report.txt runs/v3_sft_medium/report.txt runs/v3_sft_medium/mbpp_report.txt | tee -a "$LOG"

# =====================================================================
# STEP 4: Stage-2 On-Policy Execution-Verified DPO+RPO
# =====================================================================
log "STEP 4A: Collecting on-policy DPO pairs from v3_sft (140 prompts x 4 rollouts)"
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
  --k 4 --limit 140 --student-conc 4 --teacher-conc 4 --ff-workers 4 \
  > logs/collect_dpo.log 2>&1
stop_servers
log "STEP 4A DONE: $(wc -l < data/v3/dpo_pairs.jsonl) DPO pairs collected"

log "STEP 4B: Training Stage-2 On-Policy DPO+RPO"
rm -rf runs/v3_dpo
$PY train/dpo_v3.py \
  --model-path runs/v3_sft/merged \
  --pairs-file data/v3/dpo_pairs.jsonl \
  --output-dir runs/v3_dpo \
  --epochs 1.5 --grad-accum 8 --lr 3.0e-5 --expert-lr 1.5e-5 --beta 2.5 --rpo-alpha 0.35 \
  --lora-r 32 --lora-alpha 64 --expert-r 8 --expert-alpha 16 --max-length 4096 \
  > logs/train_v3_dpo.log 2>&1

log "STEP 4C: Converting v3_dpo -> GGUF bf16 + Q6_K"
$PY src/llama.cpp-olala/convert_hf_to_gguf.py runs/v3_dpo/merged --outtype bf16 --outfile models/gguf/olala-v3-dpo-bf16.gguf > logs/convert_v3_dpo.log 2>&1
$BIN_OLALA/llama-quantize models/gguf/olala-v3-dpo-bf16.gguf models/gguf/olala-v3-dpo-q6_k.gguf Q6_K > logs/quant_v3_dpo.log 2>&1

log "STEP 5: Evaluating v3_dpo on 12 held-out web apps + 60 MBPP+ (disabled & medium)"
rm -rf runs/v3_dpo_disabled runs/v3_dpo_medium
bash eval/eval_all.sh v3_dpo models/gguf/olala-v3-dpo-bf16.gguf > logs/eval_v3_dpo.log 2>&1
log "=== V3_DPO RESULTS ==="
cat runs/v3_dpo_disabled/report.txt runs/v3_dpo_disabled/mbpp_report.txt runs/v3_dpo_medium/report.txt runs/v3_dpo_medium/mbpp_report.txt | tee -a "$LOG"
log "CHAIN_V3_POST_SFT_ALL_DONE"
