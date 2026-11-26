#!/usr/bin/env bash
# Resume of the v2 run after the training process was lost (host ran out of memory):
#   LoRA training resumes from the last checkpoint in runs/v2/checkpoints -> merge -> GGUF bf16
#   -> eval of the fine-tuned model (without reasoning, then medium reasoning).
# Progress and reports are appended to logs/chain_v2.log.
cd ~/codelala || exit 1
. train/env.sh
# Block size of the pure-PyTorch Mamba-3 scan (see models/olala-hf/modeling_olala.py).
# 128 gives the same result as the default 64 to bf16 noise and is ~1.45x faster.
export OLALA_TORCH_MAMBA3_BLOCK=128
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
BIN=src/llama.cpp-olala/build-cuda/bin
TEACHER=$HOME/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a logs/chain_v2.log; }

wait_health() {
  for _ in $(seq 1 120); do
    curl -s -m 3 "http://127.0.0.1:$1/health" | grep -q ok && return 0
    sleep 3
  done
  return 1
}

eval_run() { # <run name> <reasoning effort>
  $PY eval/gen.py --tasks eval/tasks_webgen.jsonl --out "runs/$1/gen.jsonl" --samples 3 --effort "$2" --concurrency 4 > "logs/$1_gen.log" 2>&1
  $PY eval/webcheck.py --gen "runs/$1/gen.jsonl" --out "runs/$1/web.jsonl" > "logs/$1_web.log" 2>&1
  $PY eval/judge.py --gen "runs/$1/gen.jsonl" --web "runs/$1/web.jsonl" --out "runs/$1/judge.jsonl" > "logs/$1_judge.log" 2>&1
  $PY eval/report.py "runs/$1" > "runs/$1/report.txt" 2>&1
  log "eval $1 done"
  cat "runs/$1/report.txt" >> logs/chain_v2.log
}

log "resuming training from $(ls -d runs/v2/checkpoints/checkpoint-* | sort -t- -k2 -n | tail -1)"
$PY src/ovhai_llm_sft_tutorial/sft/sft_lora.py --config train/v2.yaml > logs/train_v2_resume.log 2>&1 || { log "training failed"; exit 1; }
log "training done"

for f in chat_template.jinja config.json configuration_olala.py modeling_olala.py generation_config.json tokenizer.json tokenizer_config.json; do
  [ -e "runs/v2/merged/$f" ] || cp "models/olala-hf/$f" "runs/v2/merged/$f"
done
$PY src/llama.cpp-olala/convert_hf_to_gguf.py runs/v2/merged --outtype bf16 --outfile models/gguf/olala-v2-ft-bf16.gguf > logs/convert_v2.log 2>&1 || { log "GGUF conversion failed"; exit 1; }
log "GGUF written"

tmux new -d -s srv "cd ~/codelala; $BIN/llama-server -m models/gguf/olala-v2-ft-bf16.gguf -ngl 99 -fa 1 -c 65536 -np 4 --host 127.0.0.1 --port 8090 > logs/server_v2.log 2>&1"
tmux new -d -s teacher "cd ~/codelala; $BIN/llama-server -m $TEACHER -ngl 99 -fa 1 -c 131072 -np 8 --host 127.0.0.1 --port 8091 > logs/server_teacher.log 2>&1"
wait_health 8090 && wait_health 8091 || { log "servers failed to start"; exit 1; }

eval_run v2_disabled disabled
eval_run v2_medium medium
log "CHAIN_DONE"
