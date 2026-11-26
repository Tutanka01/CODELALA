"""Stage 1 SFT trainer for CODELALA v3:
  - Compiled chunk-parallel Mamba-3 SSD scan (~6-8x faster on GB10)
  - High-rank rsLoRA (r=64) on all 13 linear targets + 3D Expert-LoRA (r=16) on all 256 routed experts
  - Fixed-bucket padding [1024, 2048, 3072, 4096] for zero recompilations
  - Step-0 and periodic per-bucket validation loss logging
  - Exact fp32 surgical safetensors merge preserving all base fp32 scalars byte-identical
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["OLALA_TORCH_MAMBA3"] = "1"

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from olala_fast import (
    DEFAULT_LINEAR_TARGETS,
    ExpertLoRA3D,
    attach_expert_lora,
    bucket_length,
    merge_all_to_safetensors,
    patch_fast_mamba3,
    save_expert_lora,
)


def tokenize_jsonl(path: Path, tok, max_length: int):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = []
    dropped = 0
    for r in rows:
        kwargs = r.get("chat_template_kwargs") or {}
        enc = tok.apply_chat_template(
            r["messages"],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **kwargs,
        )
        ids = enc["input_ids"]
        mask = enc["assistant_masks"]
        labels = [t if m else -100 for t, m in zip(ids, mask)]
        if len(ids) > max_length or all(l == -100 for l in labels):
            dropped += 1
            continue
        bucket_name = (r.get("meta") or {}).get("bucket", "unknown")
        out.append({"input_ids": ids, "labels": labels, "bucket": bucket_name})
    print(f"Loaded {path.name}: {len(out)} rows kept, {dropped} dropped (max_len={max_length})", flush=True)
    return out


def pad_to_bucket(item, pad_id: int, buckets=(1024, 2048, 3072, 4096)):
    ids = item["input_ids"]
    labels = item["labels"]
    target = bucket_length(len(ids), buckets)
    pad = target - len(ids)
    if pad > 0:
        ids = ids + [pad_id] * pad
        labels = labels + [-100] * pad
    return (
        torch.tensor([ids], dtype=torch.long, device="cuda"),
        torch.tensor([labels], dtype=torch.long, device="cuda"),
    )


@torch.no_grad()
def evaluate(model, eval_rows, pad_id: int, buckets):
    model.eval()
    by_bucket = collections.defaultdict(list)
    all_losses = []
    for item in eval_rows:
        inp, lbl = pad_to_bucket(item, pad_id, buckets)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=inp, labels=lbl).loss.item()
        if math.isfinite(loss):
            all_losses.append(loss)
            by_bucket[item["bucket"]].append(loss)
    model.train()
    summary = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_bucket.items())}
    mean_loss = round(sum(all_losses) / max(1, len(all_losses)), 4)
    return mean_loss, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/olala-hf")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4.0e-5)
    ap.add_argument("--expert-lr", type=float, default=5.0e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.08)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--use-rslora", action="store_true", help="Enable rsLoRA (alpha/sqrt(r)); off by default for stability")
    ap.add_argument("--expert-r", type=int, default=16)
    ap.add_argument("--expert-alpha", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--save-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2))

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    train_rows = tokenize_jsonl(Path(args.train_file), tok, args.max_length)
    eval_rows = tokenize_jsonl(Path(args.eval_file), tok, args.max_length)
    buckets = tuple(b for b in (1024, 2048, 3072, 4096) if b <= args.max_length)

    print(f"Loading base model {args.model_path} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False
    patch_fast_mamba3(model, block=32, compile_fn=True)

    # Attach calibrated LoRA on all 13 linear projection types
    leaf_set = set(DEFAULT_LINEAR_TARGETS)
    linear_names = [
        n for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in leaf_set
    ]
    lin_scale = args.lora_alpha / math.sqrt(args.lora_r) if args.use_rslora else args.lora_alpha / args.lora_r
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        use_rslora=args.use_rslora,
        target_modules=linear_names,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    print(f"Attached Linear LoRA to {len(linear_names)} modules (r={args.lora_r}, alpha={args.lora_alpha}, rslora={args.use_rslora}, scale={lin_scale:.2f})", flush=True)

    # Attach 3D Expert-LoRA on all 256 routed experts per layer
    n_exp = attach_expert_lora(model, r=args.expert_r, alpha=args.expert_alpha)
    exp_scale = (args.expert_alpha / args.expert_r) if args.expert_r > 0 else 0.0
    print(f"Attached 3D Expert-LoRA to {n_exp} ParallelExperts tensors (r={args.expert_r}, alpha={args.expert_alpha}, scale={exp_scale:.2f})", flush=True)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # Parameter groups: separate LR for linear rsLoRA vs routed 3D Expert-LoRA
    linear_params, expert_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "mlp.experts." in n and ("lora_A" in n or "lora_B" in n):
            expert_params.append(p)
        else:
            linear_params.append(p)

    n_lin = sum(p.numel() for p in linear_params)
    n_ex = sum(p.numel() for p in expert_params)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: linear={n_lin:,} + expert3d={n_ex:,} = {n_lin+n_ex:,} / {n_tot:,} ({100*(n_lin+n_ex)/n_tot:.2f}%)", flush=True)

    opt = torch.optim.AdamW(
        [
            {"params": linear_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": expert_params, "lr": args.expert_lr, "weight_decay": args.weight_decay},
        ],
        betas=(0.9, 0.95),
        fused=True,
    )

    total_samples = int(len(train_rows) * args.epochs)
    total_steps = max(1, math.ceil(total_samples / args.grad_accum))
    warmup_steps = max(5, int(total_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    print(f"Training plan: {total_samples} samples, grad_accum={args.grad_accum}, total_steps={total_steps}, warmup={warmup_steps}", flush=True)

    # Step-0 baseline eval loss!
    t0 = time.time()
    eval0, by_b0 = evaluate(model, eval_rows, pad_id, buckets)
    print(f"[step 0/{total_steps}] eval_loss={eval0:.4f} by_bucket={by_b0} ({time.time()-t0:.1f}s)", flush=True)
    log_hist = [{"step": 0, "eval_loss": eval0, "eval_by_bucket": by_b0}]
    best_eval = eval0

    import random
    rnd = random.Random(args.seed)
    order = []
    for ep in range(math.ceil(args.epochs)):
        idxs = list(range(len(train_rows)))
        rnd.shuffle(idxs)
        order.extend(idxs)
    order = order[:total_samples]

    model.train()
    opt.zero_grad(set_to_none=True)
    running_loss = 0.0
    micro = 0
    step = 0
    step_t0 = time.time()

    for idx in order:
        item = train_rows[idx]
        inp, lbl = pad_to_bucket(item, pad_id, buckets)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=inp, labels=lbl).loss / args.grad_accum
        loss.backward()
        running_loss += loss.item()
        micro += 1

        if micro % args.grad_accum == 0 or micro == len(order):
            grad_norm = torch.nn.utils.clip_grad_norm_(linear_params + expert_params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            dt = time.time() - step_t0
            step_t0 = time.time()
            lr_now = sched.get_last_lr()[0]
            rec = {"step": step, "loss": round(running_loss, 4), "grad_norm": round(grad_norm, 4), "lr": lr_now, "sec": round(dt, 1)}
            log_hist.append(rec)
            print(f"[step {step:4d}/{total_steps}] loss={running_loss:.4f} grad_norm={grad_norm:.3f} lr={lr_now:.2e} ({dt:.1f}s)", flush=True)
            running_loss = 0.0

            if step % args.eval_steps == 0 or step == total_steps:
                ev, by_b = evaluate(model, eval_rows, pad_id, buckets)
                print(f"[step {step:4d}/{total_steps}] EVAL loss={ev:.4f} (best={min(best_eval, ev):.4f}) by_bucket={by_b}", flush=True)
                log_hist.append({"step": step, "eval_loss": ev, "eval_by_bucket": by_b})
                (out / "trainer_log.json").write_text(json.dumps(log_hist, indent=2))
                if ev < best_eval:
                    best_eval = ev
                    best_dir = out / "adapter_best"
                    model.save_pretrained(str(best_dir))
                    save_expert_lora(model, best_dir)
                    tok.save_pretrained(str(best_dir))

            if step % args.save_steps == 0 and step < total_steps:
                ckpt_dir = out / f"adapter_step{step}"
                model.save_pretrained(str(ckpt_dir))
                save_expert_lora(model, ckpt_dir)

    final_dir = out / "adapter_final"
    model.save_pretrained(str(final_dir))
    save_expert_lora(model, final_dir)
    tok.save_pretrained(str(final_dir))
    (out / "trainer_log.json").write_text(json.dumps(log_hist, indent=2))
    print(f"SFT_DONE best_eval={best_eval:.4f}", flush=True)

    # Free GPU memory before surgical safetensors merge
    del model, opt
    torch.cuda.empty_cache()

    if not args.no_merge:
        merge_src = out / "adapter_best" if (out / "adapter_best").exists() else final_dir
        print(f"Merging {merge_src} into {out / 'merged'} ...", flush=True)
        merge_all_to_safetensors(Path(args.model_path), merge_src, out / "merged")


if __name__ == "__main__":
    main()
