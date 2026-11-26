"""Stage 2 On-Policy Execution-Verified DPO + RPO trainer for CODELALA v3.

Takes (prompt, chosen, rejected) pairs collected from Stage-1 Olala-SFT rollouts verified by
real headless Firefox execution + QA judge (and Python unit tests):
  1. Precomputes exact reference average log-probabilities under the Stage-1 merged model
     (so only ONE model lives in VRAM during training, keeping peak memory ~20 GB).
  2. Trains length-normalized DPO + RPO (NLL anchor on chosen) with rsLoRA + 3D Expert-LoRA:
       L = -log sigmoid(beta * ((logp_w - ref_w) - (logp_l - ref_l))) + rpo_alpha * nll_w
     Length-normalized log-probs prevent length inflation and kill repetition loops!
  3. Surgically merges back into Stage-1 safetensors in fp32.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ["OLALA_TORCH_MAMBA3"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from olala_fast import (
    DEFAULT_LINEAR_TARGETS,
    SAFE_LINEAR_TARGETS,
    attach_expert_lora,
    bucket_length,
    merge_all_to_safetensors,
    patch_fast_mamba3,
    save_expert_lora,
)


def encode_side(tok, messages, effort: str, max_length: int):
    enc = tok.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        reasoning_effort=effort,
    )
    ids = enc["input_ids"]
    mask = enc["assistant_masks"]
    labels = [t if m else -100 for t, m in zip(ids, mask)]
    if len(ids) > max_length or all(l == -100 for l in labels):
        return None
    return ids, labels


def pad_pair_side(ids, labels, pad_id: int, buckets):
    target = bucket_length(len(ids), buckets)
    pad = target - len(ids)
    if pad > 0:
        ids = ids + [pad_id] * pad
        labels = labels + [-100] * pad
    return (
        torch.tensor([ids], dtype=torch.long, device="cuda"),
        torch.tensor([labels], dtype=torch.long, device="cuda"),
    )


def avg_logprob(model, inp: torch.Tensor, lbl: torch.Tensor):
    """Returns (mean_logp, nll_loss) over non-masked assistant tokens."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inp).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = lbl[:, 1:]
    valid = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid, 0)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    tok_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    tok_lp = tok_lp * valid
    n_tok = valid.sum().clamp_min(1)
    mean_lp = tok_lp.sum() / n_tok
    return mean_lp, -mean_lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Stage 1 SFT merged model directory")
    ap.add_argument("--pairs-file", required=True, help="JSONL with {chosen_messages, rejected_messages, effort}")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=1.5)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--expert-lr", type=float, default=2.0e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=2.5, help="DPO temperature on length-normalized avg logprobs")
    ap.add_argument("--rpo-alpha", type=float, default=0.35, help="NLL anchor weight on chosen")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--use-rslora", action="store_true")
    ap.add_argument("--targets", choices=["safe", "all"], default="safe", help="safe=OVH 7 targets (freezes Mamba-3 in_proj_dyn)")
    ap.add_argument("--expert-r", type=int, default=8)
    ap.add_argument("--expert-alpha", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dpo_args.json").write_text(json.dumps(vars(args), indent=2))

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    buckets = tuple(b for b in (512, 1024, 2048, 3072, 4096) if b <= args.max_length)

    raw = [json.loads(l) for l in Path(args.pairs_file).read_text().splitlines() if l.strip()]
    pairs = []
    for r in raw:
        eff = r.get("effort", "medium")
        c = encode_side(tok, r["chosen_messages"], eff, args.max_length)
        j = encode_side(tok, r["rejected_messages"], eff, args.max_length)
        if c is not None and j is not None:
            pairs.append({"chosen": c, "rejected": j, "id": r.get("id", "")})
    print(f"Loaded {len(pairs)} valid DPO pairs (from {len(raw)} raw)", flush=True)
    if not pairs:
        raise SystemExit("No valid DPO pairs!")

    print(f"Loading Stage-1 model {args.model_path} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False
    patch_fast_mamba3(model, block=32, compile_fn=True)

    # 1. Precompute exact reference avg log-probs ONCE (zero extra VRAM during training!)
    print("Precomputing reference log-probs ...", flush=True)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(pairs):
            ci, cl = pad_pair_side(*p["chosen"], pad_id, buckets)
            ri, rl = pad_pair_side(*p["rejected"], pad_id, buckets)
            p["ref_w"] = avg_logprob(model, ci, cl)[0].item()
            p["ref_l"] = avg_logprob(model, ri, rl)[0].item()
            if (i + 1) % 50 == 0 or i + 1 == len(pairs):
                print(f"  ref {i+1}/{len(pairs)} ({time.time()-t0:.1f}s)", flush=True)

    # 2. Attach adapters for DPO stage
    leaf_set = set(SAFE_LINEAR_TARGETS if args.targets == "safe" else DEFAULT_LINEAR_TARGETS)
    linear_names = [
        n for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in leaf_set
    ]
    print(f"Target set '{args.targets}': {len(linear_names)} linear modules (in_proj_dyn included: {'in_proj_dyn' in leaf_set})", flush=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            use_rslora=args.use_rslora, target_modules=linear_names, bias="none", task_type="CAUSAL_LM",
        ),
    )
    attach_expert_lora(model, r=args.expert_r, alpha=args.expert_alpha)
    lin_scale = args.lora_alpha / math.sqrt(args.lora_r) if args.use_rslora else args.lora_alpha / args.lora_r
    exp_scale = (args.expert_alpha / args.expert_r) if args.expert_r > 0 else 0.0
    print(f"DPO adapters: linear scale={lin_scale:.2f} (rslora={args.use_rslora}), expert scale={exp_scale:.2f}", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    linear_params, expert_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "mlp.experts." in n and ("lora_A" in n or "lora_B" in n):
            expert_params.append(p)
        else:
            linear_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": linear_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": expert_params, "lr": args.expert_lr, "weight_decay": args.weight_decay},
        ],
        betas=(0.9, 0.95),
        fused=True,
    )

    rnd = random.Random(args.seed)
    total_items = int(len(pairs) * args.epochs)
    order = []
    for _ in range(math.ceil(args.epochs)):
        idxs = list(range(len(pairs)))
        rnd.shuffle(idxs)
        order.extend(idxs)
    order = order[:total_items]

    total_steps = max(1, math.ceil(len(order) / args.grad_accum))
    sched = get_cosine_schedule_with_warmup(opt, max(3, int(total_steps * 0.08)), total_steps)
    print(f"DPO+RPO training: {len(order)} pair steps -> {total_steps} optimizer steps (beta={args.beta}, rpo_alpha={args.rpo_alpha})", flush=True)

    model.train()
    opt.zero_grad(set_to_none=True)
    acc_loss = acc_margin = acc_acc = 0.0
    micro = step = 0
    st0 = time.time()
    hist = []

    for idx in order:
        p = pairs[idx]
        ci, cl = pad_pair_side(*p["chosen"], pad_id, buckets)
        ri, rl = pad_pair_side(*p["rejected"], pad_id, buckets)

        lp_w, nll_w = avg_logprob(model, ci, cl)
        lp_l, _ = avg_logprob(model, ri, rl)

        chosen_reward = args.beta * (lp_w - p["ref_w"])
        rejected_reward = args.beta * (lp_l - p["ref_l"])
        margin = chosen_reward - rejected_reward
        dpo_loss = -F.logsigmoid(margin)
        loss = (dpo_loss + args.rpo_alpha * nll_w) / args.grad_accum
        loss.backward()

        acc_loss += loss.item()
        acc_margin += margin.item() / args.grad_accum
        acc_acc += float(margin.item() > 0) / args.grad_accum
        micro += 1

        if micro % args.grad_accum == 0 or micro == len(order):
            gn = torch.nn.utils.clip_grad_norm_(linear_params + expert_params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            dt = time.time() - st0
            st0 = time.time()
            rec = {"step": step, "loss": round(acc_loss, 4), "margin": round(acc_margin, 4), "pref_acc": round(acc_acc, 3), "grad_norm": round(gn, 3), "sec": round(dt, 1)}
            hist.append(rec)
            print(f"[DPO step {step:3d}/{total_steps}] loss={acc_loss:.4f} margin={acc_margin:+.3f} pref_acc={acc_acc:.2f} gn={gn:.3f} ({dt:.1f}s)", flush=True)
            acc_loss = acc_margin = acc_acc = 0.0

    adapter_dir = out / "adapter"
    model.save_pretrained(str(adapter_dir))
    save_expert_lora(model, adapter_dir)
    tok.save_pretrained(str(adapter_dir))
    (out / "dpo_log.json").write_text(json.dumps(hist, indent=2))

    del model, opt
    torch.cuda.empty_cache()
    merge_all_to_safetensors(Path(args.model_path), adapter_dir, out / "merged")
    print("DPO_DONE", flush=True)


if __name__ == "__main__":
    main()
