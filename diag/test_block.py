"""Speed and equivalence of the pure-PyTorch Mamba-3 fallback for several block sizes,
on one randomly initialised OlalaMamba3MimoFast layer (forward + backward)."""
import os
import sys
import time

os.environ["OLALA_TORCH_MAMBA3"] = "1"
import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL = "models/olala-hf"
L = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
BLOCKS = [int(b) for b in sys.argv[2].split(",")] if len(sys.argv) > 2 else [64, 128, 256, 512]

cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
cls = get_class_from_dynamic_module("modeling_olala.OlalaMamba3MimoFast", MODEL)
mod = sys.modules[cls.__module__]
torch.manual_seed(0)
layer = cls(cfg, layer_idx=0).to("cuda", torch.bfloat16)
layer.train()
h = (torch.randn(1, L, cfg.hidden_size, device="cuda") * 0.5).to(torch.bfloat16)
rel = lambda a, b: ((a - b).norm() / b.norm().clamp_min(1e-9)).item()

ref = None
for blk in BLOCKS:
    mod._mamba3_mimo_torch_segment.__defaults__ = (blk,)
    times = []
    for _ in range(4):
        x = h.clone().requires_grad_(True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        out = layer(x)
        out = out[0] if isinstance(out, tuple) else out
        out.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        times.append(time.time() - t)
    o, g = out.detach().float(), x.grad.float()
    if ref is None:
        ref = (o, g)
    print(f"block {blk:4d}: best {min(times[1:]):.3f}s  peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB  "
          f"rel out {rel(o, ref[0]):.2e}  rel grad {rel(g, ref[1]):.2e}", flush=True)
