"""Compare the TileLang Mamba-3 chunk kernel with the pure-PyTorch fallback on one
randomly initialised OlalaMamba3MimoFast layer: outputs, input gradients, speed.
Run with PYTHONPATH=src/mamba so that mamba_ssm is importable. Does not touch the model files."""
import sys
import time

import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL = "models/olala-hf"
L = int(sys.argv[1]) if len(sys.argv) > 1 else 2048

cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
cls = get_class_from_dynamic_module("modeling_olala.OlalaMamba3MimoFast", MODEL)
mod = sys.modules[cls.__module__]
kernel = mod._mamba3_mimo
print("kernel available:", kernel is not None)

torch.manual_seed(0)
layer = cls(cfg, layer_idx=0).to("cuda", torch.bfloat16)
layer.train()
h = (torch.randn(1, L, cfg.hidden_size, device="cuda") * 0.5).to(torch.bfloat16)


CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else None


def cast_kernel(**kw):
    if CHUNK:
        kw["chunk_size"] = CHUNK
    # the public kernel wants Q/K in the low-precision dtype; the model hands them over in float32
    kw["Q"] = kw["Q"].to(kw["dtype"])
    kw["K"] = kw["K"].to(kw["dtype"])
    return kernel(**kw)


def run(impl, tag, reps=3):
    mod._mamba3_mimo = impl
    out = g = None
    times = []
    for _ in range(reps):
        x = h.clone().requires_grad_(True)
        torch.cuda.synchronize()
        t = time.time()
        out = layer(x)
        out = out[0] if isinstance(out, tuple) else out
        loss = out.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.time() - t)
        g = x.grad.float()
    print(f"{tag}: fwd+bwd times {[round(t, 3) for t in times]} loss {loss.item():.6f}", flush=True)
    return out.detach().float(), g


ref_o, ref_g = run(None, "torch fallback")
if kernel is not None:
    print("layer chunk_size", layer.chunk_size, "mimo_dim", layer.mimo_dim, "d_state", layer.d_state, "headdim", layer.headdim, "override", CHUNK)
    k_o, k_g = run(cast_kernel, "kernel with Q/K cast")
    rel = lambda a, b: ((a - b).norm() / b.norm().clamp_min(1e-9)).item()
    print(f"rel diff output {rel(k_o, ref_o):.3e}  rel diff input-grad {rel(k_g, ref_g):.3e}")
    print("nan in kernel output:", bool(torch.isnan(k_o).any()), "nan in grad:", bool(torch.isnan(k_g).any()))
