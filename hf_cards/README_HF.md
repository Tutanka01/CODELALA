---
license: apache-2.0
base_model: OVHaiLLM/Olala-v1.0-mini-7A1B
language:
- en
- fr
pipeline_tag: text-generation
tags:
- code
- python
- olala
- mamba
- moe
- french
- bilingual
- token-efficient
library_name: transformers
---

# CodeLALA-v1 (7A1B)

<p align="center">
  <img src="codelala-banner.jpg" alt="CodeLALA-v1" width="520" />
</p>

**`CodeLALA-v1`** is a first post-training iteration (`v1`) of **[OVHaiLLM/Olala-v1.0-mini-7A1B](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B)** (6.8B total / ~1B active parameters, hybrid Mamba-3 + Attention MoE).

Trained with execution-verified on-policy Python DPO + RPO while keeping Mamba-3 recurrent dynamics (`in_proj_dyn`) frozen, **`v1` cuts token usage in half in `medium` reasoning mode** while matching the base model's `MBPP+` score under a 4,096-token budget — with fewer cutoffs and zero repetition loops on stability checks.

> **Looking for ready-to-run GGUF weights?** Get `Q6_K` at **[Tutanka01/CodeLALA-v1-7A1B-GGUF](https://huggingface.co/Tutanka01/CodeLALA-v1-7A1B-GGUF)**.
>
> **`v1` status**: This release establishes a clean, stable baseline focused on **concision and turn latency**. Work is actively underway on **`v2`** to push absolute coding accuracy above the base model and close the `disabled`-mode gap.

---

## Key Highlights (`v1`)

- **2× fewer tokens per turn (`medium` mode)**: median generation drops from `975` to **`481` tokens (`-50.7%`)**, average from `1,722` to **`1,041` (`-39.5%`)** (shorter on `124/148` tasks).
- **Matching `MBPP+` score under 4,096 tokens**: **`68.7%` (`103/150`)** on strictly held-out EvalPlus `MBPP+` tasks, with **`-39%` fewer truncations (`33 → 20`)** and **`0/8` loops** on bilingual FR/EN stability checks.
- **Weight integrity**: all 614 native `F32` Mamba-3 / geodesic scalars and `in_proj_dyn` weights are preserved byte-identical to the base checkpoint.

### Known Limitations (`v1`)

- **Completed-response accuracy**: on non-truncated `medium` completions (`finish == stop`), `v1` scores `103/128 (80.5%)` vs `103/117 (88.0%)` for base — the tie in overall `MBPP+` score (`103/150`) comes from fewer cutoffs at the 4,096-token cap.
- **`disabled` mode regression**: direct mode becomes ultra-compact (`37` median tokens, `0` truncations) but drops `-6.7 pts` on `MBPP+` (`56.0%` vs `62.7%`). **`reasoning_effort="medium"` is recommended.**
- **Scope**: `v1` targets Python; multi-language and broader coding benchmarks are planned for `v2`.

---

## Benchmarks (150 Held-Out `MBPP+` Tasks)

Evaluated deterministically (`temperature = 0.0`, `max_tokens = 4096`) on the first 150 tasks of `evalplus/mbppplus` (all 378 tasks strictly excluded from training).

| Model | Mode | `MBPP+` (Strict) | `MBPP` | `MBPP+` (`finish == stop`) | Median Tokens | Avg Tokens | Truncations (`4096`) | Loops (8 tests) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Olala-v1.0-mini-7A1B` (Base) | `medium` | 68.7% (103/150) | 76.7% (115/150) | 88.0% (103/117) | 975 | 1,722 | 33 / 150 | 1 / 8 |
| **`CodeLALA-v1`** | **`medium` (recommended)** | **68.7% (103/150)**\* | 76.0% (114/150) | 80.5% (103/128) | **481 (-50.7%)** | **1,041 (-39.5%)** | **20 / 150** | **0 / 8** |
| `Olala-v1.0-mini-7A1B` (Base) | `disabled` | **62.7% (94/150)** | **70.7% (106/150)** | 83.7% (77/92) | 795 | 1,835 | 58 / 150 | — |
| **`CodeLALA-v1`** | `disabled` | 56.0% (84/150) | 62.0% (93/150) | 56.0% (84/150) | **37 (-95.3%)** | **52 (-97.2%)** | **0 / 150** | — |

*\*Excluding 2 transient server timeouts during eval: `69.6%` (`103/148`). On the 88 tasks solved by both models in `medium`, median length drops from `888` to `441` tokens (`-50.3%`).*

---

## Quickstart (`transformers`)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Tutanka01/CodeLALA-v1-7A1B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "user", "content": "Écris une fonction Python propre pour fusionner une liste d'intervalles qui se chevauchent."}
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    reasoning_effort="medium",
    return_tensors="pt",
    return_dict=True,
).to(model.device)

out = model.generate(**inputs, max_new_tokens=1024, temperature=0.8, top_p=0.9)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

---

## Roadmap (`v1 → v2`)

- Push absolute coding accuracy clearly above the base model with larger execution-verified datasets and iterative RL/DPO.
- Recover `disabled`-mode accuracy while keeping concise outputs.
- Expand evaluation to official sampling (`T = 0.8`), HumanEval+, and additional languages (TypeScript, Rust, SQL).

---

## License & Attribution

Released under the **Apache-2.0** license, based on **[OVHaiLLM/Olala-v1.0-mini-7A1B](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B)** ([Official GGUF](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF)).
