<p align="center">
  <img src="assets/codelala-banner.jpg" alt="CODELALA" width="520" />
</p>

<h1 align="center">CODELALA</h1>

<p align="center">
  Post-entraînement open-source de <code>Olala-v1.0-mini-7A1B</code> (OVHcloud) — <strong>v1</strong> orientée concision et stabilité en code Python local.
</p>

<p align="center">
  <a href="https://huggingface.co/Tutanka01/CodeLALA-v1-7A1B-GGUF"><img src="https://img.shields.io/badge/🤗_Hugging_Face-CodeLALA--v1--7A1B--GGUF-FFD21E?style=for-the-badge" alt="Hugging Face GGUF" /></a>
  <a href="https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B"><img src="https://img.shields.io/badge/Base-Olala--v1.0--mini--7A1B-0066CC?style=for-the-badge" alt="Base Model" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge" alt="License Apache 2.0" /></a>
</p>

---

## Aperçu (`CodeLALA-v1`)

**[Olala-v1.0-mini-7A1B](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B)** est un modèle hybride **6,8 B total / ~1 B actif** (29 couches Mamba-3 + 7 Attention + 256 experts MoE) capable de tourner à **100+ tok/s** en local via le [fork `llama.cpp` d'OVHai-LLM](https://github.com/OVHai-LLM/llama.cpp).

**`CodeLALA-v1`** est une première itération de post-entraînement (DPO + RPO on-policy vérifié par exécution Python, sans contamination `MBPP+`) qui vise des réponses plus courtes et plus stables pour un usage local :

- **2× moins de tokens en mode `medium`** : médiane ramenée de `975` à **`481` tokens (`-50,7 %`)**, moyenne de `1 722` à **`1 041` (`-39,5 %`)** — plus court sur `124/148` tâches.
- **Score `MBPP+` conservé sous budget 4 096 tokens** : **`68,7 %` (`103/150`)** comme la base, avec **`-39 %` de troncatures (`33 → 20`)** et **`0/8` boucle** sur le test de stabilité bilingue FR/EN.
- **Poids vérifiés** : les 614 tenseurs `F32` géodésiques/Mamba et `in_proj_dyn` restent strictement intacts (`diag/gguf_diff.py`).

### Limites connues de la v1 (en cours de travail pour la v2)

- **Précision brute** : sur les seules réponses terminées sans troncature (`finish == stop`), `v1` atteint `103/128 (80,5 %)` contre `103/117 (88,0 %)` pour la base ; l'égalité au score global (`103/150`) provient de la baisse des troncatures au plafond de `4 096` tokens.
- **Mode `disabled`** : réponses ultra-courtes (`37` tokens en médiane, `0` troncature), mais régression de `-6,7 pts` sur `MBPP+` (`56,0 %` vs `62,7 %`). Le mode **`medium`** est recommandé.
- **Périmètre** : `v1` a été entraînée et évaluée sur Python (`MBPP` / `MBPP+`) et n'a pas encore été réévaluée sur la suite WebGen.

---

## Benchmarks (`MBPP` / `MBPP+`, 150 tâches held-out)

Évaluation déterministe (`temperature = 0.0`, `max_tokens = 4096`) sur les 150 premières tâches de `evalplus/mbppplus` (les 378 tâches étant strictement exclues de l'entraînement). Journaux bruts dans [`results/v4/`](results/v4/).

| Modèle | Mode | `MBPP+` (strict) | `MBPP` | Tokens médians | Tokens moyens | Troncatures (`4096`) | Boucles (8 tests) |
|---|---|---:|---:|---:|---:|---:|---:|
| `Olala-v1.0-mini-7A1B` (Base) | `medium` | 68,7 % (103/150) | 76,7 % (115/150) | 975 | 1 722 | 33 / 150 | 1 / 8 |
| **`CodeLALA-v1`** | **`medium` (recommandé)** | **68,7 % (103/150)**\* | 76,0 % (114/150) | **481 (-50,7 %)** | **1 041 (-39,5 %)** | **20 / 150** | **0 / 8** |
| `Olala-v1.0-mini-7A1B` (Base) | `disabled` | **62,7 % (94/150)** | **70,7 % (106/150)** | 795 | 1 835 | 58 / 150 | — |
| **`CodeLALA-v1`** | `disabled` | 56,0 % (84/150) | 62,0 % (93/150) | **37 (-95,3 %)** | **52 (-97,2 %)** | **0 / 150** | — |

*\*2 timeouts serveur pendant l'évaluation (`103/148 = 69,6 %` hors timeouts). Sur les 88 tâches résolues par les deux modèles, la médiane passe de `888` à `441` tokens (`-50,3 %`).*

---

## Ingénierie & Enseignements

Entraîner un hybride **Mamba-3 + ScatterMoE** sur GPU **NVIDIA GB10 (DGX Spark, 119 Go unifiés)** — où le noyau backward TileLang d'OVH dépasse la mémoire partagée — a nécessité plusieurs briques dédiées dans [`train/olala_fast.py`](train/olala_fast.py) :

- **Scan Mamba-3 SSD chunk-parallèle (`patch_fast_mamba3`)** : remplace la boucle Python séquentielle par un scan par blocs (`block=32`) TF32 + `torch.compile` → **`~6–8×` plus rapide** et `~2,5×` moins de VRAM.
- **3D Expert-LoRA (`ExpertLoRA3D`)** : adapte directement les tenseurs 3D `[256, out, in]` des 256 experts routés (`5,44 B` paramètres, `80,3 %` du modèle, gelés par un LoRA standard), tout en gardant le routeur figé.
- **Stabilité Mamba-3 (`--targets safe`)** : nos essais précédents ont montré deux pièges majeurs — `use_rslora=True` déforme excessivement les poids (`max_rel = 31 %`), et modifier `in_proj_dyn` provoque des boucles infinies en mode `disabled`. Geler `in_proj_dyn` résout le problème.
- **Fusion chirurgicale fp32 & audit GGUF** : `merge_all_to_safetensors()` + [`diag/gguf_diff.py`](diag/gguf_diff.py) garantissent `0` erreur de type et une intégrité bit à bit des scalaires Mamba/géodésiques.

### Historique des itérations

| Itération | Méthode | Bilan |
|---|---|---|
| **v2** | SFT LoRA sur datasets publics longs (`CodeFeedback` + `html-css-js-cot`) | Dégradé : réponses trop longues, `14 %` de boucles en `disabled`. |
| **v3a / v3b** | Patch Mamba-3 SSD + `ExpertLoRA3D` + SFT calibré | `~6–8×` plus rapide ; identification des pièges `rsLoRA` (`v3a`) et `in_proj_dyn` (`v3b`). |
| **v4 (`CodeLALA-v1`)** | DPO + RPO on-policy Python vérifié par exécution + `--targets safe` | `2×` moins de tokens en `medium` à score `MBPP+` égal sous 4k, `0/8` boucle. |

Détails complets dans [`RAPPORT.md`](RAPPORT.md).

---

## Utilisation rapide

### Inférence locale (`Q6_K` GGUF)

Utiliser le [fork `llama.cpp` d'OVHai-LLM](https://github.com/OVHai-LLM/llama.cpp) (requis pour l'architecture `olala`) avec le modèle disponible sur [`Tutanka01/CodeLALA-v1-7A1B-GGUF`](https://huggingface.co/Tutanka01/CodeLALA-v1-7A1B-GGUF) :

```bash
llama-server \
  -m CodeLALA-v1-7A1B-Q6_K.gguf \
  -ngl 99 -fa 1 -c 32768 \
  --host 127.0.0.1 --port 8080
```

### Reproduire le pipeline v4

```bash
chmod +x run_v4_python.sh
./run_v4_python.sh
```

---

## Feuille de route (`v1 → v2`)

`v1` pose une base technique propre et rapide. Les travaux en cours pour la **`v2`** portent sur :

- **Gain réel en précision absolue** : passer de `168` paires à un volume plus large d'exercices vérifiés par exécution (avec RL/GRPO ou DPO multi-tours) et évaluer aussi aux réglages d'échantillonnage officiels (`T = 0.8`).
- **Correction du mode `disabled`** : retrouver la précision de la base sans perdre la concision.
- **Élargissement des benchmarks** : couvrir HumanEval+, multi-langages (TypeScript, Rust, SQL) et WebGen.

---

## Structure du dépôt

```text
CODELALA/
├── assets/         # Bannière
├── data/           # Synthèse vérifiée par exécution & collecte de paires DPO
├── diag/           # gguf_diff.py et diagnostics Mamba-3
├── eval/           # Benchmarks MBPP/MBPP+, stabilité FR/EN et WebGen
├── hf_cards/       # Model cards Hugging Face
├── results/        # Journaux et rapports bruts (v2, v3, v4)
├── scripts/        # Publication Hugging Face
├── train/          # olala_fast.py (SSD + ExpertLoRA3D), sft_v3.py, dpo_v3.py
├── run_v4_python.sh
└── RAPPORT.md
```

---

## Licence

Distribué sous licence **[Apache License 2.0](LICENSE)** — Copyright © 2026 **Mohamad El Akhal** (voir [`NOTICE`](NOTICE)).
