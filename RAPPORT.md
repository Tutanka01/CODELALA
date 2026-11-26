# CODELALA — Rapport technique des expériences (v2 → v4)

Compte-rendu des itérations de post-entraînement menées sur **`Olala-v1.0-mini-7A1B`** (OVHcloud, 6,8 B total / ~1 B actif, 29 couches Mamba-3 + 7 Attention + 256 experts MoE, Apache 2.0) sur **NVIDIA DGX Spark (GB10, 119 Go unifiés)**.

---

## 1. Résumé exécutif

- **`v2` (échec)** : SFT LoRA standard sur datasets publics longs → perte de validation plate, réponses rallongées et `14 %` de boucles infinies en mode `disabled`.
- **`v3a` (échec diagnostiqué)** : introduction du scan Mamba-3 rapide et de `ExpertLoRA3D`, mais `use_rslora=True` (`scale = 16.0`) a déformé jusqu'à `31,1 %` des poids.
- **`v3b` (arrêté par gate)** : correction `scale = 2.0` → courbe propre et `max_rel <= 1,41 %`, mais `3/8` boucles infinies en `disabled`. Diagnostic de la cause racine : **`in_proj_dyn` ne doit pas être modifié** sur Mamba-3.
- **`v4` → `CodeLALA-v1`** : DPO + RPO on-policy Python vérifié par exécution (`168` paires, `--targets safe` avec `in_proj_dyn` gelé).
  - **En `medium`** : score `MBPP+` identique à la base sous budget `4 096` tokens (**`68,7 %`**, `103/150`), **2× moins de tokens (`-50,7 %` en médiane, `-39,5 %` en moyenne)**, **`-39 %` de troncatures (`33 → 20`)** et **`0/8` boucle**. Sur les seules réponses terminées (`finish == stop`), la précision est de `103/128 (80,5 %)` contre `103/117 (88,0 %)` pour la base : l'égalité globale vient de la réduction des troncatures.
  - **En `disabled`** : réponses ultra-courtes (`37` tokens médians, `0` troncature vs `58/150`), mais régression de **`-6,7 pts`** sur `MBPP+` (`84/150` vs `94/150`).

---

## 2. Architecture et contraintes matérielles

| Paramètre | Détail |
|---|---|
| Modèle de base | `OVHaiLLM/Olala-v1.0-mini-7A1B` (6,8 B total / ~1 B actif) |
| Architecture | 36 couches hybrides : 29 Mamba-3 + 7 Attention différentielle, 256 experts routés (6 actifs + 1 partagé) |
| Réglages officiels | `temperature = 0.8`, `top_p = 0.9`, `top_k = 40` ; `reasoning_effort` ∈ `{disabled, low, medium, high}` |
| Matériel d'entraînement | NVIDIA DGX Spark (GB10, `sm_121`, 119 GiB unifiés CPU/GPU) |
| Contrainte noyau | Le backward TileLang Mamba-3 d'OVH demande 123–224 Ko de shared memory par bloc (> 101 Ko sur GB10, cf. `diag/test_kernel.py`) |

---

## 3. Expérience `v2` — SFT sur datasets publics (échec)

### Protocole
- Banc WebGen : 12 tâches web tenues hors entraînement (`eval/tasks_webgen.jsonl`), 3 essais par tâche (`36` réponses), testées en Firefox headless (`eval/webcheck.py`) puis notées par `Qwen3.6-35B-A3B` (`eval/judge.py`).
- Données : `1 642` exemples issus de `CodeFeedback` (Apache-2.0) et `AdhyanshVerma/html-css-js-cot` (GPL-3.0, non redistribué), entraînés sur 206 pas (~11 h 30 en PyTorch pur ; accélération ×1,45 obtenue en passant la taille de bloc Mamba de `64` à `128`, écart numérique `3e-5`, cf. `diag/test_block.py`).

### Résultats WebGen (36 réponses par configuration)

| Indicateur | Base (`disabled`) | `v2` (`disabled`) | Base (`medium`) | `v2` (`medium`) |
|---|---:|---:|---:|---:|
| Test de fumée Firefox | 39 % | 28 % | 47 % | 33 % |
| Pages jugées fonctionnelles | 0 % | 0 % | 3 % (1/36) | non mesuré |
| Note moyenne du juge | 1,1/10 | 0,5/10 | 1,4/10 | non mesuré |
| Coupées à 14 000 tokens | 0 % | 14 % (5/36) | 3 % (1/36) | 11 % (4/36) |
| Tokens moyens | 1 961 | 3 371 | 6 980 | 3 971 |

### Diagnostic
- **Perte de validation plate** (`0,820 → 0,810` sur 206 pas, cf. `results/v2_trainer_state.json`).
- **Décalage de distribution** : les prompts d'entraînement faisaient `4 700` caractères en médiane (cahiers des charges longs) avec des réponses filtrées à ≥ `4 000` caractères. Le modèle a appris la verbosité, pas la justesse, et part en boucles de répétition (`score = score;`).
- **Conversion GGUF hors de cause** : `diag/gguf_diff.py` a confirmé que seuls les poids ciblés avaient évolué (`0,3 %` à `3,7 %`).

---

## 4. Contributions d'ingénierie (`train/olala_fast.py`)

Pour lever le goulot d'étranglement de `v2` (~500 s/pas) et débloquer l'adaptation des experts routés :

1. **Scan Mamba-3 SSD chunk-parallèle (`patch_fast_mamba3`)** : remplace la boucle Python séquentielle fp32 (`_mamba3_mimo_torch_segment`) par un scan par blocs (`block=32`) en TF32 + `torch.compile(dynamic=False)` + padding en buckets fixes (`512/1024/2048/3072/4096`). Temps ramené à **~72 s/pas (16 exemples)** (~6–8× plus rapide), pic mémoire **37 / 119 GiB**.
2. **3D Expert-LoRA sur ScatterMoE (`ExpertLoRA3D`)** : wrappe les tenseurs 3D `[256, out, in]` des 256 experts routés (`5,44 B / 6,77 B` paramètres = `80,3 %` du modèle, totalement gelés en `v2`) avec maîtres `float32` et fusion fp32 (`merge_all_to_safetensors()`).
3. **Intégrité des poids** : la fusion conserve les 614 tenseurs `F32` géodésiques/Mamba strictement identiques octet pour octet.

---

## 5. Expériences `v3a` et `v3b` — SFT calibré & diagnostic des boucles

- **`v3a` (rejeté)** : `153` apps HTML/CSS/JS synthétisées par `Qwen3.6-35B-A3B` (vérifiées Firefox + juge QA, notes 8–10/10, 0 % chevauchement held-out) + `79` ancres FR/EN. Entraîné avec `r=64, alpha=128, use_rslora=True, lr=1.2e-4, webapp_repeat=3`.
  - **Cause d'échec** : `use_rslora=True` donne `scale = alpha / sqrt(r) = 16.0` (8× trop élevé vs `2.0`), déformant `ffn_down_shexp` de **31,1 %** et `ssm_in_dyn` de **28,8 %** (`results/v3/v3a_sft/`). Smoke `disabled` tombé à `7/36 (19 %)`.
- **`v3b` (rejeté par gate de stabilité)** : correction `use_rslora=False` (`scale = 2.0`), `lr=4e-5`, `webapp_repeat=1` (`905` train / `71` eval, 114 pas, ~2 h 20).
  - Courbe propre (`eval_loss : 0,8033 → 0,6832`) et `gguf_diff` propre (`max_rel <= 1,41 %`).
  - Sur `quick_eval.sh` : `medium` = `3/8` web (`0` boucle) et `17/20` MBPP+ ; mais `disabled` = `1/8` web et **`3/8` boucles infinies à 14 000 tokens** (`results/v3/v3b_sft/`).
  - **Cause racine identifiée** : contrairement au script officiel d'OVH (`target_modules = ["in_proj", "c_q", "mixer_proj", "fc_1", "fc_2", "down_proj", "up_proj"]`), `v2`, `v3a` et `v3b` incluaient **`in_proj_dyn`** (dynamique récurrente `dt`/`A`/angles des 29 couches Mamba-3). Le moindre écart (`~1,2 %`) déstabilise le mode `disabled`.

---

## 6. Expérience `v4` → `CodeLALA-v1` (DPO + RPO Python vérifié par exécution)

### Protocole (`run_v4_python.sh`)
- **Anti-contamination** : les **378 tâches `evalplus/mbppplus`** sont strictement exclues (filtrage par noms de fonctions + similarité Jaccard).
- **Synthèse vérifiée (`data/synth_python_v4.py`)** : **179 exercices Python originaux** (13 catégories, anglais et français), chacun validé par exécution sur 3 tests publics + 10 tests cachés de cas limites.
- **Collecte on-policy (`data/collect_dpo_python.py`)** : `K = 6` rollouts d'`Olala-base` par tâche exécutés en sous-processus isolés → **168 paires contrastives `(PASS, FAIL)`**.
- **Entraînement (`train/dpo_v3.py`)** : DPO + RPO normalisé en longueur, `--targets safe` (**`in_proj_dyn` 100 % gelé**), `scale = 2.0`, `lr = 3e-5`, `expert_r = 8`, `beta = 2.5`, `rpo_alpha = 0.35`, 42 pas (`loss : 0,906 → 0,50–0,67`, `pref_acc : 87,5–100 %`).
- **Audit GGUF (`results/v4/eval/gguf_diff.txt`)** : `989/989` tenseurs, `type_mismatch = 0`, 614 tenseurs `F32` strictement identiques, `ssm_in_dyn` 100 % intact, `max_rel = 1,00 %` (`attn_output`).

### Résultats comparatifs (150 tâches `MBPP` / `MBPP+` held-out, `T = 0.0`, `max_tokens = 4096`)

| Modèle / Mode | `MBPP` | `MBPP+` (strict) | `MBPP+` sur `finish == stop` | Tokens médians | Tokens moyens | Troncatures (`4096`) | Boucles (8 tests) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`Olala-base`** (`medium`) | 115/150 (76,7 %) | **103/150 (68,7 %)** | 103/117 (88,0 %) | 975 | 1 722 | 33/150 (22 %) | 1/8 |
| **`CodeLALA-v1`** (`medium`) | 114/150 (76,0 %) | **103/150 (68,7 %)**\* | 103/128 (80,5 %) | **481 (-50,7 %)** | **1 041 (-39,5 %)** | **20/150 (-39 %)** | **0/8** |
| **`Olala-base`** (`disabled`) | 106/150 (70,7 %) | **94/150 (62,7 %)** | 77/92 (83,7 %) | 795 | 1 835 | 58/150 (39 %) | — |
| **`CodeLALA-v1`** (`disabled`) | 93/150 (62,0 %) | 84/150 (56,0 %) | 84/150 (56,0 %) | **37 (-95,3 %)** | **52 (-97,2 %)** | **0/150 (0 %)** | — |

*\*Hors 2 timeouts serveur lors de l'évaluation : `103/148 (69,6 %)`. Détail tâche par tâche : `15` tâches gagnées et `15` perdues en `medium` ; `v1` est plus court sur `124/148` tâches (ratio médian `0,542`), et passe de `888` à `441` tokens médians (`-50,3 %`) sur les 88 tâches réussies par les deux modèles.*

### Limites de l'évaluation & pistes `v2`
- Évaluation effectuée à `T = 0.0` avec un plafond à `4 096` tokens (les réglages officiels d'OVH utilisent `T = 0.8`). L'égalité des scores `MBPP+` en `medium` dépend en partie de ce plafond où la base échoue sur ses `33` troncatures.
- En `disabled`, la perte de `10` tâches (`3` gagnées, `13` perdues) montre une concision excessive (`37` tokens médians) qui coupe la réflexion nécessaire sur les cas limites.
- `v1` n'a pas été réévaluée sur WebGen.
- **Pistes `v2`** : augmenter la taille et la difficulté du jeu vérifié par exécution (passer au-delà de `168` paires), équilibrer les paires `disabled` pour restaurer sa précision, et évaluer aussi à `T = 0.8` ainsi que sur HumanEval+ et multi-langages.

Tous les journaux bruts sont archivés dans `results/v4/` (`base/`, `dpo/`, `eval/`).
