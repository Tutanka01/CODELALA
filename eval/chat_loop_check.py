"""Quick stability & bilingual anti-forgetting check (zero loops + clean French/English chat)."""
import argparse
import asyncio
import json
from pathlib import Path
import httpx

PROMPTS = [
    ("fr_chat_1", "Explique simplement en 4 phrases la différence entre un processus et un thread en informatique.", "disabled"),
    ("fr_chat_2", "Résume en 3 points pourquoi l'architecture hybride Mamba + Attention est intéressante pour les LLM.", "medium"),
    ("en_chat_1", "Explain in 3 concise sentences why hash tables have O(1) average lookup time.", "disabled"),
    ("fr_py_1", "Écris une fonction Python `est_palindrome(s)` qui ignore la casse, les espaces et la ponctuation.", "disabled"),
    ("web_loop_1", "Fais-moi un jeu Flappy Bird complet dans un seul fichier HTML, avec le CSS et le JavaScript inclus.", "disabled"),
    ("web_loop_2", "Write a complete Snake game in a single HTML file (HTML, CSS and JavaScript together).", "disabled"),
    ("web_loop_3", "Build a complete calculator web app in a single HTML file with keyboard support.", "disabled"),
    ("web_loop_4", "Fais-moi une application de liste de tâches (to-do list) complète dans un seul fichier HTML.", "disabled"),
]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        for pid, prompt, eff in PROMPTS:
            body = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 4096,
                "chat_template_kwargs": {"reasoning_effort": eff},
            }
            r = await client.post(f"{args.url}/v1/chat/completions", json=body)
            r.raise_for_status()
            d = r.json()
            msg = d["choices"][0]["message"]
            rec = {
                "id": pid,
                "effort": eff,
                "finish": d["choices"][0].get("finish_reason"),
                "tokens": d.get("usage", {}).get("completion_tokens", 0),
                "preview": (msg.get("content") or "")[:240].replace("\n", " "),
            }
            rows.append(rec)
            print(f"[{pid:10s} {eff:8s}] finish={rec['finish']:6s} tok={rec['tokens']:4d} | {rec['preview'][:110]}", flush=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    n_loops = sum(1 for r in rows if r["finish"] == "length")
    print(f"STABILITY_CHECK: loops={n_loops}/{len(rows)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
