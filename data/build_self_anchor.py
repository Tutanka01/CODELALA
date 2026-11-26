"""Generate a bilingual FR/EN self-distillation anti-forgetting anchor from Olala-base itself.

Samples clean general-knowledge, French/English reasoning, summarization, and explanation
answers directly from Olala-base so Stage-1 SFT keeps KL divergence near zero outside coding.
"""
import argparse
import asyncio
import json
from pathlib import Path

import httpx

ANCHOR_PROMPTS = [
    # French general knowledge, explanation, writing, reasoning
    "Explique simplement comment fonctionne la photosynthèse en 5 phrases.",
    "Quelle est la différence entre la mémoire RAM et un disque SSD ?",
    "Résume en quelques points les causes principales de la Révolution française.",
    "Rédige un court e-mail professionnel pour reporter une réunion de mardi à jeudi matin.",
    "Pourquoi le ciel est-il bleu en journée et rouge au coucher du soleil ?",
    "Donne 5 conseils pratiques pour mieux gérer son temps au travail.",
    "Explique la différence entre un virus et une bactérie.",
    "Qu'est-ce que l'effet de serre et quel est son rôle sur le climat terrestre ?",
    "Un train parcourt 180 km en 1h30. Quelle est sa vitesse moyenne en km/h ?",
    "Compare brièvement les avantages et limites de l'énergie solaire et de l'énergie éolienne.",
    "Explique ce qu'est l'inflation en économie avec un exemple concret.",
    "Comment fonctionne un vaccin dans le système immunitaire ?",
    "Rédige une courte recette de crêpes pour 4 personnes.",
    "Quelle est la différence entre météo et climat ?",
    "Explique le principe de la sélection naturelle de Darwin en termes simples.",
    "Cite trois grandes différences entre l'ADN et l'ARN.",
    "Pourquoi la glace flotte-t-elle sur l'eau liquide ?",
    "Comment fonctionne le protocole HTTPS pour sécuriser un site web ?",
    "Résume le théorème de Pythagore et donne un exemple d'application.",
    "Qu'est-ce que le PIB d'un pays et que ne mesure-t-il pas ?",
    "Explique la différence entre apprentissage supervisé et non supervisé en IA.",
    "Donne une courte biographie de Marie Curie.",
    "Quels sont les principaux organes du système digestif humain ?",
    "Explique pourquoi les saisons existent sur Terre.",
    "Si un article coûte 80 € et bénéficie d'une réduction de 25 %, quel est son nouveau prix ?",
    "Quelle est la différence entre un roman, une nouvelle et un essai ?",
    "Explique le fonctionnement d'un panneau photovoltaïque.",
    "Qu'est-ce que la tectonique des plaques ?",
    "Donne 4 astuces pour améliorer la qualité de son sommeil.",
    "Explique simplement ce qu'est un trou noir.",
    "Quelle est la différence entre HTTP GET et POST ?",
    "Écris un court paragraphe motivant pour encourager un étudiant avant un examen.",
    "Comment fonctionne un moteur électrique par rapport à un moteur thermique ?",
    "Explique la règle d'accord du participe passé avec l'auxiliaire avoir.",
    "Qu'est-ce qu'une étoile à neutrons ?",
    "Pourquoi entend-on le tonnerre après avoir vu l'éclair ?",
    "Explique le rôle de la banque centrale dans la zone euro.",
    "Quelles sont les principales couches de l'atmosphère terrestre ?",
    "Résous l'équation 3x - 7 = 14 en détaillant les étapes.",
    "Explique ce qu'est un écosystème et donne un exemple.",
    # English general knowledge, explanation, writing, reasoning
    "Explain how a refrigerator moves heat from inside to outside in simple terms.",
    "What is the difference between a process and a thread in an operating system?",
    "Summarize the main principles of the scientific method in 5 bullet points.",
    "Write a polite short email thanking a colleague for their help on a presentation.",
    "Why do astronauts float inside the International Space Station?",
    "Explain compound interest with a concrete numerical example.",
    "What is the difference between machine learning and deep learning?",
    "How does GPS determine your exact location on Earth?",
    "A store increases a $50 item by 20% and then discounts it by 20%. What is the final price?",
    "Explain the water cycle in one concise paragraph.",
    "What is the difference between TCP and UDP?",
    "Describe three major causes of ocean tides.",
    "How do noise-canceling headphones reduce ambient sound?",
    "Explain the difference between correlation and causation with an example.",
    "What are the main differences between classical and quantum bits?",
    "Give 5 practical tips for giving a clear technical presentation.",
    "Explain how public-key cryptography works using an everyday analogy.",
    "Why is biodiversity important for ecosystem resilience?",
    "If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?",
    "Summarize the key differences between SQL and NoSQL databases.",
    "How does photosynthesis differ from cellular respiration?",
    "Explain what DNS does when you type a URL into a browser.",
    "What causes earthquakes along tectonic plate boundaries?",
    "Write a brief, clear summary of Newton's three laws of motion.",
    "How do lithium-ion batteries store and release electrical energy?",
    "Explain the difference between accuracy, precision, and recall.",
    "What is the Doppler effect? Give an everyday example.",
    "Explain why airplanes have curved wings and how lift is generated.",
    "How does an mRNA vaccine differ from a traditional attenuated vaccine?",
    "Solve step by step: what is 15% of 240 plus 30% of 150?",
    "Explain the concept of supply and demand in economics.",
    "What is the difference between RAM, CPU cache, and disk storage?",
    "Describe the life cycle of a medium-sized star like our Sun.",
    "How does fiber-optic cable transmit data using light?",
    "Explain what a hash function is and list two common uses.",
    "What is the difference between renewable and non-renewable energy?",
    "Write a short 3-sentence summary of why sleep consolidates memory.",
    "Explain how a compass works using Earth's magnetic field.",
    "What is the greenhouse effect and why is Venus hotter than Mercury?",
    "Explain the difference between a stack and a queue in everyday terms.",
]


async def one(client, url, prompt, effort, sem, fout, lock):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 2048,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    async with sem:
        r = await client.post(f"{url}/v1/chat/completions", json=body)
        r.raise_for_status()
        d = r.json()
        msg = d["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        if not content or d["choices"][0].get("finish_reason") != "stop":
            return
        assistant = {"role": "assistant", "content": content}
        if effort != "disabled" and reasoning:
            assistant["reasoning_content"] = reasoning
        row = {
            "messages": [{"role": "user", "content": prompt}, assistant],
            "chat_template_kwargs": {"reasoning_effort": effort},
            "meta": {"source": "olala_self_anchor", "bucket": "anchor_general"},
        }
        async with lock:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")
    sem, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    jobs = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        with out.open("a") as fout:
            for i, p in enumerate(ANCHOR_PROMPTS):
                eff = "disabled" if i % 2 == 0 else "low"
                jobs.append(one(client, args.url, p, eff, sem, fout, lock))
            await asyncio.gather(*jobs)
    print(f"Wrote self-anchor to {out} ({len(out.read_text().splitlines())} rows)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
