"""Jeu d'évaluation : détecte les régressions de qualité du RAG.

Usage : python -m eval.run_eval [--threshold 0.8]
Un test passe si tous les mots-clés attendus sont dans la réponse ET
(si renseignée) la source attendue fait partie des sources retournées.
Code de sortie 1 si le score est sous le seuil.
"""
import argparse
import json
import sys
from pathlib import Path

from app import rag


def evaluate(cases: list[dict]) -> tuple[float, list[dict]]:
    results = []
    for case in cases:
        out = rag.answer(case["question"])
        text = out["answer"].lower()
        kw_ok = all(k.lower() in text for k in case["expected_keywords"])
        src = case.get("expected_source")
        src_ok = src is None or any(s["source"] == src for s in out["sources"])
        results.append({"question": case["question"], "ok": kw_ok and src_ok,
                        "keywords_ok": kw_ok, "source_ok": src_ok})
    score = sum(r["ok"] for r in results) / len(results) if results else 0.0
    return score, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="eval/questions.json")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    score, results = evaluate(json.loads(Path(args.file).read_text(encoding="utf-8")))
    for r in results:
        print(("OK  " if r["ok"] else "FAIL"), r["question"], "" if r["ok"] else f"(mots-clés={r['keywords_ok']}, source={r['source_ok']})")
    print(f"\nScore : {score:.0%} (seuil {args.threshold:.0%})")
    sys.exit(0 if score >= args.threshold else 1)
