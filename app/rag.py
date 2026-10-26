import anthropic

from app.config import ANTHROPIC_MODEL, TOP_K
from app.db import connect
from app.embeddings import embed

SYSTEM_PROMPT = """Tu es un assistant qui répond à des questions sur des documents d'assurance.
Règles :
- Réponds uniquement à partir des extraits fournis.
- Cite tes sources avec leur numéro entre crochets, par exemple [1], [2].
- Si les extraits ne contiennent pas la réponse, réponds exactement : "Je ne trouve pas cette information dans les documents."
- Réponds en français, de façon concise."""

_client = None


def _llm() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # lit ANTHROPIC_API_KEY
    return _client


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    vec = embed([question])[0]
    conn = connect()
    rows = conn.execute(
        """
        SELECT source, page, content, 1 - (embedding <=> %s) AS score
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (vec, vec, k),
    ).fetchall()
    conn.close()
    return [{"source": r[0], "page": r[1], "content": r[2], "score": float(r[3])} for r in rows]


def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] ({p['source']}, page {p['page']})\n{p['content']}"
        for i, p in enumerate(passages, start=1)
    )
    return f"Extraits :\n{context}\n\nQuestion : {question}"


def answer(question: str) -> dict:
    passages = retrieve(question)
    if not passages:
        return {"answer": "Aucun document n'a été ingéré.", "sources": []}
    response = _llm().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, passages)}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return {
        "answer": text,
        "sources": [{"source": p["source"], "page": p["page"], "score": round(p["score"], 3)} for p in passages],
    }
