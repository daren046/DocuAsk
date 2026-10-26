from fastapi.testclient import TestClient

from app import rag
from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_ok(monkeypatch):
    monkeypatch.setattr(
        rag, "answer",
        lambda q: {"answer": "Oui [1]", "sources": [{"source": "cg.pdf", "page": 2, "score": 0.9}]},
    )
    r = client.post("/ask", json={"question": "Quel est le délai de carence ?"})
    assert r.status_code == 200
    assert r.json()["sources"][0]["page"] == 2


def test_ask_question_too_short():
    assert client.post("/ask", json={"question": "a"}).status_code == 422


def test_ask_backend_error(monkeypatch):
    def boom(q):
        raise RuntimeError("down")
    monkeypatch.setattr(rag, "answer", boom)
    assert client.post("/ask", json={"question": "Question valide ?"}).status_code == 502


def test_build_prompt_numbers_sources():
    prompt = rag.build_prompt("Q ?", [
        {"source": "a.pdf", "page": 1, "content": "texte A", "score": 0.8},
        {"source": "b.pdf", "page": 5, "content": "texte B", "score": 0.7},
    ])
    assert "[1] (a.pdf, page 1)" in prompt and "[2] (b.pdf, page 5)" in prompt
