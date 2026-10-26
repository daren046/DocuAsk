from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import rag

app = FastAPI(title="DocuAsk", description="Assistant RAG sur documents d'assurance")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Source(BaseModel):
    source: str
    page: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return rag.answer(req.question)
    except Exception as exc:  # erreur LLM ou base indisponible
        raise HTTPException(status_code=502, detail=f"Erreur du service : {exc}") from exc
