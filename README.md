# DocuAsk – Assistant RAG sur documents d'assurance

API FastAPI qui répond à des questions sur des PDF (conditions générales d'assurance)
en citant ses sources. Recherche vectorielle PostgreSQL/pgvector + LLM Claude.

## Architecture
PDF -> extraction texte (OCR Tesseract si page scannée) -> chunks (200 mots, 40 de recouvrement)
-> embeddings multilingues (fastembed) -> pgvector.
Question -> embedding -> top-k chunks (distance cosinus) -> prompt -> Claude -> réponse + sources.

## Lancer
    cp .env.example .env        # renseigner ANTHROPIC_API_KEY
    # déposer des PDF dans data/
    docker compose up -d --build
    docker compose run --rm api python -m app.ingest data
    curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
         -d '{"question": "Quel est le délai de carence ?"}'

Doc interactive : http://localhost:8000/docs

## Tests
    pip install -r requirements-dev.txt && pytest
    docker compose run --rm api python -m eval.run_eval --threshold 0.8

## Choix techniques
- Embeddings locaux multilingues : pas de coût API, bons résultats en français.
- Ré-ingestion idempotente (suppression des chunks d'un fichier avant réinsertion).
- Prompt strict : réponse uniquement depuis les extraits, sinon "Je ne trouve pas".
- Évaluation : mots-clés attendus + source attendue, seuil bloquant (code retour 1).

## Pistes d'amélioration
Reranking, recherche hybride (BM25 + vecteurs), index HNSW, streaming, métriques de latence.
