"""Ingestion : PDF -> texte (OCR si page scannée) -> chunks -> embeddings -> pgvector.

Usage : python -m app.ingest data/
"""
import sys
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader

from app.chunking import chunk_text
from app.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS
from app.db import connect, init_schema
from app.embeddings import embed

MIN_CHARS_BEFORE_OCR = 30


def extract_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if len(text) < MIN_CHARS_BEFORE_OCR:  # page probablement scannée -> OCR
            image = convert_from_path(str(pdf_path), first_page=i + 1, last_page=i + 1)[0]
            text = pytesseract.image_to_string(image, lang="fra").strip()
        pages.append(text)
    return pages


def ingest_pdf(conn, pdf_path: Path) -> int:
    conn.execute("DELETE FROM chunks WHERE source = %s", (pdf_path.name,))  # ré-ingestion idempotente
    total = 0
    for page_number, text in enumerate(extract_pages(pdf_path), start=1):
        chunks = chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        if not chunks:
            continue
        vectors = embed(chunks)
        for chunk, vec in zip(chunks, vectors):
            conn.execute(
                "INSERT INTO chunks (source, page, content, embedding) VALUES (%s, %s, %s, %s)",
                (pdf_path.name, page_number, chunk, vec),
            )
        total += len(chunks)
    return total


def main(folder: str) -> None:
    conn = connect()
    init_schema(conn)
    pdfs = sorted(Path(folder).glob("*.pdf"))
    if not pdfs:
        print(f"Aucun PDF trouvé dans {folder}")
        return
    for pdf in pdfs:
        n = ingest_pdf(conn, pdf)
        print(f"{pdf.name} : {n} chunks")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
