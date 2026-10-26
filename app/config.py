import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://docuask:docuask@localhost:5432/docuask")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "200"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))
TOP_K = int(os.getenv("TOP_K", "4"))
