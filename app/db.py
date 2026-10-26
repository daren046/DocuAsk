import psycopg
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL, EMBEDDING_DIM


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            page INT NOT NULL,
            content TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}) NOT NULL
        )
        """
    )
