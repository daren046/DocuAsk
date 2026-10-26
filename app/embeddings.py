from functools import lru_cache

from fastembed import TextEmbedding

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=EMBEDDING_MODEL)


def embed(texts: list[str]):
    return list(_model().embed(texts))
