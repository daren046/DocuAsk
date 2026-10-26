def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Découpe un texte en chunks de `size` mots avec `overlap` mots de recouvrement."""
    if size <= 0:
        raise ValueError("size doit être > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap doit être >= 0 et < size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks
