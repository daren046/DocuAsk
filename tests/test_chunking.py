import pytest

from app.chunking import chunk_text


def test_empty_text():
    assert chunk_text("") == []


def test_short_text_single_chunk():
    assert chunk_text("un deux trois", size=10, overlap=2) == ["un deux trois"]


def test_overlap_between_chunks():
    text = " ".join(str(i) for i in range(20))
    chunks = chunk_text(text, size=10, overlap=3)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]


def test_all_words_covered():
    words = [str(i) for i in range(53)]
    chunks = chunk_text(" ".join(words), size=10, overlap=2)
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(words)


def test_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=5, overlap=5)
