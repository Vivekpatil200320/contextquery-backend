from rank_bm25 import BM25Okapi
from app.services.vector_store import get_collection

_index: BM25Okapi | None = None
_corpus: list[dict] | None = None  # [{text, metadata}]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


_PAGE_SIZE = 250  # Chroma Cloud free tier caps get() limit at 300


def build_bm25_index() -> None:
    global _index, _corpus
    collection = get_collection()

    all_docs: list[str] = []
    all_metas: list[dict] = []
    offset = 0
    while True:
        result = collection.get(
            include=["documents", "metadatas"],
            limit=_PAGE_SIZE,
            offset=offset,
        )
        page_docs = result.get("documents") or []
        page_metas = result.get("metadatas") or []
        if not page_docs:
            break
        all_docs.extend(page_docs)
        all_metas.extend(page_metas)
        if len(page_docs) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    _corpus = [
        {"text": text, "metadata": meta}
        for text, meta in zip(all_docs, all_metas)
        if text
    ]
    _index = BM25Okapi([_tokenize(d["text"]) for d in _corpus])


def _ensure_index() -> tuple[BM25Okapi, list[dict]]:
    if _index is None or _corpus is None:
        build_bm25_index()
    return _index, _corpus  # type: ignore[return-value]


def bm25_search(query: str, top_k: int = 10) -> dict:
    index, corpus = _ensure_index()
    if not corpus:
        return {"documents": [], "metadatas": [], "ids": []}

    scores = index.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return {
        "documents": [corpus[i]["text"] for i in ranked],
        "metadatas": [corpus[i]["metadata"] for i in ranked],
        "ids": [f"bm25_{i}" for i in ranked],
    }
