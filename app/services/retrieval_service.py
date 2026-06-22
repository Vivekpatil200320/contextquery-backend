import concurrent.futures

from app.services.vector_store import get_collection
from app.services.embedding_service import get_embedder
from app.services.bm25_service import bm25_search
from app.core.config import settings


def _semantic_search(question: str, top_k: int = 10) -> dict:
    embedder = get_embedder()
    query_embedding = embedder.embed_query(question)
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return {
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "ids": results["ids"][0] if results["ids"] else [],
    }


def _reciprocal_rank_fusion(
    semantic: dict,
    bm25: dict,
    top_n: int = 15,
    k: int = 60,
) -> dict:
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, (text, meta) in enumerate(zip(semantic["documents"], semantic["metadatas"])):
        key = f"{meta.get('document_id', '')}::{meta.get('chunk_index', '')}"
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = {"text": text, "metadata": meta}

    for rank, (text, meta) in enumerate(zip(bm25["documents"], bm25["metadatas"])):
        key = f"{meta.get('document_id', '')}::{meta.get('chunk_index', '')}"
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = {"text": text, "metadata": meta}

    top_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]
    return {
        "documents": [doc_map[key]["text"] for key in top_keys],
        "metadatas": [doc_map[key]["metadata"] for key in top_keys],
        "ids": top_keys,
    }


def retrieve_chunks(question: str, top_k: int = 10, mode: str | None = None) -> dict:
    effective_mode = mode or settings.retrieval_mode

    if effective_mode == "hybrid":
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            semantic_future = executor.submit(_semantic_search, question, top_k)
            bm25_future = executor.submit(bm25_search, question, top_k)
        semantic = semantic_future.result()  # re-raises on failure — semantic is required
        try:
            bm25 = bm25_future.result()
        except Exception:
            bm25 = {"documents": [], "metadatas": [], "ids": []}
        return _reciprocal_rank_fusion(semantic, bm25)

    return _semantic_search(question, top_k)
