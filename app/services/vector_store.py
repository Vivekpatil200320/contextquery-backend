import chromadb
from typing import Any
from app.core.config import settings

_client: Any = None
_collection: Any = None

def get_collection() -> Any:
    global _client, _collection
    if _client is None:
        # CloudClient does not take a path — auth is via API key + tenant + database
        _client = chromadb.CloudClient(
            api_key=settings.chroma_api_key,
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
        )
        # get_or_create_collection is safe — idempotent on Cloud too
        _collection = _client.get_or_create_collection(settings.chroma_collection)
    return _collection

def store_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    collection = get_collection()
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in chunks]
    )

def list_documents() -> list[dict]:
    collection = get_collection()
    all_data = collection.get(include=["metadatas"])

    docs: dict[str, dict] = {}
    for meta in all_data.get("metadatas", []):
        if not isinstance(meta, dict):
            continue
        doc_id = meta.get("document_id")
        if not isinstance(doc_id, str):
            continue

        if doc_id not in docs:
            docs[doc_id] = {
                "document_id": doc_id,
                "filename": meta.get("filename", "unknown"),
                "chunk_count": 0
            }
        docs[doc_id]["chunk_count"] += 1

    return list(docs.values())

def delete_document(document_id: str) -> int:
    collection = get_collection()
    existing = collection.get(where={"document_id": document_id})

    if not existing or not existing.get("ids"):
        return 0

    collection.delete(where={"document_id": document_id})
    return len(existing["ids"])