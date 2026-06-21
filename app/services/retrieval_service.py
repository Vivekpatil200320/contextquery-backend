from app.services.vector_store import get_collection
from app.services.embedding_service import get_embedder

def retrieve_chunks(question: str, top_k: int = 10) -> dict:
    embedder = get_embedder()
    query_embedding = embedder.embed_query(question)

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return {
        "documents": results["documents"][0] if results["documents"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "ids": results["ids"][0] if results["ids"] else []
    }