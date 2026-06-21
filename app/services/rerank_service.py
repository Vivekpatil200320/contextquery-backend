from langchain_nvidia_ai_endpoints import NVIDIARerank
from app.core.config import settings

_reranker = None

def get_reranker() -> NVIDIARerank:
    global _reranker
    if _reranker is None:
        _reranker = NVIDIARerank(
            model="nvidia/llama-nemotron-rerank-1b-v2",
            api_key=settings.nvidia_api_key
        )
    return _reranker

def rerank_chunks(question: str, documents: list[str], metadatas: list[dict], top_n: int = 4) -> list[dict]:
    if not documents:
        return []

    reranker = get_reranker()

    # NVIDIARerank expects LangChain Document-like objects
    from langchain_core.documents import Document
    docs = [Document(page_content=text, metadata=meta) for text, meta in zip(documents, metadatas)]

    reranked = reranker.compress_documents(query=question, documents=docs)

    results = []
    for doc in reranked[:top_n]:
        results.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": doc.metadata.get("relevance_score")
        })
    return results