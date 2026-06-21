from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from app.core.config import settings

_embedder = None

def get_embedder() -> NVIDIAEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = NVIDIAEmbeddings(
            model="nvidia/llama-nemotron-embed-1b-v2",
            api_key=settings.nvidia_api_key
        )
    return _embedder

def embed_chunks_batch(chunks: list[dict]) -> list[list[float]]:
    """Single batched call — never loop one chunk per API call."""
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    return embedder.embed_documents(texts)