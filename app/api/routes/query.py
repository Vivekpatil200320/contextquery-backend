import json as json_lib

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services.retrieval_service import retrieve_chunks
from app.services.rerank_service import rerank_chunks
from app.services.llm_service import generate_grounded_answer, stream_grounded_answer

router = APIRouter(prefix="/api", tags=["query"])

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        retrieved = retrieve_chunks(request.question, top_k=request.top_k)

        if not retrieved["documents"]:
            return QueryResponse(
                answer="No documents have been ingested yet — upload a document before asking questions.",
                sources=[],
                chunks_retrieved=0,
                chunks_used=0
            )

        reranked = rerank_chunks(
            request.question,
            retrieved["documents"],
            retrieved["metadatas"],
            top_n=request.rerank_top_n
        )

        answer = await generate_grounded_answer(request.question, reranked)

        sources = [
            SourceChunk(
                filename=c["metadata"].get("filename", "unknown"),
                chunk_index=c["metadata"].get("chunk_index", -1),
                text=c["text"][:200],
                relevance_score=c.get("score")
            )
            for c in reranked
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            chunks_retrieved=len(retrieved["documents"]),
            chunks_used=len(reranked)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    retrieved = retrieve_chunks(request.question, top_k=request.top_k)

    if not retrieved["documents"]:
        async def empty_gen():
            yield "No documents have been ingested yet — upload a document before asking questions."
        return StreamingResponse(empty_gen(), media_type="text/plain")

    reranked = rerank_chunks(
        request.question,
        retrieved["documents"],
        retrieved["metadatas"],
        top_n=request.rerank_top_n
    )

    async def event_generator():
        # First, send sources as a single SSE event so the frontend can render citations immediately
        sources_payload = {
            "type": "sources",
            "data": [
                {
                    "filename": c["metadata"].get("filename", "unknown"),
                    "chunk_index": c["metadata"].get("chunk_index", -1),
                    "text": c["text"][:200]
                }
                for c in reranked
            ]
        }
        yield f"data: {json_lib.dumps(sources_payload)}\n\n"

        # Then stream the answer token by token
        async for token in stream_grounded_answer(request.question, reranked):
            token_payload = {"type": "token", "data": token}
            yield f"data: {json_lib.dumps(token_payload)}\n\n"

        yield f"data: {json_lib.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")