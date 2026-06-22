import json as json_lib
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services.retrieval_service import retrieve_chunks
from app.services.rerank_service import rerank_chunks
from app.services.llm_service import generate_grounded_answer, stream_grounded_answer
from app.core.config import settings
from app.core.observability import start_trace_or_span

router = APIRouter(prefix="/api", tags=["query"])


def _narrative_order(chunks: list[dict]) -> list[dict]:
    """Within each document, sort chunks by chunk_index (narrative order).
    Cross-document ordering is preserved by the first appearance of each document."""
    seen: dict[str, int] = {}   # document_id -> insertion order
    groups: dict[str, list[dict]] = {}
    for chunk in chunks:
        doc_id = chunk["metadata"].get("document_id", "")
        if doc_id not in seen:
            seen[doc_id] = len(seen)
            groups[doc_id] = []
        groups[doc_id].append(chunk)
    result: list[dict] = []
    for doc_id in sorted(seen, key=lambda d: seen[d]):
        result.extend(sorted(groups[doc_id], key=lambda c: c["metadata"].get("chunk_index", 0)))
    return result


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    with start_trace_or_span(name="rag-query", as_type="span", input={"question": request.question}) as trace:
        try:
            # Retrieval span
            t0 = time.perf_counter()
            with start_trace_or_span(
                name="retrieval",
                as_type="retriever",
                input={"question": request.question, "top_k": request.top_k}
            ) as retrieval_span:
                retrieved = retrieve_chunks(
                    request.question,
                    top_k=request.top_k,
                    mode=request.retrieval_mode,
                )
                retrieval_ms = round((time.perf_counter() - t0) * 1000)
                retrieval_span.update(
                    output={"chunks_retrieved": len(retrieved["documents"])},
                    metadata={"latency_ms": retrieval_ms}
                )

            if not retrieved["documents"]:
                return QueryResponse(
                    answer="No documents have been ingested yet — upload a document before asking questions.",
                    sources=[],
                    chunks_retrieved=0,
                    chunks_used=0
                )

            # Reranking span
            t0 = time.perf_counter()
            with start_trace_or_span(
                name="reranking",
                as_type="span",
                input={"chunks_in": len(retrieved["documents"])}
            ) as reranking_span:
                reranked = rerank_chunks(
                    request.question,
                    retrieved["documents"],
                    retrieved["metadatas"],
                    top_n=request.rerank_top_n
                )
                rerank_ms = round((time.perf_counter() - t0) * 1000)
                reranking_span.update(
                    output={
                        "chunks_out": len(reranked),
                        "top_scores": [round(c.get("score") or 0, 3) for c in reranked[:3]]
                    },
                    metadata={"latency_ms": rerank_ms}
                )

            reranked = _narrative_order(reranked)

            # Generation span
            t0 = time.perf_counter()
            answer = await generate_grounded_answer(request.question, reranked)
            generation_ms = round((time.perf_counter() - t0) * 1000)

            # Update trace with output & metadata
            trace.update(
                output={"answer": answer, "sources_used": len(reranked)},
                metadata={
                    "total_latency_ms": retrieval_ms + rerank_ms + generation_ms,
                    "chunks_retrieved": len(retrieved["documents"]),
                    "chunks_used": len(reranked)
                }
            )

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
            trace.update(
                level="ERROR",
                status_message=str(e),
                metadata={"error": str(e)}
            )
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    retrieved = retrieve_chunks(
        request.question,
        top_k=request.top_k,
        mode=request.retrieval_mode,
    )

    if not retrieved["documents"]:
        async def empty_gen():
            yield "No documents have been ingested yet — upload a document before asking questions."
        return StreamingResponse(empty_gen(), media_type="text/plain")

    reranked = _narrative_order(rerank_chunks(
        request.question,
        retrieved["documents"],
        retrieved["metadatas"],
        top_n=request.rerank_top_n
    ))

    async def event_generator():
        with start_trace_or_span(name="rag-query-stream", as_type="span", input={"question": request.question}) as trace:
            try:
                # Log retrieval span retrospectively since it happened before the stream generator started
                with start_trace_or_span(
                    name="retrieval",
                    as_type="retriever",
                    input={"question": request.question, "top_k": request.top_k}
                ) as retrieval_span:
                    retrieval_span.update(output={"chunks_retrieved": len(retrieved["documents"])})

                # Log rerank span retrospectively
                with start_trace_or_span(
                    name="reranking",
                    as_type="span",
                    input={"chunks_in": len(retrieved["documents"])}
                ) as reranking_span:
                    reranking_span.update(
                        output={
                            "chunks_out": len(reranked),
                            "top_scores": [round(c.get("score") or 0, 3) for c in reranked[:3]]
                        }
                    )

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

                # Then stream the answer token by token (internally traced in llm_service)
                full_answer = []
                t0 = time.perf_counter()

                async for token in stream_grounded_answer(request.question, reranked):
                    token_payload = {"type": "token", "data": token}
                    yield f"data: {json_lib.dumps(token_payload)}\n\n"
                    full_answer.append(token)

                generation_ms = round((time.perf_counter() - t0) * 1000)

                trace.update(
                    output={"answer": "".join(full_answer), "sources_used": len(reranked)},
                    metadata={
                        "chunks_retrieved": len(retrieved["documents"]),
                        "chunks_used": len(reranked),
                        "generation_latency_ms": generation_ms
                    }
                )

                yield f"data: {json_lib.dumps({'type': 'done'})}\n\n"
            except Exception as e:
                trace.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={"error": str(e)}
                )
                yield f"data: {json_lib.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")