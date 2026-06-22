from pydantic import BaseModel, Field
from typing import Literal

class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: Literal["success", "failed"]
    message: str = ""

class ErrorResponse(BaseModel):
    detail: str
    error_type: str

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=20)
    rerank_top_n: int = Field(default=5, ge=1, le=10)
    retrieval_mode: Literal["semantic", "hybrid"] | None = None

class SourceChunk(BaseModel):
    filename: str
    chunk_index: int
    text: str
    relevance_score: float | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    chunks_retrieved: int
    chunks_used: int

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunk_count: int

class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int

class DeleteResponse(BaseModel):
    document_id: str
    chunks_deleted: int
    status: str