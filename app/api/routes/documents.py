from fastapi import APIRouter, HTTPException
from app.models.schemas import DocumentListResponse, DocumentInfo, DeleteResponse
from app.services.vector_store import list_documents, delete_document

router = APIRouter(prefix="/api", tags=["documents"])

@router.get("/documents", response_model=DocumentListResponse)
async def get_documents():
    docs = list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total_documents=len(docs),
        total_chunks=sum(d["chunk_count"] for d in docs)
    )

@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def remove_document(document_id: str):
    deleted_count = delete_document(document_id)

    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    return DeleteResponse(
        document_id=document_id,
        chunks_deleted=deleted_count,
        status="deleted"
    )
