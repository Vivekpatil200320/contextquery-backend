from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.file_parser import parse_document
from app.services.text_splitter import split_text
from app.services.embedding_service import embed_chunks_batch
from app.services.vector_store import store_chunks
from app.models.schemas import IngestResponse

router = APIRouter(prefix="/api", tags=["ingestion"])

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    try:
        document_id, text, safe_filename = await parse_document(file)
        chunks = split_text(text, document_id, safe_filename)

        if not chunks:
            raise HTTPException(status_code=422, detail="No chunks generated from document")

        embeddings = embed_chunks_batch(chunks)
        store_chunks(chunks, embeddings)

        return IngestResponse(
            document_id=document_id,
            filename=safe_filename,
            chunk_count=len(chunks),
            status="success",
            message=f"Ingested {len(chunks)} chunks successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
