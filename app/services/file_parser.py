import re
import uuid
from pathlib import Path
from fastapi import UploadFile, HTTPException
from pypdf import PdfReader
from docx import Document as DocxDocument

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE_MB = 20

def sanitize_filename(filename: str) -> str:
    # Strip path traversal attempts and unsafe characters
    name = filename.replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^\w\s\-\.\(\)]", "", name)
    return name[:255]

async def validate_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only PDF and DOCX are allowed."
        )

    file.file.seek(0, 2)
    size_bytes = file.file.tell()
    file.file.seek(0)

    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max is {MAX_FILE_SIZE_MB}MB."
        )

def extract_text_from_pdf(file_bytes: bytes) -> str:
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(text_parts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    import io
    doc = DocxDocument(io.BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

async def parse_document(file: UploadFile) -> tuple[str, str, str]:
    """Returns (document_id, extracted_text, safe_filename)"""
    await validate_file(file)
    file_bytes = await file.read()
    ext = Path(file.filename).suffix.lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file_bytes)
    elif ext == ".docx":
        text = extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text found — file may be scanned/image-based."
        )

    document_id = str(uuid.uuid4())
    safe_filename = sanitize_filename(file.filename)
    return document_id, text, safe_filename