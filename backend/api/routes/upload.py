"""
upload.py
API routes for uploading legal documents, extracting raw content,
chunking, embedding, and storing in ChromaDB.
"""
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from backend.core.config import settings
from backend.rag.pipeline import ingest_document

router = APIRouter()

MAX_FILE_SIZE = settings.MAX_FILE_SIZE



@router.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a legal document (PDF only).
    Maximum file size: 10 MB.
    """
    # 1. Check if the file extension is .pdf
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF documents are allowed."
        )
    
    # 2. Check the content type
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content type. Only PDF documents are allowed."
        )

    # 3. Read the file content to check the size
    file_content = await file.read()
    
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the 10 MB limit."
        )
    
    # 4. Ingest the document directly from memory
    try:
        chunks_stored = ingest_document(file_content, file.filename)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File uploaded but ingestion failed: {str(e)}"
        )
    
    return {
        "filename": file.filename,
        "message": "Document uploaded and ingested successfully (memory-only)",
        "size_bytes": len(file_content),
        "chunks_stored": chunks_stored
    }

