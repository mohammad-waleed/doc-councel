"""
upload.py
API routes for uploading legal documents, extracting raw content,
chunking, embedding, and storing in ChromaDB.
"""
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
STORAGE_DIR = Path("backend/storage")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


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
    
    # Reset the file pointer if it needs to be read again in further processing
    await file.seek(0)
    
    # 4. Save the file to the storage directory
    file_path = STORAGE_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    return {
        "filename": file.filename,
        "message": "Document uploaded and saved successfully",
        "size_bytes": len(file_content),
        "saved_path": str(file_path)
    }
