"""
test_rag.py
Unit and integration tests for document loaders, text chunkers,
embeddings, and vectorstore retrieval logic.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.rag.pipeline import ingest_document, query_rag

router = APIRouter()


class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_stored: int


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str
    source: str


@router.post("/test/rag/ingest", response_model=IngestResponse)
def test_rag_ingest(filename: str):
    """
    Test endpoint to ingest a PDF document through the full RAG pipeline.
    The file must already exist in backend/storage/.
    """
    try:
        chunks_stored = ingest_document(filename)
        return IngestResponse(
            message="Document ingested successfully",
            filename=filename,
            chunks_stored=chunks_stored,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/rag/query", response_model=QueryResponse)
def test_rag_query(request: QueryRequest):
    """
    Test endpoint to query the RAG pipeline with a user question.
    Documents must be ingested first.
    """
    try:
        result = query_rag(request.query)
        return QueryResponse(
            response=result["response"],
            source=result["source"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
