"""
chat.py
API routes for handling real-time chat interactions and retrieving
legal advice from the RAG assistant/agent.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.rag.pipeline import query_rag

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    source: str

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Endpoint to interact with the legal assistant.
    Takes a user query and returns a context-aware answer based on ingested documents.
    """
    try:
        result = query_rag(request.query)
        return ChatResponse(
            response=result["response"],
            source=result["source"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
