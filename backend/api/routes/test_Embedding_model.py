from typing import List, Union
from fastapi import APIRouter
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEndpointEmbeddings

router = APIRouter()

class EmbeddingResponse(BaseModel):
    embeddings: Union[List[float], str]

@router.get("/Embedding_model", response_model=EmbeddingResponse)
def test_embedding_model(prompt: str = "Test prompt for embeddings"):
    """Test endpoint for Hugging Face Embeddings using langchain-huggingface."""
    try:
        embeddings_model = HuggingFaceEndpointEmbeddings(model="BAAI/bge-m3")
        embeddings = embeddings_model.embed_query(prompt)
        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        return EmbeddingResponse(embeddings=f"Error: {str(e)}")
