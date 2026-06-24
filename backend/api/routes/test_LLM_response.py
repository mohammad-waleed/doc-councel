from fastapi import APIRouter
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

router = APIRouter()

class LLMResponse(BaseModel):
    response: str

@router.get("/Groq_LLM", response_model=LLMResponse)
def test_groq_llm(prompt: str = "Say hello!"):
    """Test endpoint for Groq LLM using langchain-groq."""
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile")
        result = llm.invoke(prompt)
        return LLMResponse(response=result.content)
    except Exception as e:
        return LLMResponse(response=f"Error: {str(e)}")
