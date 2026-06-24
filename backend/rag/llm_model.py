# initlize LLM
"""
LLM.py
Initializes the Groq LLM instance for the legal assistant.
"""
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

from backend.core.config import settings


def get_llm() -> ChatGroq:
    """
    Returns a ChatGroq LLM instance configured with the
    model and the Groq API key from settings.
    """
    llm = ChatGroq(model=settings.LLM_MODEL, api_key=settings.GROQ_API_KEY)
    return llm
