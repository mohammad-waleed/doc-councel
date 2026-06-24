"""
tests_main.py
Standalone FastAPI app for testing the RAG pipeline endpoints.
Run with: uvicorn tests.tests_main:app --reload  (from project root)
"""
import sys
import os

# Add project root to path so imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from tests.test_rag import router as test_rag_router
from tests.test_API_health import router as health_router
from tests.test_LLM_response import router as llm_router
from tests.test_Embedding_model import router as embedding_router

app = FastAPI(title="DocCounsel RAG Tests")

app.include_router(test_rag_router)
app.include_router(health_router)
app.include_router(llm_router)
app.include_router(embedding_router)



