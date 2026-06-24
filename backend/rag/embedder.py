"""
embedder.py
Generates vector representations of document chunks using sentence-transformers.
"""
from typing import List
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEndpointEmbeddings
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

from backend.core.config import settings


def get_embedding_model() -> HuggingFaceEndpointEmbeddings:
    """
    Returns a HuggingFaceEndpointEmbeddings instance configured
    with the embedding model and the HF token from settings.
    """
    embedding_model = HuggingFaceEndpointEmbeddings(
        model=settings.EMBEDDING_MODEL,
        huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
    )
    return embedding_model


def embed_chunks(chunks: List[Document]) -> List[List[float]]:
    """
    Converts a list of document chunks into vector embeddings.

    Args:
        chunks (List[Document]): The chunked documents from the splitter.

    Returns:
        List[List[float]]: A list of embedding vectors, one per chunk.
    """
    embedding_model = get_embedding_model()

    # Extract the text content from each chunk
    texts = [chunk.page_content for chunk in chunks]

    # Generate embeddings for all texts
    embeddings = embedding_model.embed_documents(texts)
    return embeddings 

    
