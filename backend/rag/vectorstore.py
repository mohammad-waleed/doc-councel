"""
vectorstore.py
Manages connections to ChromaDB, indexing text chunks, and running
semantic search queries.
"""
import uuid
from typing import List
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

from backend.database.chroma import get_collection


def store_embeddings(chunks: List[Document], embeddings: List[List[float]]) -> int:
    """
    Stores pre-computed embeddings and their corresponding document chunks
    into the ChromaDB vector database.

    Args:
        chunks (List[Document]): The chunked documents.
        embeddings (List[List[float]]): The embedding vectors (one per chunk).

    Returns:
        int: The number of chunks stored.
    """
    # Get the ChromaDB collection
    collection = get_collection()

    # Prepare data for ChromaDB
    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Add to the collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(chunks)
