"""
chunker.py
Responsible for splitting extracted raw text into optimal chunks
using recursive text splitters for vector embedding.
"""
from typing import List
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

from backend.core.config import settings

def split_documents(documents: List[Document], chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
    """
    Splits a list of LangChain Document objects into smaller chunks.
    
    Args:
        documents (List[Document]): The loaded documents to be split.
        chunk_size (int): The maximum size of each chunk. Default from settings.
        chunk_overlap (int): The overlap between consecutive chunks. Default from settings.
        
    Returns:
        List[Document]: A list of chunked Document objects.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks
