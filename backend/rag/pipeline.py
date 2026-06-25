"""
pipeline.py
End-to-end RAG pipeline coordinating retrieval of relevant documents
and generation of context-aware answers.
"""
from typing import Dict
from backend.rag.loader import load_pdf_document
from backend.rag.chunker import split_documents
from backend.rag.embedder import embed_chunks, get_embedding_model
from backend.rag.vectorstore import store_embeddings
from backend.rag.llm_model import get_llm
from backend.database.chroma import get_collection


def ingest_document(file_content: bytes, filename: str) -> int:
    """
    Ingestion pipeline: loads a PDF from memory, chunks it, embeds it, and stores
    the vectors in ChromaDB.

    Args:
        file_content (bytes): The PDF binary content.
        filename (str): The PDF filename (for metadata).

    Returns:
        int: The number of chunks stored.
    """
    # 1. Load the PDF directly from memory
    documents = load_pdf_document(file_content, filename) 

    # 2. Split into chunks
    chunks = split_documents(documents)

    # 3. Generate embeddings
    embeddings = embed_chunks(chunks)

    # 4. Store in ChromaDB
    stored_count = store_embeddings(chunks, embeddings)

    return stored_count


def query_rag(user_query: str) -> Dict[str, str]:
    """
    RAG query pipeline: embeds the user query, retrieves relevant chunks
    from ChromaDB, builds a prompt with context, and generates an answer
    using the LLM.

    Args:
        user_query (str): The user's question.

    Returns:
        Dict with keys:
            - response (str): The LLM-generated answer.
            - source (str): The source metadata of the retrieved chunks.
    """
    # 1. Embed the user query
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(user_query)

    # 2. Retrieve relevant chunks from ChromaDB
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )

    # 3. Build context from retrieved documents
    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_metadatas = results.get("metadatas", [[]])[0]

    context = "\n\n".join(retrieved_docs)

    # Build source info from metadata
    sources = []
    for meta in retrieved_metadatas:
        source_info = meta.get("source", "Unknown")
        page = meta.get("page", "N/A")
        sources.append(f"{source_info} (page {page})")
    source_str = "; ".join(dict.fromkeys(sources))  # deduplicate while preserving order

    # 4. Create the prompt with context
    prompt = (
        "You are a legal document assistant. Use the following context from legal "
        "documents to answer the user's question. If the answer is not found in the "
        "context, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n\n"
        "Answer:"
    )

    # 5. Generate response using the LLM
    llm = get_llm()
    result = llm.invoke(prompt)

    return {
        "response": result.content,
        "source": source_str,
    }
