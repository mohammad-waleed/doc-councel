"""
loader.py
Responsible for loading and extracting raw text from uploaded
PDF documents using PyPDFLoader and pdfplumber.
"""
import io
# pyrefly: ignore [missing-import]
from pypdf import PdfReader
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

def load_pdf_document(file_content: bytes, filename: str):
    """
    Loads a PDF document directly from memory (bytes) without saving to disk.
    
    Args:
        file_content (bytes): The binary content of the PDF.
        filename (str): The name of the file (for metadata).
        
    Returns:
        List[Document]: A list of LangChain Document objects containing the text and metadata.
    """
    # Read the PDF from bytes in memory
    reader = PdfReader(io.BytesIO(file_content))
    documents = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # Create a LangChain Document for each page
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "page": i}
                )
            )
            
    if not documents:
        raise ValueError(f"The file {filename} contains no extractable text.")
        
    return documents
