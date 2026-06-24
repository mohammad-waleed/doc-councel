"""
loader.py
Responsible for loading and extracting raw text from uploaded
PDF documents using PyPDFLoader and pdfplumber.
"""
from pathlib import Path
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader

from backend.core.config import settings

# Storage directory
STORAGE_DIR = Path(settings.STORAGE_DIR)

def load_pdf_document(filename: str):
    """
    Loads a PDF document from the storage directory using PyPDFLoader.
    
    Args:
        filename (str): The name of the PDF file (e.g., 'document.pdf').
        
    Returns:
        List[Document]: A list of LangChain Document objects containing the text and metadata.
    """
    file_path = STORAGE_DIR / filename
    
    # Check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(f"The file {filename} does not exist in the storage directory.")
        
    # Ensure it's a PDF format
    if file_path.suffix.lower() != '.pdf':
        raise ValueError(f"The file {filename} is not a PDF document.")
        
    # Initialize the PyPDFLoader with the file path
    loader = PyPDFLoader(str(file_path))
    
    # Load and return the documents (pages)
    documents = loader.load()
    return documents
