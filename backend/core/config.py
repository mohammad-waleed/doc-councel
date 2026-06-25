"""
config.py
Configuration manager for handling environment variables,
API keys, model parameters, and global server settings.
"""
from pathlib import Path
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

class Settings(BaseSettings):
    # --- API Keys ---
    GROQ_API_KEY: str
    HUGGINGFACEHUB_API_TOKEN: str

    # --- Model Names ---
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    # --- Paths ---
    STORAGE_DIR: str = "backend/storage"
    CHROMA_PERSIST_DIR: str = "backend/database/chroma_store"

    # --- ChromaDB ---
    CHROMA_COLLECTION_NAME: str = "legal_documents"

    # --- Chunker ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 70

    # --- Upload ---
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB

    # --- Server ---
    APP_TITLE: str = "DocCounsel API"

    model_config = SettingsConfigDict(
        env_file=_ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
