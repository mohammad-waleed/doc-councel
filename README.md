# DocCounsel

DocCounsel is an AI-powered legal document assistant built on Retrieval-Augmented Generation (RAG). It allows freelancers, small business owners, and startup founders to upload legal documents — contracts, NDAs, service agreements, and tax notices — and ask questions in plain English, without needing a lawyer. It analyzes these documents to provide legal consultation.
## Tech Stack
- **Backend API**: FastAPI, Uvicorn
- **LLM Orchestration**: LangChain, LangChain Community, LangChain Core
- **Document Loading & Extraction**: PyPDF, PDFPlumber
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence-Transformers
- **User Interface**: Streamlit
- **Utilities**: Pydantic, python-dotenv, requests

## Folder Structure Overview

```
doc-counsel/
├── backend/
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── chat.py
│   │       ├── upload.py
│   │       └── health.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── prompts.py
│   │   └── guardrails.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── vectorstore.py
│   │   └── pipeline.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py
│   │   └── agent.py
│   └── models/
│       ├── __init__.py
│       └── schemas.py
│
├── frontend/
│   └── app.py
│
├── data/
│   └── vectorstore/
│
├── tests/
│   ├── __init__.py
│   ├── test_rag.py
│   ├── test_api.py
│   └── sample_contracts/
│
├── .env.example
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository_url>
cd doc-counsel
```

### 2. Create a virtual environment
Ensure you have `uv` installed, then run:
```bash
uv venv
```
Activate the environment:
- **Windows**: `.venv\Scripts\activate`
- **macOS/Linux**: `source .venv/bin/activate`

### 3. Install dependencies
```bash
uv pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the `.env.example` file to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### 5. Run the Application
Start the backend API server:
```bash
uvicorn backend.main:app --reload
```
Start the Streamlit frontend dashboard:
```bash
streamlit run frontend/app.py
```

## Screenshots
*(Placeholder for application screenshots)*
