# doc-councel
DocCounsel is an AI-powered legal document assistant built on 
Retrieval-Augmented Generation (RAG). It allows freelancers, 
small business owners, and startup founders to upload legal 
documents — contracts, NDAs, service agreements, and tax notices 
— and ask questions in plain English, without needing a lawyer.

Unlike general-purpose AI chatbots, DocCounsel grounds every 
answer strictly in the uploaded document. It retrieves the most 
relevant clauses using semantic search, generates clear and 
concise responses via a Large Language Model, and always cites 
the exact source clause — so users can verify every answer 
themselves.

Built with Python, LangChain, FastAPI, FAISS, and Streamlit, 
DocCounsel demonstrates a production-grade RAG pipeline with 
an agent layer for multi-tool reasoning, guardrails for 
hallucination prevention, and a conversational interface 
designed for non-technical users.
