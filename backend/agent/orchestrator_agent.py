"""
orchestrator_agent.py

Configures and initializes the orchestrator agent using LangChain,
enabling multi-step reasoning, tool usage, and task delegation.
"""

# pyrefly: ignore [missing-import]
from langchain.agents import create_agent
# pyrefly: ignore [missing-import]
from backend.models.schemas import ChatResponse
from backend.rag.llm_model import get_llm
from backend.agent.tools import (
    legal_document_validation,
    ingest_document,
    legal_analysis_agent,
)
from backend.agent.sub_agents.rag_graph_agent import rag_agent





def doc_counsel_orchestrator():
    """
    Initializes and returns the Doc-Counsel orchestrator agent.

    This agent coordinates document verification, ingestion, and querying tasks
    using specialized tools.
    """
    # Initialize the main LLM
    llm = get_llm()

    # Define the list of tools the agent can access
    tools = [
        legal_analysis_agent,
        rag_agent,
    ]

    # Construct the agent using the modern LangChain v1+ create_agent factory
    agent = create_agent(
        model=llm,
        tools=tools,
        response_format=ChatResponse,
        system_prompt=(
            "You are Doc-Counsel, an expert legal AI assistant. "
            "You help users manage, validate, and analyze legal documents. "
            "Use the provided tools to ingest documents, validate if a document is a legal document, "
            "summarize documents, analyze risky clauses, and retrieve context/answers from ingested documents. "
            "Be precise, professional, and helpful."
        ),
    )

    return agent
