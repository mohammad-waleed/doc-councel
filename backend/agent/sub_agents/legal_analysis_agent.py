# pyrefly: ignore [missing-import]
from langchain.agents import create_agent 
# pyrefly: ignore [missing-import]
from backend.rag.llm_model import get_llm
# pyrefly: ignore [missing-import]
from backend.models.schemas import ChatResponse
# pyrefly: ignore [missing-import]





llm = get_llm()


# a sub agent specialized for legal document analysis

def legal_analysis_sub_agent():
    """
    Creates the Legal Analysis sub-agent responsible for
    document-level legal analysis.

    The agent decides which tool to invoke based on the
    user's request.
    """
    from backend.agent.tools import summary_of_legal_doc, analyze_risky_clauses


    agent = create_agent(
        model=llm,
        tools=[
            summary_of_legal_doc,
            analyze_risky_clauses,
        ],
        response_format=ChatResponse,
        
        system_prompt="""
    You are a Legal Analysis Agent specialized in analyzing legal documents.

    Your responsibilities include:
    - Generating concise legal summaries.
    - Identifying risky, hidden, or unusual clauses.

    Available tools:

    1. summary_of_legal_doc
    Use ONLY when the user requests:
    - summary
    - overview
    - key points
    - simplified explanation

    2. analyze_risky_clauses
    Use ONLY when the user requests:
    - risky clauses
    - hidden clauses
    - unfair terms
    - legal risks

    Select the most appropriate tool based on the user's request.
    Do NOT invoke multiple tools unless the user explicitly requests multiple analyses.

    Always return your final response in the following JSON format:
    source is mandatory just when you analyze_risky_clauses tool is used.
    not for summary_of_legal_doc tool.
    {
        "response": "<final response as string>",
        "source": "<page number, heading, section, etc.>" or ""
    }

    Examples:

    Summary:
    {
        "response": "...",
        "source": ""
    }

    Risk Analysis:
    {
        "response": "...",
        "source": "analyze_risky_clauses"
    }
    """,
        )
    
    return agent


