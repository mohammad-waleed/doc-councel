"""
tools.py
Custom tools for the AI agent, such as searching the vectorstore
or fetching web references for specific legal terms.
"""

from backend.models.schemas import RiskyClausesResponse
from backend.models.schemas import ChatResponse
from langchain.tools import tool  #pyright: ignore
from langchain_core.prompts import PromptTemplate  #pyright: ignore
from backend.rag.llm_model import get_llm
from backend.models.schemas import LegalDocumentValidationResponse, ResponseEvaluation
from backend.rag.pipeline import ingest_document as rag_ingest_document
from backend.rag.pipeline import query_rag as pipeline_query_rag
from backend.api.dependencies import active_document_text
from backend.api.dependencies import active_document_name

# pyrefly: ignore [missing-import]
from backend.agent.sub_agents.legal_analysis_agent import legal_analysis_sub_agent 

# tools for Orchestrator (Doc-Counsel Agent)

def legal_document_validation(document_text: str) -> dict:
    """
    Agent tool to analyze a document and determine if it is a legal document,
    its type, confidence score, and a short summary.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(LegalDocumentValidationResponse)
    
    prompt = """
you are Legal Documents Analzer who read documents uploaded by users.
you have to read this document and decide whther this document is legal document means such as :

Legal Documents may Inlcude:


* Business and Commercial Documents

  * Terms and conditions
  * Policies
  * Invoices
  * Purchase orders
  * Lease or rental agreements
  * Business correspondence containing legal obligations


* Contracts

  * Freelancer contracts
  * Employment/job contracts
  * Service agreements
  * Vendor contracts
  * Small business contracts

* Agreements

  * Business partnership agreements
  * Startup founder agreements
  * Non-disclosure agreements (NDAs)
  * Collaboration agreements
  * Memorandums of understanding (MOUs)

* Legal and Official Notices

  * Legal notices
  * Demand notices
  * Tax notices
  * Regulatory notices
  * Business compliance notices
  * Notices received by merchants, shopkeepers, or small businesses


-- you give confident_score out of 10.0
-- Indentify the type of doc 
-- legal yes/no 
(yes when document is from above type, no when it is of other type irrirelivant)
-- a short summary of the document

Document Content:
{document}
"""
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | structured_llm
    
    result = chain.invoke({"document": document_text})
    
    return {
        "legal_document": result.legal_document,
        "document-type": result.document_type,
        "confident_score": result.confident_score,
        "Description": result.Description
    }




def ingest_document(file_content: bytes, filename: str) -> int:
    """
    Ingest a document's binary content and its filename into the vector store.
    Loads the document, chunks it, embeds it, and stores the vectors in ChromaDB.
    """
    return rag_ingest_document(file_content, filename)


#tools for Legal Analysis Agent
@tool
def summary_of_legal_doc() -> str:
    """
    Generate a simple, easy-to-understand summary of the active legal document.
    """
    
    
    document_text = active_document_text.get()
    filename = active_document_name.get()
    # If context vars are not set (e.g., tool called outside of a request), load the most recent uploaded PDF directly.
    if not document_text:
        # Fallback: load the latest PDF from storage directory
        from backend.core.config import settings
        import os
        from backend.rag.loader import load_pdf_document
        storage_dir = settings.STORAGE_DIR
        if not os.path.isdir(storage_dir):
            return "No active document found. Please upload a PDF document first."
        pdf_files = [f for f in os.listdir(storage_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            return "No active document found. Please upload a PDF document first."
        pdf_files.sort(key=lambda x: os.path.getmtime(os.path.join(storage_dir, x)), reverse=True)
        active_file = pdf_files[0]
        file_path = os.path.join(storage_dir, active_file)
        with open(file_path, 'rb') as f:
            content = f.read()
        docs = load_pdf_document(content, active_file)
        document_text = "\n\n".join([doc.page_content for doc in docs])
        filename = active_file

    if not document_text:
        return "No active document found. Please upload a PDF document first."

    llm = get_llm()
    prompt = f"""
You are an expert legal counsel who excels at simplifying complex legal texts.
Analyze the following document and write a summary using easy-to-understand, plain words.
Break down the main purpose of the document, key rights, key obligations, and any critical deadlines or risks.

Document Filename: {filename}
Document Content:
{document_text}

Provide the summary in a clear, well-structured format.
"""
    result = llm.invoke(prompt)
    return result.content


@tool
def analyze_risky_clauses() -> str:
    """
    Analyze the active legal document to identify sensitive, hidden, risky, or potentially dangerous clauses.
    """
    
    
    document_text = active_document_text.get()
    filename = active_document_name.get()
    # If context vars are not set (e.g., tool called outside of a request), load the latest PDF directly.
    if not document_text:
        from backend.core.config import settings
        import os
        from backend.rag.loader import load_pdf_document
        storage_dir = settings.STORAGE_DIR
        if not os.path.isdir(storage_dir):
            return "No active document found. Please upload a PDF document first."
        pdf_files = [f for f in os.listdir(storage_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            return "No active document found. Please upload a PDF document first."
        pdf_files.sort(key=lambda x: os.path.getmtime(os.path.join(storage_dir, x)), reverse=True)
        active_file = pdf_files[0]
        file_path = os.path.join(storage_dir, active_file)
        with open(file_path, 'rb') as f:
            content = f.read()
        docs = load_pdf_document(content, active_file)
        document_text = "\n\n".join([doc.page_content for doc in docs])
        filename = active_file

    if not document_text:
        return "No active document found. Please upload a PDF document first."

    llm = get_llm()
    prompt = f"""
You are a highly experienced legal auditor. Your task is to critically analyze the legal document provided below to identify:
1. Sensitive Clauses: Sections containing critical commitments, confidentiality, or data sharing.
2. Hidden Clauses: Unusually phrased or buried terms that might escape casual reading but carry significant legal weight.
3. Risky Clauses: High-liability terms, indemnification, unfavorable dispute resolution forums, or severe penalties.
4. Potentially Dangerous Terms: Provisions that could become problematic or costly in the near future (e.g., automatic renewals with price hikes, unilateral amendment rights, extreme termination penalties).

Document Filename: {filename}
Document Content:
{document_text}

Provide a detailed analysis, highlighting each risky clause, explaining why it is risky or dangerous, and suggesting how to negotiate or mitigate the risk.
"""
    structured_llm = llm.with_structured_output(RiskyClausesResponse)
    result = structured_llm.invoke(prompt)
    
    return RiskyClausesResponse(
        risky_clauses=result.risky_clauses,
        source=result.source
    )


# tools for RAG Agent
@tool
def retrieve_context(user_query: str) -> dict:
    """
    Agent tool to query the legal document assistant. retrieve context from the document.
    """
    return pipeline_query_rag(user_query)


@tool
def response_evaluator(user_query: str, response: str, context: str) -> dict:
    """
    Agent tool to critically evaluate if the generated response is well grounded
    in the retrieved context for the user query.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(ResponseEvaluation)

    prompt = f"""
You are a meticulous legal quality controller. Evaluate the following response for a user query against the retrieved context.

User Query: {user_query}
Retrieved Context: {context}
Generated Response: {response}

Analyze:
1. Is the response fully grounded in and supported by the retrieved context?
2. Are there any hallucinations, assumptions, or ungrounded claims?
3. What critical details are missing or incorrect?

Give a confidence score out of 10.0 and detailed feedback explaining your evaluation.
"""
    result = structured_llm.invoke(prompt)
    return {
        "well_grounded": result.well_grounded,
        "confidence_score": result.confidence_score,
        "feedback": result.feedback,
    }


@tool
def modify_user_query(original_query: str, feedback: str) -> str:
    """
    Agent tool to rewrite/modify the user's query to make it more precise
    and better suited for document retrieval, based on evaluation feedback.
    """
    llm = get_llm()
    prompt = f"""
You are an expert search query optimizer. Rewrite the following query to improve document retrieval in a RAG pipeline.

Original Query: {original_query}
Feedback on previous search attempt: {feedback}

Based on the feedback, rewrite the query to target the missing information.
Provide ONLY the rewritten query without any preamble or explanation.
"""
    result = llm.invoke(prompt)
    return result.content.strip()



@tool
def legal_analysis_agent(query: str) -> dict:
    """
    Delegate legal analysis requests to the Legal Analysis Agent.
    """

    agent = legal_analysis_sub_agent()

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ]
    })
    return result["structured_response"]




