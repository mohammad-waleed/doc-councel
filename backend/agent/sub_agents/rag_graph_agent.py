from typing import TypedDict
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END
from backend.agent.tools import retrieve_context, response_evaluator, modify_user_query

# ─────────────────────────────────────────────
# 1. STATE
# ─────────────────────────────────────────────
class RAGState(TypedDict):
    original_query: str
    current_query: str
    context: str
    response: str
    source: str
    confidence_score: float
    well_grounded: bool
    feedback: str
    iteration: int

# ─────────────────────────────────────────────
# 2. NODES
# ─────────────────────────────────────────────
def retrieve_node(state: RAGState) -> RAGState:
    result = retrieve_context.func(user_query=state["current_query"])
    return {
        **state,
        "response": result.get("response", ""),
        "context": result.get("context", ""),
        "source": result.get("source", ""),
    }

def evaluate_node(state: RAGState) -> RAGState:
    result = response_evaluator.func(
        user_query=state["original_query"],
        response=state["response"],
        context=state["context"],
    )
    return {
        **state,
        "well_grounded": result.get("well_grounded", False),
        "confidence_score": result.get("confidence_score", 0.0),
        "feedback": result.get("feedback", ""),
    }

def modify_node(state: RAGState) -> RAGState:
    improved_query = modify_user_query.func(
        original_query=state["current_query"],
        feedback=state["feedback"],
    )
    return {
        **state,
        "current_query": improved_query,
        "iteration": state["iteration"] + 1,
    }

def clarification_node(state: RAGState) -> RAGState:
    return {
        **state,
        "response": (
            "I tried to retrieve and verify an answer from the document, "
            "but the available context is not sufficient to provide a grounded, "
            "high-confidence response. Please modify or clarify your query."
        ),
        "source": "N/A",
    }

# ─────────────────────────────────────────────
# 3. CONDITIONAL EDGES
# ─────────────────────────────────────────────
def should_continue(state: RAGState) -> str:
    if state["well_grounded"] or state["confidence_score"] >= 7.0:
        return "end"
    if state["iteration"] >= 3:
        return "clarify"
    return "modify"

# ─────────────────────────────────────────────
# 4. BUILD GRAPH
# ─────────────────────────────────────────────
def build_rag_graph():
    graph = StateGraph(RAGState)

    # add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("modify", modify_node)
    graph.add_node("clarify", clarification_node)

    # edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "end": END,
            "clarify": "clarify",
            "modify": "modify",
        }
    )
    graph.add_edge("modify", "retrieve")   # loop back
    graph.add_edge("clarify", END)

    return graph.compile()

rag_graph = build_rag_graph()

# ─────────────────────────────────────────────
# 5. TOOL WRAPPER (for orchestrator to call)
# ─────────────────────────────────────────────
# pyrefly: ignore [missing-import]
from langchain.tools import tool

@tool
def rag_agent(user_query: str) -> str:
    """
    RAG subagent that retrieves context and answers queries from legal documents.
    Runs a stateful verification loop (up to 3 iterations) to ensure the answer
    is well-grounded and accurate.
    """
    result = rag_graph.invoke({
        "original_query": user_query,
        "current_query": user_query,
        "context": "",
        "response": "",
        "source": "",
        "confidence_score": 0.0,
        "well_grounded": False,
        "feedback": "",
        "iteration": 1,
    })

    return result["response"]