"""
analysis.py

API routes for document-level legal analysis.
These endpoints are triggered directly from UI actions
(e.g. Summary, Risky Clauses).
"""
import json
import re
from fastapi import APIRouter, Depends, HTTPException

from backend.models.schemas import (
    AnalysisRequest,
    ChatResponse,
)

from backend.agent.sub_agents.legal_analysis_agent import legal_analysis_sub_agent 
from backend.api.dependencies import get_active_document

router = APIRouter(prefix="/analysis", tags=["Legal Analysis"])


@router.post("/analysis", response_model=ChatResponse)
def analyze_document(
    request: AnalysisRequest,
    active_doc: dict = Depends(get_active_document),
):
    """
    
    """

    try:
        agent = legal_analysis_sub_agent()

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": request.task
                }
            ]
        })
        
        return result["structured_response"]


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))