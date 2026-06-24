from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class UserResponse(BaseModel):
    status: str
    message: str

@router.get("/API_health", response_model=UserResponse)
def health(name: str = "User"):
    """Health check — accepts an optional name and returns status and message."""
    return UserResponse(
        status="ok",
        message=f"Hello {name}, DocCounsel API is running successfully.",
    )
