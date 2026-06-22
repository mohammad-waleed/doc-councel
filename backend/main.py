"""
main.py
Responsible for initializing and running the FastAPI backend server,
setting up middleware, and mounting API routers.
"""


from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="DocCounsel API")


class UserResponse(BaseModel):
    status: str
    message: str


@app.get("/health", response_model=UserResponse)
def health(name: str = "User"):
    """Health check — accepts an optional name and returns status and message."""
    return UserResponse(
        status="ok",
        message=f"Hello {name}, DocCounsel API is running successfully.",
    )