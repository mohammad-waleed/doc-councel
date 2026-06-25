"""
main.py
Responsible for initializing and running the FastAPI backend server,
setting up middleware, and mounting API routers.
"""

from fastapi import FastAPI


from backend.api.routes.upload import router as upload_router
from backend.api.routes.chat import router as chat_router
from tests.test_rag import router as test_rag_router

from backend.core.config import settings

app = FastAPI(title=settings.APP_TITLE)


app.include_router(upload_router)
app.include_router(chat_router)



