from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="HackRx LLM Query-Retrieval System")

app.include_router(router, prefix="/api/v1")

