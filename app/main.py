from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.database import init_db

app = FastAPI(
    title="Multimodal RAG API",
    description="RAG system for PDFs, CSVs, images, and code files",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup():
    init_db()

@app.get("/health")
async def health():
    return {"status": "ok"}
