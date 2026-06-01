import time
import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Document, QueryHistory
from app.ingestion.embedder import EmbeddingManager
from app.ingestion.ingestor import ingest_file
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.generator import generate_answer
from app.api.schemas import QueryRequest, QueryResponse, DocumentResponse
from app.config import settings

router = APIRouter()

# Single shared embedder instance (model loads once)
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingManager()
    return _embedder

@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and ingest a file (PDF, CSV, or code file)."""
    allowed = {".pdf", ".csv", ".py", ".js", ".ts", ".java", ".cpp", ".go", ".rs", ".md"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Save uploaded file to disk
    upload_path = Path(settings.upload_dir) / file.filename
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        embedder = get_embedder()
        document = ingest_file(str(upload_path), db, embedder)
        return {
            "message": "File ingested successfully",
            "document_id": document.id,
            "filename": document.filename,
            "file_type": document.file_type.value,
            "status": document.status,
            "metadata": document.metadata_,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """Query across all ingested documents."""
    embedder = get_embedder()
    retriever = HybridRetriever(db, embedder)

    # Time the retrieval
    t0 = time.time()
    retrieval = retriever.retrieve_with_context(
        request.query,
        top_k=request.top_k,
    )
    retrieval_latency = (time.time() - t0) * 1000

    # Time the generation
    t1 = time.time()
    result = generate_answer(retrieval)
    generation_latency = (time.time() - t1) * 1000

    # Save to query history
    history = QueryHistory(
        query_text=request.query,
        answer_text=result["answer"],
        confidence_score=result["confidence"],
        retrieval_latency_ms=retrieval_latency,
        generation_latency_ms=generation_latency,
        chunks_retrieved=[s["chunk_id"] for s in result["all_sources"]],
        document_filters=request.document_ids or [],
    )
    db.add(history)
    db.commit()

    return QueryResponse(
        query=request.query,
        answer=result["answer"],
        confidence=result["confidence"],
        reasoning=result["reasoning"],
        citations=result["citations"],
        sources=result["sources"],
        retrieval_latency_ms=round(retrieval_latency, 2),
        generation_latency_ms=round(generation_latency, 2),
    )

@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(db: Session = Depends(get_db)):
    """List all ingested documents."""
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type.value,
            status=doc.status,
            file_size=doc.file_size,
            page_count=doc.page_count,
            metadata=doc.metadata_,
            created_at=doc.created_at,
        )
        for doc in documents
    ]

@router.delete("/documents/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document and all its chunks."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    db.delete(document)
    db.commit()
    return {"message": f"Document {document_id} deleted successfully"}
