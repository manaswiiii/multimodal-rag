from pathlib import Path
from sqlalchemy.orm import Session
from app.models import FileType
from app.ingestion.pdf_ingester import ingest_pdf
from app.ingestion.csv_ingester import ingest_csv
from app.ingestion.code_ingester import ingest_code
from app.ingestion.embedder import EmbeddingManager
from app.models import Chunk

EXTENSION_MAP = {
    ".pdf": FileType.PDF,
    ".csv": FileType.CSV,
    ".py": FileType.CODE,
    ".js": FileType.CODE,
    ".ts": FileType.CODE,
    ".java": FileType.CODE,
    ".cpp": FileType.CODE,
    ".go": FileType.CODE,
    ".rs": FileType.CODE,
    ".md": FileType.CODE,
}

def ingest_file(file_path: str, db: Session, embedder: EmbeddingManager):
    """
    Route any file to the correct ingester, then embed its chunks.
    This is the single entry point for all file ingestion.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in EXTENSION_MAP:
        raise ValueError(f"Unsupported file type: {ext}")

    file_type = EXTENSION_MAP[ext]

    # 1. Ingest into PostgreSQL
    if file_type == FileType.PDF:
        document = ingest_pdf(file_path, db)
    elif file_type == FileType.CSV:
        document = ingest_csv(file_path, db)
    elif file_type == FileType.CODE:
        document = ingest_code(file_path, db)

    # 2. Embed the new chunks into FAISS
    chunks = db.query(Chunk).filter(
        Chunk.document_id == document.id,
        Chunk.embedding_id == None
    ).all()

    if chunks:
        faiss_ids = embedder.embed_chunks(chunks)
        for chunk, faiss_id in zip(chunks, faiss_ids):
            chunk.embedding_id = faiss_id
        db.commit()

    print(f"\n🎉 {path.name} fully ingested and embedded!")
    return document
