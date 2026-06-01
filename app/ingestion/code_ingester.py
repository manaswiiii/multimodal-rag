from pathlib import Path
from sqlalchemy.orm import Session
from app.models import Document, Chunk, FileType
from app.ingestion.chunker import chunk_text
from app.config import settings

SUPPORTED_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".cpp": "cpp", ".c": "c",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".md": "markdown"
}

def ingest_code(file_path: str, db: Session) -> Document:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    language = SUPPORTED_EXTENSIONS.get(ext, "unknown")

    print(f"💻 Processing: {path.name} ({language})")

    document = Document(
        filename=path.name,
        file_type=FileType.CODE,
        file_path=str(path.absolute()),
        file_size=path.stat().st_size,
        status="processing"
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    print(f"  ✅ Document record created (id={document.id})")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    lines = code.splitlines()
    print(f"  📝 Found {len(lines)} lines of {language} code")

    # Prefix each chunk with language context so the LLM knows what it is
    prefixed_code = f"Language: {language}\nFile: {path.name}\n\n{code}"
    text_chunks = chunk_text(prefixed_code, settings.chunk_size, settings.chunk_overlap)

    chunks = []
    for i, chunk_content in enumerate(text_chunks):
        chunks.append(Chunk(
            document_id=document.id,
            content=chunk_content,
            chunk_index=i,
            chunk_type="code_block",
            token_count=len(chunk_content.split()),
            metadata_={"language": language}
        ))

    db.bulk_save_objects(chunks)
    document.status = "complete"
    document.metadata_ = {"language": language, "line_count": len(lines), "total_chunks": len(chunks)}
    db.commit()

    print(f"  ✅ Done! {len(chunks)} chunks saved")
    return document
