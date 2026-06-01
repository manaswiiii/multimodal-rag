import pandas as pd
from pathlib import Path
from sqlalchemy.orm import Session
from app.models import Document, Chunk, FileType
from app.ingestion.chunker import chunk_text
from app.config import settings

def ingest_csv(file_path: str, db: Session) -> Document:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"📊 Processing: {path.name}")

    document = Document(
        filename=path.name,
        file_type=FileType.CSV,
        file_path=str(path.absolute()),
        file_size=path.stat().st_size,
        status="processing"
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    print(f"  ✅ Document record created (id={document.id})")

    df = pd.read_csv(file_path)
    print(f"  📋 Found {len(df)} rows, {len(df.columns)} columns: {list(df.columns)}")

    chunks = []
    chunk_index = 0

    # Chunk 1: column summary
    summary = f"CSV file: {path.name}. Columns: {', '.join(df.columns)}. Total rows: {len(df)}."
    chunks.append(Chunk(
        document_id=document.id,
        content=summary,
        chunk_index=chunk_index,
        chunk_type="csv_summary",
        token_count=len(summary.split())
    ))
    chunk_index += 1

    # Chunk each row as a readable sentence
    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
        sub_chunks = chunk_text(row_text, settings.chunk_size, settings.chunk_overlap)
        for sub in sub_chunks:
            chunks.append(Chunk(
                document_id=document.id,
                content=sub,
                chunk_index=chunk_index,
                chunk_type="csv_row",
                token_count=len(sub.split())
            ))
            chunk_index += 1

    db.bulk_save_objects(chunks)
    document.status = "complete"
    document.metadata_ = {"row_count": len(df), "columns": list(df.columns), "total_chunks": chunk_index}
    db.commit()

    print(f"  ✅ Done! {chunk_index} chunks saved")
    return document
