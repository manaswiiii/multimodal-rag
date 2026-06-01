import os
from pathlib import Path
from pypdf import PdfReader
from sqlalchemy.orm import Session
from app.models import Document, Chunk, FileType
from app.ingestion.chunker import chunk_text
from app.config import settings

def ingest_pdf(file_path: str, db: Session) -> Document:
    """
    Read a PDF, extract text page by page, chunk it, and save to DB.
    Returns the Document object.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"📄 Processing: {path.name}")

    # 1. Create the Document record
    document = Document(
        filename=path.name,
        file_type=FileType.PDF,
        file_path=str(path.absolute()),
        file_size=path.stat().st_size,
        status="processing"
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    print(f"  ✅ Document record created (id={document.id})")

    # 2. Extract text from each page
    reader = PdfReader(file_path)
    page_count = len(reader.pages)
    document.page_count = page_count
    print(f"  📖 Found {page_count} pages")

    all_chunks = []
    chunk_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text or not text.strip():
            print(f"  ⚠️  Page {page_num}: no text found (might be scanned image)")
            continue

        # 3. Chunk the page text
        page_chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        print(f"  📝 Page {page_num}: {len(page_chunks)} chunks")

        for chunk_text_content in page_chunks:
            chunk = Chunk(
                document_id=document.id,
                content=chunk_text_content,
                chunk_index=chunk_index,
                page_number=page_num,
                chunk_type="text",
                token_count=len(chunk_text_content.split())
            )
            all_chunks.append(chunk)
            chunk_index += 1

    # 4. Save all chunks and mark document complete
    db.bulk_save_objects(all_chunks)
    document.status = "complete"
    document.metadata_ = {"page_count": page_count, "total_chunks": chunk_index}
    db.commit()

    print(f"  ✅ Done! {chunk_index} chunks saved to DB")
    return document
