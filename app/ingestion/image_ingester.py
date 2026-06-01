from pathlib import Path
from PIL import Image
import pytesseract
from sqlalchemy.orm import Session
from app.models import Document, Chunk, FileType
from app.ingestion.chunker import chunk_text
from app.config import settings

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}

def ingest_image(file_path: str, db: Session) -> Document:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {ext}")

    print(f"🖼️  Processing: {path.name}")

    document = Document(
        filename=path.name,
        file_type=FileType.IMAGE,
        file_path=str(path.absolute()),
        file_size=path.stat().st_size,
        status="processing"
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    print(f"  ✅ Document record created (id={document.id})")

    # Open image and run OCR
    image = Image.open(file_path)
    width, height = image.size
    print(f"  📐 Image size: {width}x{height}")

    # Extract text with Tesseract
    ocr_text = pytesseract.image_to_string(image)
    ocr_text = ocr_text.strip()

    if not ocr_text:
        print("  ⚠️  No text found in image")
        document.status = "complete"
        document.metadata_ = {
            "width": width,
            "height": height,
            "ocr_text_length": 0,
            "total_chunks": 0
        }
        db.commit()
        return document

    print(f"  📝 Extracted {len(ocr_text)} characters via OCR")

    # Chunk the extracted text
    text_chunks = chunk_text(ocr_text, settings.chunk_size, settings.chunk_overlap)
    print(f"  ✂️  Split into {len(text_chunks)} chunks")

    chunks = []
    for i, chunk_content in enumerate(text_chunks):
        chunks.append(Chunk(
            document_id=document.id,
            content=chunk_content,
            chunk_index=i,
            chunk_type="ocr_text",
            token_count=len(chunk_content.split()),
            metadata_={"width": width, "height": height}
        ))

    db.bulk_save_objects(chunks)
    document.status = "complete"
    document.metadata_ = {
        "width": width,
        "height": height,
        "ocr_text_length": len(ocr_text),
        "total_chunks": len(text_chunks)
    }
    db.commit()

    print(f"  ✅ Done! {len(text_chunks)} chunks saved")
    return document
