from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.models import Chunk, Document
from app.ingestion.embedder import EmbeddingManager

class Retriever:
    def __init__(self, db: Session, embedder: EmbeddingManager):
        self.db = db
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        file_type_filter: str = None,
        document_ids: List[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for the most relevant chunks for a query.
        Optionally filter by file type or specific document IDs.
        """
        # 1. Get more candidates than needed so we can filter
        candidates = self.embedder.search(query, top_k=top_k * 3)

        if not candidates:
            return []

        results = []
        for chunk_id, distance in candidates:
            chunk = self.db.query(Chunk).filter(Chunk.id == chunk_id).first()
            if not chunk:
                continue

            document = self.db.query(Document).filter(Document.id == chunk.document_id).first()
            if not document:
                continue

            # Apply filters
            if file_type_filter and document.file_type.value != file_type_filter:
                continue
            if document_ids and document.id not in document_ids:
                continue

            # Convert distance to a 0-1 confidence score
            # Lower distance = more similar = higher confidence
            confidence = max(0.0, 1.0 - distance)

            results.append({
                "chunk_id": chunk.id,
                "content": chunk.content,
                "chunk_type": chunk.chunk_type,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "document_id": document.id,
                "filename": document.filename,
                "file_type": document.file_type.value,
                "confidence": round(confidence, 4),
                "distance": round(distance, 4),
            })

            if len(results) >= top_k:
                break

        # Sort by confidence descending
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def retrieve_with_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve chunks and format them as a context block for the LLM.
        """
        chunks = self.retrieve(query, top_k=top_k)

        if not chunks:
            return {"query": query, "context": "", "sources": []}

        # Build context string with source labels
        context_parts = []
        sources = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['filename']} (page {chunk['page_number'] or 'N/A'})]\n{chunk['content']}"
            )
            sources.append({
                "index": i,
                "filename": chunk["filename"],
                "file_type": chunk["file_type"],
                "page_number": chunk["page_number"],
                "confidence": chunk["confidence"],
                "chunk_id": chunk["chunk_id"],
            })

        return {
            "query": query,
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "chunks": chunks,
        }
