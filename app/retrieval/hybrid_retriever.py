from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from rank_bm25 import BM25Okapi
from app.models import Chunk, Document
from app.ingestion.embedder import EmbeddingManager

class HybridRetriever:
    def __init__(self, db: Session, embedder: EmbeddingManager):
        self.db = db
        self.embedder = embedder
        self._bm25 = None
        self._chunk_ids = []

    def _build_bm25_index(self, chunks: List[Chunk]):
        """Build BM25 index from a list of chunks."""
        tokenized = [chunk.content.lower().split() for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._chunk_ids = [chunk.id for chunk in chunks]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        file_type_filter: Optional[str] = None,
        document_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: combine semantic (FAISS) + keyword (BM25) scores.
        semantic_weight + keyword_weight should equal 1.0
        """
        # 1. Semantic search — get more candidates for reranking
        semantic_results = self.embedder.search(query, top_k=top_k * 4)
        semantic_scores = {}
        for chunk_id, distance in semantic_results:
            semantic_scores[chunk_id] = max(0.0, 1.0 - distance)

        # 2. BM25 keyword search
        all_chunks = self.db.query(Chunk).all()
        if not all_chunks:
            return []

        self._build_bm25_index(all_chunks)
        bm25_raw = self._bm25.get_scores(query.lower().split())

        # Normalize BM25 scores to 0-1
        max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
        bm25_scores = {}
        for i, chunk in enumerate(all_chunks):
            bm25_scores[chunk.id] = bm25_raw[i] / max_bm25

        # 3. Combine scores
        all_chunk_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())
        combined = {}
        for chunk_id in all_chunk_ids:
            sem = semantic_scores.get(chunk_id, 0.0)
            kw = bm25_scores.get(chunk_id, 0.0)
            combined[chunk_id] = (semantic_weight * sem) + (keyword_weight * kw)

        # 4. Sort by combined score
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        # 5. Build results with metadata
        results = []
        for chunk_id, score in ranked:
            if len(results) >= top_k:
                break

            chunk = self.db.query(Chunk).filter(Chunk.id == chunk_id).first()
            if not chunk:
                continue

            document = self.db.query(Document).filter(Document.id == chunk.document_id).first()
            if not document:
                continue

            if file_type_filter and document.file_type.value != file_type_filter:
                continue
            if document_ids and document.id not in document_ids:
                continue

            results.append({
                "chunk_id": chunk.id,
                "content": chunk.content,
                "chunk_type": chunk.chunk_type,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "document_id": document.id,
                "filename": document.filename,
                "file_type": document.file_type.value,
                "confidence": round(score, 4),
                "semantic_score": round(semantic_scores.get(chunk_id, 0.0), 4),
                "keyword_score": round(bm25_scores.get(chunk_id, 0.0), 4),
            })

        return results

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> Dict[str, Any]:
        chunks = self.retrieve(query, top_k=top_k,
                               semantic_weight=semantic_weight,
                               keyword_weight=keyword_weight)

        if not chunks:
            return {"query": query, "context": "", "sources": []}

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
                "semantic_score": chunk["semantic_score"],
                "keyword_score": chunk["keyword_score"],
            })

        return {
            "query": query,
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "chunks": chunks,
        }
