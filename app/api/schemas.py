from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    file_type_filter: Optional[str] = None
    document_ids: Optional[List[int]] = None

class SourceResponse(BaseModel):
    index: int
    filename: str
    file_type: str
    page_number: Optional[int]
    confidence: float
    chunk_id: int

class QueryResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    reasoning: str
    citations: List[int]
    sources: List[SourceResponse]
    retrieval_latency_ms: float
    generation_latency_ms: float

class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    status: str
    file_size: Optional[int]
    page_count: Optional[int]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True
