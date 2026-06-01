from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

class FileType(str, enum.Enum):
    PDF = "pdf"
    IMAGE = "image"
    CODE = "code"
    CSV = "csv"

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(Enum(FileType), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    page_count = Column(Integer)
    status = Column(String(50), default="pending")
    metadata_ = Column("metadata", JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer)
    chunk_type = Column(String(50))
    embedding_id = Column(String(100))
    token_count = Column(Integer)
    metadata_ = Column("metadata", JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")

class QueryHistory(Base):
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text)
    confidence_score = Column(Float)
    retrieval_latency_ms = Column(Float)
    generation_latency_ms = Column(Float)
    chunks_retrieved = Column(JSON, default=[])
    document_filters = Column(JSON, default=[])
    rating = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
