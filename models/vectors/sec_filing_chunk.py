import uuid
from sqlalchemy import Column, String, Text, DateTime, func, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from database import Base

class SecFilingChunk(Base):
    __tablename__ = "sec_filing_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String, index=True, nullable=False)

    # --- New Field for Section Routing ---
    # Stores values like "Item 1", "Item 1A", "Item 7", etc.
    section_name = Column(String, index=True, nullable=True) 

    filing_id = Column(String, index=True, nullable=True)
    form_type = Column(String, nullable=True)
    filed_date = Column(String, nullable=True)

    element_type = Column(String, nullable=True)
    chunk_hash = Column(String, nullable=False)

    content = Column(Text, nullable=False)
    metadata_info = Column(JSONB)

    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("symbol", "chunk_hash", name="sec_filing_chunks_symbol_chunk_hash_uniq"),
        # Composite index for lightning-fast filtered vector searches
        Index("idx_symbol_section_vector", symbol, section_name),
    )