from sqlalchemy import Column, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from database import Base
import uuid

class CompanyInsight(Base):
    __tablename__ = "company_insights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String, index=True, nullable=False)
    content = Column(Text, nullable=False)
    metadata_info = Column(JSONB)
    
    # Using 1536 for OpenAI text-embedding-3-small
    embedding = Column(Vector(1536)) 
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())