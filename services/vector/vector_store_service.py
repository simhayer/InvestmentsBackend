import hashlib
import warnings
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from models.vectors.sec_filing_chunk import SecFilingChunk
import sec_parser as sp
from sqlalchemy import text
from services.openai.client import get_openai_client
from sqlalchemy.dialects.postgresql import insert

def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def hash_chunk(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def insert_chunks_ignore_duplicates(db: Session, rows: List[Dict[str, Any]]):
    if not rows:
        return

    stmt = insert(SecFilingChunk).values(rows)
    stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "chunk_hash"])
    db.execute(stmt)

class VectorStoreService:
    def __init__(self):
        self.ai = get_openai_client()

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Fetch multiple embeddings in one call to save time and money."""
        if not texts:
            return []
        cleaned_texts = [t.replace("\n", " ") for t in texts]
        response = self.ai.embeddings.create(
            model="text-embedding-3-small",
            input=cleaned_texts
        )
        return [data.embedding for data in response.data]
    
    def get_embedding(self, text: str) -> List[float]:
        """Fetch a single embedding."""
        if not text:
            return []
        cleaned_text = text.replace("\n", " ")
        response = self.ai.embeddings.create(
            model="text-embedding-3-small",
            input=[cleaned_text]
        )
        return response.data[0].embedding

    def process_and_save_filing(self, db: Session, symbol: str, html_content: str, metadata: Dict[str, Any]):
        parser = sp.Edgar10QParser()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            elements = parser.parse(html_content)

        filing_id = metadata.get("filing_id") or metadata.get("accession_number")
        pending_chunks = []
        seen = set()

        # 1) Collect chunks (skip tables)
        for element in elements:
            el_type = type(element).__name__
            if el_type == "TableElement":
                continue

            text = getattr(element, "text", None)
            if not text or len(text.strip()) <= 150:
                continue

            for chunk in chunk_text(text.strip(), max_chars=6000):
                c_hash = hash_chunk(chunk)
                if c_hash in seen:
                    continue
                seen.add(c_hash)

                pending_chunks.append({
                    "content": chunk,
                    "hash": c_hash,
                    "el_type": el_type,
                })

        # 2) Embed + upsert in batches
        batch_size = 50
        for i in range(0, len(pending_chunks), batch_size):
            batch = pending_chunks[i:i + batch_size]
            vectors = self.get_embeddings_batch([item["content"] for item in batch])

            rows = []
            for idx, item in enumerate(batch):
                rows.append({
                    "symbol": symbol,
                    "filing_id": filing_id,
                    "form_type": metadata.get("form") or metadata.get("source_type"),
                    "filed_date": str(metadata.get("filedDate") or metadata.get("filed_date") or ""),
                    "element_type": item["el_type"],
                    "chunk_hash": item["hash"],
                    "content": item["content"],
                    "metadata_info": metadata,      # <-- your column name
                    "embedding": vectors[idx],
                })

            insert_chunks_ignore_duplicates(db, rows)

        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

    def get_context_for_analysis(self, db: Session, symbol: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query_vector = self.get_embedding(query)

        stmt = text("""
            SELECT
                content,
                metadata_info,
                form_type,
                filed_date,
                filing_id,
                element_type,
                chunk_hash,
                1 - (embedding <=> ((:v)::real[]::vector(1536))) AS similarity
            FROM sec_filing_chunks
            WHERE symbol = :s
            ORDER BY embedding <=> ((:v)::real[]::vector(1536))
            LIMIT :l
        """)

        rows = db.execute(stmt, {"v": query_vector, "s": symbol, "l": limit}).mappings().all()

        out = []
        for r in rows:
            meta = dict(r["metadata_info"] or {})
            meta.update({
                "form_type": r["form_type"],
                "filed_date": r["filed_date"],
                "filing_id": r["filing_id"],
                "element_type": r["element_type"],
                "chunk_hash": r["chunk_hash"],
            })
            out.append({
                "content": r["content"],
                "metadata": meta,
                "score": float(r["similarity"]),
            })
        return out