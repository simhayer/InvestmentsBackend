import hashlib
import re
from typing import Dict, Any, List, Optional
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
        self._emb_cache: Dict[str, List[float]] = {}

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

    def has_symbol(self, db: Session, symbol: str) -> bool:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return False
        stmt = text("SELECT 1 FROM sec_filing_chunks WHERE symbol = :s LIMIT 1")
        return db.execute(stmt, {"s": symbol}).first() is not None
    
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

    def get_embedding_cached(self, text: str):
        key = (text or "").strip()
        if not key:
            return None
        if key in self._emb_cache:
            return self._emb_cache[key]
        vec = self.get_embedding(key)
        if vec:
            self._emb_cache[key] = vec
        return vec

    def process_and_save_filing(self, db: Session, symbol: str, html_content: str, metadata: Dict[str, Any]) -> int:
        symbol = (symbol or "").strip().upper()
        if not symbol or not html_content:
            return 0

        form = (metadata.get("form") or metadata.get("form_type") or "").upper().strip()
        filing_id = metadata.get("filing_id") or metadata.get("accession_number") or metadata.get("accessNumber")

        # 1) Pick correct parser
        try:
            parser = sp.Edgar10QParser()

            elements = parser.parse(html_content)
            tree = sp.TreeBuilder().build(elements)
        except Exception as e:
            # if parser fails, don't silently do nothing
            raise RuntimeError(f"SEC parse failed ({form}): {e}") from e

        pending_chunks: List[Dict[str, Any]] = []

        item_re = re.compile(r"\bITEM\s*([0-9]{1,2}[A-Z]?)\b", re.IGNORECASE)

        def walk_tree(node, current_section: Optional[str] = None):
            # Some semantic elements may not expose 'text' the way you expect
            element_text = (getattr(node.semantic_element, "text", None) or "").strip()
            el_type = type(node.semantic_element).__name__

            node_section = current_section

            # Detect short header-ish lines containing ITEM
            if element_text and len(element_text) <= 120:
                m = item_re.search(element_text)
                if m:
                    node_section = f"Item {m.group(1).upper()}"

            # Store non-trivial text blocks
            if el_type != "TableElement":
                txt = element_text
                if txt and len(txt) >= 200:
                    for chunk in chunk_text(txt, max_chars=6000):
                        h = hash_chunk(chunk)
                        pending_chunks.append({
                            "content": chunk,
                            "hash": h,
                            "el_type": el_type,
                            "section_name": node_section,
                        })

            for child in getattr(node, "children", []) or []:
                walk_tree(child, node_section)

        # Tree may be iterable or expose roots differently depending on sec_parser version
        roots = list(tree) if tree is not None else []
        for root_node in roots:
            walk_tree(root_node, None)

        if not pending_chunks:
            # Make it obvious when nothing is extracted
            raise RuntimeError(f"No text chunks produced for {symbol} {form} {filing_id}. Parser/tree produced no usable text.")

        # 2) Embed + upsert
        inserted_total = 0
        batch_size = 50

        for i in range(0, len(pending_chunks), batch_size):
            batch = pending_chunks[i:i + batch_size]
            vectors = self.get_embeddings_batch([item["content"] for item in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding batch size mismatch")

            rows = []
            for idx, item in enumerate(batch):
                rows.append({
                    "symbol": symbol,
                    "section_name": item.get("section_name"),
                    "filing_id": filing_id,
                    "form_type": form or None,
                    "filed_date": metadata.get("filedDate") or metadata.get("filed_date"),
                    "element_type": item.get("el_type"),
                    "chunk_hash": item.get("hash"),
                    "content": item.get("content"),
                    "metadata_info": metadata,
                    "embedding": vectors[idx],
                })

            insert_chunks_ignore_duplicates(db, rows)
            inserted_total += len(rows)

        return inserted_total


    def _to_pgvector_literal(self, vec: List[float]) -> str:
        # pgvector accepts: '[1,2,3]'::vector
        return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"

    def get_context_for_analysis(
        self,
        db: Session,
        symbol: str,
        query: str,
        section_name: Optional[str] = None,
        limit: int = 10,
        *,
        query_vector: Optional[list[float]] = None,
    ) -> List[Dict[str, Any]]:
        symbol = (symbol or "").strip().upper()
        if not symbol or not query:
            return []

        query_vector = query_vector or self.get_embedding_cached(query)
        if not query_vector:
            return []

        filter_clause = "AND section_name = :sect" if section_name else ""

        stmt = text(f"""
            SELECT
                content,
                metadata_info,
                form_type,
                filed_date,
                filing_id,
                element_type,
                chunk_hash,
                similarity
            FROM (
                SELECT
                    content,
                    metadata_info,
                    form_type,
                    filed_date,
                    filing_id,
                    element_type,
                    chunk_hash,
                    1 - (embedding <=> (:v)::vector) AS similarity
                FROM sec_filing_chunks
                WHERE symbol = :s {filter_clause}
                ORDER BY similarity DESC
                LIMIT :l
            ) sub
            ORDER BY similarity DESC
        """)

        params: Dict[str, Any] = {
            "v": self._to_pgvector_literal(query_vector),  # <-- key change
            "s": symbol,
            "l": int(limit),
        }
        if section_name:
            params["sect"] = section_name

        rows = db.execute(stmt, params).mappings().all()

        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = dict(r.get("metadata_info") or {})
            meta.update({
                "form_type": r.get("form_type"),
                "filed_date": r.get("filed_date"),
                "filing_id": r.get("filing_id"),
                "element_type": r.get("element_type"),
                "chunk_hash": r.get("chunk_hash"),
                "section_name": section_name,
            })
            out.append({
                "content": r.get("content") or "",
                "metadata": meta,
                "score": float(r.get("similarity") or 0.0),
            })

        return out
