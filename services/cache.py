# services/cache.py
from __future__ import annotations
import os, json, time
from typing import Any, Dict, Optional
import asyncpg
import asyncio

Json = Dict[str, Any]

# ---- Memory cache (ultra fast)
_mem: dict[str, tuple[float, Json]] = {}
MEM_TTL = 30  # seconds for memory cache

def _mem_get(key: str) -> Optional[Json]:
    hit = _mem.get(key)
    if not hit: return None
    ts, payload = hit
    if time.time() - ts <= MEM_TTL:
        return payload
    _mem.pop(key, None)
    return None

def _mem_set(key: str, payload: Json) -> None:
    _mem[key] = (time.time(), payload)

# ---- Optional Redis (shared cache)
# Use if REDIS_URL is set
try:
    import redis.asyncio as redis
    REDIS_URL = os.getenv("REDIS_URL")
    _redis = redis.from_url(REDIS_URL) if REDIS_URL else None
except Exception:
    REDIS_URL = None
    _redis = None

REDIS_TTL = 60  # seconds

async def _redis_get(key: str) -> Optional[Json]:
    if not _redis: return None
    s = await _redis.get(key)
    return json.loads(s) if s else None

async def _redis_set(key: str, payload: Json, ttl: int = REDIS_TTL) -> None:
    if not _redis: return
    await _redis.set(key, json.dumps(payload), ex=ttl)

# ---- DB helpers (asyncpg; swap with your DB util)
DB_URL = os.getenv("DATABASE_URL")  # Supabase PG URL works here

async def _get_conn():
    return await asyncpg.connect(DB_URL)

async def db_read_latest() -> Optional[Json]:
    con = await _get_conn()
    try:
        row = await con.fetchrow("""
            select symbols, items, ai_summary, fetched_at
            from market_overview_latest
            order by id desc
            limit 1
        """)
        if not row:
            return None
        return dict(row)
    finally:
        await con.close()

async def db_upsert_latest(payload: Json) -> None:
    async with await _get_conn() as con:
        # one-row invariant via partial unique index; simplest is: delete + insert
        async with con.transaction():
            await con.execute("delete from market_overview_latest")
            await con.execute("""
                insert into market_overview_latest (symbols, items, ai_summary, fetched_at)
                values ($1, $2, $3, now())
            """, payload["symbols"], json.dumps(payload["items"]), payload.get("ai_summary"))

async def db_append_history(payload: Json) -> None:
    async with await _get_conn() as con:
        await con.execute("""
            insert into market_overview_history (symbols, items, ai_summary, ts)
            values ($1, $2, $3, now())
        """, payload["symbols"], json.dumps(payload["items"]), payload.get("ai_summary"))
