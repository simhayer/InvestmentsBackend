# services/helpers/cache_backend.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast

try:
    import redis as redis_sync
except Exception:
    redis_sync = None

JsonInput = Union[str, bytes, bytearray]
JsonValue = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

DEFAULT_TTL_SEC = int(os.getenv("CACHE_DEFAULT_TTL_SEC", "60"))
LOCAL_CACHE_TTL_SEC = int(os.getenv("CACHE_LOCAL_TTL_SEC", "60"))

REDIS_PREFIX = os.getenv("REDIS_PREFIX", "wallstreetai:")
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")

_LOCAL: Dict[str, Tuple[float, JsonValue]] = {}
_redis_client = None


def get_redis_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    if not redis_sync or not UPSTASH_REDIS_URL:
        _redis_client = None
        return None

    try:
        _redis_client = redis_sync.from_url(
            UPSTASH_REDIS_URL,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
    except Exception:
        _redis_client = None

    return _redis_client


def _norm_key(key: str) -> str:
    return (key or "").strip().upper()


def _redis_key(key: str) -> str:
    return f"{REDIS_PREFIX}{_norm_key(key)}"


def _as_json_input(v: Any) -> Optional[JsonInput]:
    if isinstance(v, (str, bytes, bytearray)):
        return v
    return None


def _local_get(k: str) -> Optional[JsonValue]:
    hit = _LOCAL.get(k)
    if not hit:
        return None
    expires_at, payload = hit
    if time.time() <= expires_at:
        return payload
    _LOCAL.pop(k, None)
    return None


def _local_set(k: str, payload: JsonValue, ttl_seconds: int) -> None:
    _LOCAL[k] = (time.time() + ttl_seconds, payload)


def cache_get(key: str) -> Optional[JsonValue]:
    k = _norm_key(key)
    if not k:
        return None

    hit = _local_get(k)
    if hit is not None:
        return hit

    r = get_redis_client()
    if not r:
        return None

    try:
        raw_any = r.get(_redis_key(k))
        raw = _as_json_input(raw_any)
        if raw is None:
            return None

        payload: JsonValue = json.loads(raw)
        _local_set(k, payload, LOCAL_CACHE_TTL_SEC)
        return payload
    except Exception:
        return None


def cache_set(key: str, payload: JsonValue, ttl_seconds: int = DEFAULT_TTL_SEC) -> None:
    k = _norm_key(key)
    if not k:
        return

    ttl_seconds = int(ttl_seconds) if ttl_seconds and ttl_seconds > 0 else DEFAULT_TTL_SEC

    _local_set(k, payload, min(LOCAL_CACHE_TTL_SEC, ttl_seconds))

    r = get_redis_client()
    if not r:
        return

    try:
        r.setex(_redis_key(k), ttl_seconds, json.dumps(payload, separators=(",", ":")))
    except Exception:
        pass


def cache_get_many(keys: List[str]) -> Dict[str, Optional[JsonValue]]:
    out: Dict[str, Optional[JsonValue]] = {}
    misses: List[str] = []
    now = time.time()

    for key in keys:
        k = _norm_key(key)
        if not k:
            continue

        hit = _LOCAL.get(k)
        if not hit:
            out[k] = None
            misses.append(k)
            continue

        expires_at, payload = hit
        if now <= expires_at:
            out[k] = payload
        else:
            _LOCAL.pop(k, None)
            out[k] = None
            misses.append(k)

    if not misses:
        return out

    r = get_redis_client()
    if not r:
        return out

    try:
        rkeys = [_redis_key(k) for k in misses]
        raws_any = r.mget(rkeys)
        raws = cast(List[Any], raws_any)

        for k, raw_any in zip(misses, raws):
            raw = _as_json_input(raw_any)
            if raw is None:
                continue
            try:
                payload: JsonValue = json.loads(raw)
            except Exception:
                continue

            out[k] = payload
            _local_set(k, payload, LOCAL_CACHE_TTL_SEC)

    except Exception:
        return out

    return out


def cache_set_many(kv: Dict[str, JsonValue], ttl_seconds: int = DEFAULT_TTL_SEC) -> None:
    if not kv:
        return

    ttl_seconds = int(ttl_seconds) if ttl_seconds and ttl_seconds > 0 else DEFAULT_TTL_SEC

    local_ttl = min(LOCAL_CACHE_TTL_SEC, ttl_seconds)
    expires_at = time.time() + local_ttl

    for key, payload in kv.items():
        k = _norm_key(key)
        if not k:
            continue
        _LOCAL[k] = (expires_at, payload)

    r = get_redis_client()
    if not r:
        return

    try:
        pipe = r.pipeline(transaction=False)
        for key, payload in kv.items():
            k = _norm_key(key)
            if not k:
                continue
            pipe.setex(_redis_key(k), ttl_seconds, json.dumps(payload, separators=(",", ":")))
        pipe.execute()
    except Exception:
        pass
