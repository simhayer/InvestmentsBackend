# services/helpers/cache_utils.py
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, TypeVar, cast

from services.cache.cache_backend import cache_get, cache_set

Json = Dict[str, Any]
T = TypeVar("T")


def should_cache_ok_json(val: Any) -> bool:
    """
    Default policy: cache only successful JSON payloads.
    Avoid caching transient upstream errors.
    """
    return isinstance(val, dict) and val.get("status") == "ok"


def cacheable(
    *,
    ttl: int,
    key_fn: Callable[..., str],
    should_cache: Callable[[Any], bool] = should_cache_ok_json,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for read-only functions.

    Example:
      @cacheable(ttl=60, key_fn=lambda symbol: f"QUOTE:{symbol}")
      def get_quote(symbol: str) -> Json: ...

    - key_fn(*args, **kwargs) must return a stable cache key string
    - ttl is shared cache TTL (Redis); local warm-cache TTL handled by cache_backend
    - should_cache decides what values get cached (default: {"status":"ok"})
    """
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = (key_fn(*args, **kwargs) or "").strip()
            if key:
                hit = cache_get(key)
                if hit is not None:
                    return cast(T, hit)

            val = fn(*args, **kwargs)

            if key and should_cache(val):
                try:
                    cache_set(key, cast(Json, val), ttl_seconds=ttl)
                except Exception:
                    pass

            return val
        return wrapper
    return deco
