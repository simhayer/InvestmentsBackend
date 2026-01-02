# services/helpers/cache_utils.py
from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union, cast

from services.cache.cache_backend import cache_get, cache_set

# Same shape as cache_backend
JsonValue = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

T = TypeVar("T")


def should_cache_ok_json(val: Any) -> bool:
    """
    Default policy: cache only successful JSON payloads.
    Good for quote/financial endpoints where you return {"status":"ok"}.
    """
    return isinstance(val, dict) and val.get("status") == "ok"


def should_cache_any_json(val: Any) -> bool:
    """
    Cache any JSON value (dict/list/primitive).
    Useful for search endpoints returning a list, etc.
    """
    return isinstance(val, (dict, list, str, int, float, bool)) or val is None


def cacheable(
    *,
    ttl: int,
    key_fn: Callable[..., str],
    should_cache: Callable[[Any], bool] = should_cache_ok_json,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for read-only functions.
    Works for BOTH sync and async functions.

    - ttl: Redis/shared TTL (local TTL handled by backend)
    - key_fn: key_fn(*args, **kwargs) -> str
    - should_cache: decide what values get cached
    """
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> T:
                key = (key_fn(*args, **kwargs) or "").strip()
                if key:
                    hit = cache_get(key)
                    if hit is not None:
                        return cast(T, hit)

                val = await cast(Callable[..., Awaitable[Any]], fn)(*args, **kwargs)

                if key and should_cache(val):
                    try:
                        cache_set(key, cast(JsonValue, val), ttl_seconds=ttl)
                    except Exception:
                        pass

                return cast(T, val)

            return cast(Callable[..., T], awrapper)

        @wraps(fn)
        def swrapper(*args: Any, **kwargs: Any) -> T:
            key = (key_fn(*args, **kwargs) or "").strip()
            if key:
                hit = cache_get(key)
                if hit is not None:
                    return cast(T, hit)

            val = fn(*args, **kwargs)

            if key and should_cache(val):
                try:
                    cache_set(key, cast(JsonValue, val), ttl_seconds=ttl)
                except Exception:
                    pass

            return val

        return swrapper

    return deco
