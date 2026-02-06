from decimal import Decimal
import math
import time
from typing import Any, Dict, List, Optional, Callable, Tuple, Type, Mapping
import httpx


import logging
def normalize_asset_type(typ: str | None) -> str | None:
    t = (typ or "").strip().lower()
    if not t:
        return None

    # unify all stock-like values
    if t in {"stock", "equity", "etf", "common stock", "adr"}:
        return "equity"

    if t in {"crypto", "cryptocurrency"}:
        return "cryptocurrency"

    return t

def median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

def to_float(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, Decimal):
        return float(x)
    try:
        return float(x)
    except Exception:
        return 0.0

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def pct_change(cur: Optional[float], base: Optional[float]) -> Optional[float]:
    if cur is None or base in (None, 0) or (isinstance(base, float) and abs(base) < 1e-6):
        return None
    return (cur / base - 1.0) * 100.0

def round(x: Optional[float], d: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), d)

def safe_div(n, d):
    try:
        return (n / d) if (n is not None and d not in (None, 0)) else None
    except ZeroDivisionError:
        return None
    
def num(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None
    
def safe_json(resp: httpx.Response) -> Optional[Dict[str, Any]]:
    try:
        data = resp.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None

def canonical_key(symbol: Optional[str], typ: Optional[str]) -> str:
    s = (symbol or "").upper().strip()
    t = normalize_asset_type(typ)
    return f"{s}:{t}" if t else s

def retry(
    fn: Callable[[], Any],
    *,
    attempts: int = 3,
    delay: float = 0.4,
    backoff: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Any:
    """
    Retry a function up to `attempts` times with exponential backoff.
    Raises RuntimeError (chained) if all attempts fail.
    """
    attempts = max(1, attempts)
    err: BaseException | None = None

    for i in range(attempts):
        try:
            return fn()
        except exceptions as e:   # <-- exceptions is a tuple of classes
            err = e
            if i < attempts - 1:
                time.sleep(delay * (backoff ** i))
            else:
                break

    raise RuntimeError(f"retry failed after {attempts} attempts") from err

def timed(
    name: str,
    logger: logging.Logger,
    *,
    state: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, Any]] = None,
):
    task_id = None
    symbol = None

    if state:
        task_id = state.get("task_id")
        symbol = state.get("symbol")

    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0

            parts = []
            if task_id:
                parts.append(f"[{task_id}]")
            if symbol:
                parts.append(str(symbol))
            if tags:
                parts.extend(f"{k}={v}" for k, v in tags.items())

            prefix = " ".join(parts)
            if prefix:
                logger.info("%s %s=%.2fs", prefix, name, dt)
            else:
                logger.info("%s=%.2fs", name, dt)

            return False

    return _T()
