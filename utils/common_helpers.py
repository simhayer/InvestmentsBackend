from decimal import Decimal
import math
from typing import Any, Optional
import json
from typing import Any, Dict
import httpx
import time
from typing import Callable, Tuple, Type
import logging
from services.ai.analyze_symbol.types import AgentState

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

def parse_json_strict(maybe: Any) -> Dict[str, Any]:
    import re
    if isinstance(maybe, list):
        maybe = "".join(str(p) for p in maybe if p is not None)
    if maybe is None:
        raise ValueError("Empty LLM response (None)")
    if not isinstance(maybe, str):
        maybe = str(maybe)
    s = maybe.strip()
    if not s:
        raise ValueError("Empty LLM response (blank)")
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}\s*$", s)
        if not m:
            raise
        return json.loads(m.group(0))
    

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

def unwrap_linkup(result: dict) -> dict:
    if isinstance(result, dict) and result.get("ok") and isinstance(result.get("data"), dict):
        return result["data"]
    raise RuntimeError(result.get("error") or "Linkup call failed")

def unwrap_layers_for_ui(layers: dict) -> dict:
    if not isinstance(layers, dict):
        return layers

    out = dict(layers)
    for k in ("news_sentiment", "performance", "scenarios_rebalance", "summary"):
        if k in out:
            out[k] = unwrap_linkup(out[k])
    return out

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

# ----------------------------
# 4) Timing helper
# ----------------------------
def timed(name: str, state: AgentState, logger: logging.Logger):
    symbol = state.get("symbol", "NA")
    task_id = state.get("task_id", "no_task")

    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            logger.info("[%s] %s: %s=%.2fs", task_id, symbol, name, dt)
            return False

    return _T()