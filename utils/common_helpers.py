from decimal import Decimal
import math
from typing import Any, Optional
import json
from typing import Any, Dict
import httpx

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
    """
    Canonical key used across the app to avoid collisions and mismatch.
    Matches your holdings layer _key(symbol, type): "AAPL:equity" or "AAPL".
    """
    s = (symbol or "").upper().strip()
    t = (typ or "").lower().strip()
    return f"{s}:{t}" if t else s