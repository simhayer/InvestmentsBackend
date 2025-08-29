import math
from typing import Any, Optional
import json
from typing import Any, Dict, List

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