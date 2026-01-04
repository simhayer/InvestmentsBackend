from __future__ import annotations
from datetime import datetime, timezone, date
from typing import Any, Optional
Number = Optional[float]
import pandas as pd

def pct(cur: Number, prev: Number) -> Number:
    try:
        if cur is None or prev in (None, 0):
            return None
        return (cur / prev - 1.0) * 100.0
    except Exception:
        return None


def dist_pct(cur: Number, ref: Number) -> Number:
    try:
        if cur is None or ref in (None, 0):
            return None
        return (cur - ref) / ref * 100.0
    except Exception:
        return None


def iso_utc_from_ts(ts: Any) -> Optional[str]:
    try:
        if ts is None:
            return None
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None
    

import re
_dt_suffix_pat = re.compile(r"(:S|[ T]S)$", re.IGNORECASE)

def to_epoch_utc(x: Any) -> int | None:
    """Coerce pandas Timestamp / datetime / date / epoch / ISO string to epoch seconds (UTC)."""
    if x is None:
        return None
    try:
        if isinstance(x, pd.Timestamp):
            x = x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")
            return int(x.timestamp())
    except Exception:
        pass
    if isinstance(x, datetime):
        x = x.astimezone(timezone.utc) if x.tzinfo else x.replace(tzinfo=timezone.utc)
        return int(x.timestamp())
    if isinstance(x, date):
        return int(datetime(x.year, x.month, x.day, tzinfo=timezone.utc).timestamp())
    if isinstance(x, (int, float)):
        try:
            return int(float(x))
        except Exception:
            return None
    if isinstance(x, str):
        # try pandas parser; utc=True gives tz-aware UTC
        try:
            ts = pd.to_datetime(x, utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            return int(ts.timestamp())
        except Exception:
            return None
    return None

def date_iso(d: Any) -> Optional[str]:
    t = to_epoch_utc(d)
    if t is None:
        return None
    return datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat()

def parse_weird_cal_dt(val: Any) -> Optional[str]:
    """
    '2025-10-30 16:00:S' -> '2025-10-30' (ISO date)
    also accepts list[...] / epoch / dict{'raw':...}
    """
    # unwrap list
    if isinstance(val, list) and val:
        val = val[0]
    # epoch
    if isinstance(val, (int, float)):
        return date_iso(val)
    # dict{'raw':...}
    if isinstance(val, dict) and "raw" in val:
        return date_iso(val["raw"])
    # string with trailing ':S'
    if isinstance(val, str):
        s = _dt_suffix_pat.sub("", val.strip())  # drop weird suffix
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date().isoformat()
    return None

def quarter_label(iso_date: str) -> str:
    """
    '2025-06-30' -> 'Q2 2025' (calendar quarter).
    (If you want fiscal quarters, you can add an offset param later.)
    """
    try:
        dt = datetime.fromisoformat(iso_date)
    except Exception:
        return iso_date
    q = (dt.month - 1) // 3 + 1
    return f"Q{q} {dt.year}"