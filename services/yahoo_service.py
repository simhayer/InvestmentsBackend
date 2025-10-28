# services/yahoo_service.py
from __future__ import annotations

import math
import time
from datetime import datetime, timezone, timedelta, date

from typing import Any, Dict, Callable, Iterable, List, Optional, Tuple, Type

from yahooquery import Ticker
import pandas as pd

Number = Optional[float]
Json = Dict[str, Any]

# ---------------------------
# Retry helper (fixed)
# ---------------------------
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


# ---------------------------
# Tiny TTL cache (optional but helpful)
# ---------------------------
_CACHE: Dict[str, Tuple[float, Json]] = {}
_CACHE_TTL_SEC = 60  # adjust 30–120s as you like


def _cache_get(symbol: str) -> Optional[Json]:
    key = symbol.upper()
    hit = _CACHE.get(key)
    if not hit:
        return None
    ts, payload = hit
    if time.time() - ts <= _CACHE_TTL_SEC:
        return payload
    _CACHE.pop(key, None)
    return None


def _cache_set(symbol: str, payload: Json) -> None:
    _CACHE[(symbol.upper())] = (time.time(), payload)


# ---------------------------
# Parsing helpers
# ---------------------------
def _ensure_symbol_dict(obj: Any, sym: str) -> Dict[str, Any]:
    """
    yahooquery can return strings, lists, or dicts not keyed by symbol.
    Normalize to a dict (or {}) for the symbol.
    """
    if isinstance(obj, dict):
        # Prefer nested by symbol if present, else accept obj as-is
        if sym in obj and isinstance(obj[sym], dict):
            return obj[sym]
        return obj
    return {}


def _fnum(x: Any) -> Number:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def _pct(cur: Number, prev: Number) -> Number:
    try:
        if cur is None or prev in (None, 0):
            return None
        return (cur / prev - 1.0) * 100.0
    except Exception:
        return None


def _dist_pct(cur: Number, ref: Number) -> Number:
    try:
        if cur is None or ref in (None, 0):
            return None
        return (cur - ref) / ref * 100.0
    except Exception:
        return None


def _iso_utc_from_ts(ts: Any) -> Optional[str]:
    try:
        if ts is None:
            return None
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


# ---------------------------
# Main API
# ---------------------------
def get_full_stock_data(symbol: str) -> Json:
    """
    Fetch quotes + fundamentals (and optionally top news) for `symbol`.
    Returns a stable JSON shape with computed metrics and data quality info.
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return {
            "status": "error",
            "error_code": "EMPTY_SYMBOL",
            "message": "Symbol is required",
        }

    # Serve from short TTL cache if available
    cached = _cache_get(sym)
    if cached:
        return cached

    try:
        # NOTE: yahooquery handles session/crumb internally; retries help
        tq = Ticker(sym, asynchronous=False, formatted=False, validate=False)

        # Fetch raw payloads with retries (handles CSRF/crumb intermittency)
        summary_detail_raw = retry(lambda: tq.summary_detail)
        financial_data_raw = retry(lambda: tq.financial_data)
        key_stats_raw = retry(lambda: tq.key_stats)
        price_raw = retry(lambda: tq.price)

        # Normalize to dicts; never call .get on a non-dict again
        summary_detail = _ensure_symbol_dict(summary_detail_raw, sym)
        financial_data = _ensure_symbol_dict(financial_data_raw, sym)
        key_stats = _ensure_symbol_dict(key_stats_raw, sym)
        price_data = _ensure_symbol_dict(price_raw, sym)

        # --- Core fields
        short_name = price_data.get("shortName") or price_data.get("longName")
        currency = price_data.get("currency")
        exchange = price_data.get("exchangeName") or price_data.get("fullExchangeName")

        current = _fnum(
            summary_detail.get("regularMarketPrice")
            or price_data.get("regularMarketPrice")
        )
        previous = _fnum(
            price_data.get("regularMarketPreviousClose")
            or summary_detail.get("previousClose")
        )

        # 52-week range
        high_52 = _fnum(summary_detail.get("fiftyTwoWeekHigh"))
        low_52 = _fnum(summary_detail.get("fiftyTwoWeekLow"))

        # Fundamentals
        pe_ratio = _fnum(summary_detail.get("trailingPE"))
        forward_pe = _fnum(summary_detail.get("forwardPE") or financial_data.get("forwardPE"))
        price_to_book = _fnum(key_stats.get("priceToBook") or summary_detail.get("priceToBook"))
        beta = _fnum(summary_detail.get("beta") or key_stats.get("beta"))
        dividend_yield = _fnum(summary_detail.get("dividendYield"))
        market_cap = _fnum(summary_detail.get("marketCap") or price_data.get("marketCap"))

        return_on_equity = _fnum(financial_data.get("returnOnEquity"))
        profit_margins = _fnum(financial_data.get("profitMargins"))
        earnings_growth = _fnum(financial_data.get("earningsGrowth"))
        revenue_growth = _fnum(financial_data.get("revenueGrowth"))
        recommendation = _fnum(financial_data.get("recommendationMean"))
        recommendation_key = financial_data.get("recommendationKey")
        target_price = _fnum(financial_data.get("targetMeanPrice"))

        # Quote time
        quote_ts = (
            price_data.get("regularMarketTime")
            or price_data.get("postMarketTime")
            or price_data.get("preMarketTime")
        )
        quote_time_utc = _iso_utc_from_ts(quote_ts)

        # Computed deltas
        day_change = (current - previous) if (current is not None and previous is not None) else None
        day_change_pct = _pct(current, previous)
        distance_from_52w_high_pct = _dist_pct(current, high_52)
        distance_from_52w_low_pct = _dist_pct(current, low_52)

        # Data quality
        missing_fields = [
            k
            for k, v in {
                "current_price": current,
                "currency": currency,
                "previous_close": previous,
            }.items()
            if v is None
        ]
        is_stale = False
        if quote_time_utc:
            try:
                qt = datetime.fromisoformat(quote_time_utc.replace("Z", "+00:00"))
                is_stale = (datetime.now(timezone.utc) - qt) > timedelta(days=2)
            except Exception:
                pass

        payload: Json = {
            "status": "ok",
            "symbol": sym,
            "name": short_name,
            "currency": currency,
            "exchange": exchange,
            "quote_time_utc": quote_time_utc,
            # Prices
            "current_price": current,
            "previous_close": previous,
            "day_change": day_change,
            "day_change_pct": day_change_pct,
            # Range
            "52_week_high": high_52,
            "52_week_low": low_52,
            "distance_from_52w_high_pct": distance_from_52w_high_pct,
            "distance_from_52w_low_pct": distance_from_52w_low_pct,
            # Fundamentals
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "forward_pe": forward_pe,
            "price_to_book": price_to_book,
            "beta": beta,
            "dividend_yield": dividend_yield,
            "return_on_equity": return_on_equity,
            "profit_margins": profit_margins,
            "earnings_growth": earnings_growth,
            "revenue_growth": revenue_growth,
            "recommendation": recommendation,
            "recommendation_key": recommendation_key,
            "target_price": target_price,
            # Quality & provenance
            "data_quality": {
                "source": "Yahoo Finance via yahooquery",
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "is_stale": is_stale,
                "missing_fields": missing_fields,
            },
        }

        _cache_set(sym, payload)
        return payload

    except Exception as e:
        # Hardened error surface (covers CSRF/crumb & shape mismatches)
        return {
            "status": "error",
            "error_code": "YAHOOQUERY_FAILURE",
            "message": f"Failed to fetch data for {sym}: {str(e)}",
        }

INTRADAY_INTERVALS = {"1m","2m","5m","15m","30m","60m","90m","1h"}

def _f(x: Any):
    try:
        return float(x)
    except Exception:
        return None

def get_price_history(symbol: str, period: str = "1y", interval: str = "1d") -> Json:
    """
    Normalize yahooquery history to:
      {
        status: "ok",
        symbol, period, interval,
        points: [{t, o, h, l, c, v, adjclose}]
      }
    - t is epoch seconds UTC
    - o/h/l/c/v/adjclose are floats or None
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"status": "error", "error_code": "EMPTY_SYMBOL", "message": "Symbol is required"}

    try:
        tq = Ticker(sym, asynchronous=False, formatted=False, validate=False)
        df = tq.history(period=period, interval=interval)

        # Make sure we have a DataFrame
        if df is None or (hasattr(df, "empty") and df.empty):
            return {"status": "ok", "symbol": sym, "period": period, "interval": interval, "points": []}

        if isinstance(df, pd.Series):
            df = df.to_frame().T

        # MultiIndex (symbol, date) is common; flatten
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()  # gives 'symbol' and 'date' cols
            # If multiple symbols were requested (not here), filter to ours:
            if "symbol" in df.columns:
                df = df[df["symbol"].str.upper() == sym]
        else:
            df = df.reset_index()
            # Some shapes call the index column something else; align to 'date'
            if "index" in df.columns and "date" not in df.columns:
                df = df.rename(columns={"index": "date"})
            if "date" not in df.columns:
                # last resort, bail with empty
                return {"status": "ok", "symbol": sym, "period": period, "interval": interval, "points": []}

        # Coerce 'date' → tz-aware UTC and keep order
        # This handles datetime.date, naive datetimes, and strings.
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        # Some intervals don't include 'adjclose' – guard it
        has_adj = "adjclose" in df.columns

        points: List[Dict[str, Any]] = []
        for row in df.itertuples(index=False):
            # Access by attribute names (created from columns)
            d = getattr(row, "date", None)
            t = int(d.timestamp()) if isinstance(d, pd.Timestamp) else _to_epoch_utc(d)
            if t is None:
                continue

            points.append({
                "t": t,
                "o": _f(getattr(row, "open", None)),
                "h": _f(getattr(row, "high", None)),
                "l": _f(getattr(row, "low", None)),
                "c": _f(getattr(row, "close", None)),
                "v": _f(getattr(row, "volume", None)),
                "adjclose": _f(getattr(row, "adjclose", None)) if has_adj else None,
            })

        return {"status": "ok", "symbol": sym, "period": period, "interval": interval, "points": points}

    except Exception as e:
        return {"status": "error", "error_code": "YQ_HISTORY_FAILED", "message": str(e)}    
def _to_epoch_utc(x: Any) -> int | None:
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

def get_financials(symbol: str, period: str = "annual") -> Json:
    """
    Prefer yahooquery.Ticker(...).all_financial_data() to build statements.

    Output (latest -> oldest):
    {
      status, symbol, period: 'annual'|'quarterly',
      income_statement: [{date, revenue, gross_profit, operating_income, net_income, eps}],
      balance_sheet:    [{date, total_assets, total_liabilities, total_equity, cash, inventory, long_term_debt}],
      cash_flow:        [{date, operating_cash_flow, investing_cash_flow, financing_cash_flow, free_cash_flow}],
    }
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"status": "error", "error_code": "EMPTY_SYMBOL", "message": "Symbol is required"}

    # Which periodType(s) correspond to the UI toggle
    want_quarterly = (period == "quarterly")
    acceptable_period_types = {"3M"} if want_quarterly else {"12M", "FY"}  # Apple tends to use 3M / 12M

    try:
        tq = Ticker(sym, asynchronous=False, formatted=False, validate=False)

        # ---- 1) Try all_financial_data (rich, wide DataFrame)
        afd = tq.all_financial_data()
        if afd is not None and not getattr(afd, "empty", True):
            # normalize shape
            df = afd.copy()

            # bring symbol out of index if needed
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            else:
                df = df.reset_index(drop=False)

            # ensure symbol filter if present
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].astype(str).str.upper()
                df = df[df["symbol"] == sym]

            # unify date & periodType
            if "asOfDate" not in df.columns and "endDate" in df.columns:
                df = df.rename(columns={"endDate": "asOfDate"})
            # if "asOfDate" not in df.columns:
                # nothing to align -> fallback
                # return _financials_df_fallback(sym, period)

            # coerce date
            df["asOfDate"] = pd.to_datetime(df["asOfDate"], utc=True, errors="coerce")
            df = df.dropna(subset=["asOfDate"])

            # periodType filter (some tickers label annual as FY or 12M)
            if "periodType" in df.columns:
                df = df[df["periodType"].isin(acceptable_period_types)]
            # If no periodType column, keep all and let dates/fields speak

            # sort latest first
            df = df.sort_values("asOfDate", ascending=False)

            # Small helper to fetch from multiple candidate columns
            def col(row: pd.Series, *names: str) -> Optional[float]:
                for name in names:
                    if name in row and pd.notna(row[name]):
                        return _f(row[name])
                return None

            income: List[Dict[str, Any]] = []
            balance: List[Dict[str, Any]] = []
            cash_flow: List[Dict[str, Any]] = []

            for _, r in df.iterrows():
                iso = r["asOfDate"].date().isoformat()

                # Income
                income.append({
                    "date": iso,
                    # many names in AFD; prefer TotalRevenue/GrossProfit/OperatingIncome/NetIncome
                    "revenue":          col(r, "TotalRevenue", "OperatingRevenue", "Revenue"),
                    "gross_profit":     col(r, "GrossProfit"),
                    "operating_income": col(r, "OperatingIncome", "TotalOperatingIncomeAsReported"),
                    "net_income":       col(r, "NetIncome", "NetIncomeCommonStockholders", "NetIncomeIncludingNoncontrollingInterests"),
                    "eps":              col(r, "DilutedEPS", "BasicEPS", "EPS"),
                })

                # Balance
                balance.append({
                    "date": iso,
                    "total_assets":      col(r, "TotalAssets"),
                    "total_liabilities": col(r, "TotalLiabilitiesNetMinorityInterest", "TotalLiabilities"),
                    "total_equity":      col(r, "StockholdersEquity", "TotalEquityGrossMinorityInterest", "CommonStockEquity"),
                    "cash":              col(r, "CashAndCashEquivalents", "Cash", "CashFinancial"),
                    "inventory":         col(r, "Inventory"),
                    "long_term_debt":    col(r, "LongTermDebt", "LongTermDebtAndCapitalLeaseObligation"),
                })

                # Cash Flow
                ocf = col(r, "OperatingCashFlow")
                icf = col(r, "InvestingCashFlow")
                fcf = col(r, "FreeCashFlow")
                if fcf is None:
                    capex = col(r, "CapitalExpenditure")
                    if ocf is not None and capex is not None:
                        fcf = ocf - capex
                cash_flow.append({
                    "date": iso,
                    "operating_cash_flow":  ocf,
                    "investing_cash_flow":  icf,
                    "financing_cash_flow":  col(r, "CashFlowFromContinuingFinancingActivities", "FinancingCashFlow"),
                    "free_cash_flow":       fcf,
                })

            return {
                "status": "ok",
                "symbol": sym,
                "period": "quarterly" if want_quarterly else "annual",
                "income_statement": income,
                "balance_sheet": balance,
                "cash_flow": cash_flow,
            }

        # ---- 2) Fallback to statement DFs if AFD is empty
        return _financials_df_fallback(sym, period)
        # return {"status": "error", "error_code": "YQ_FINANCIALS_FAILED", "message": 'Fallback to statement DFs if AFD is empty'}

    except Exception as e:
        return {"status": "error", "error_code": "YQ_FINANCIALS_FAILED", "message": str(e)}

def _financials_df_fallback(sym: str, period: str) -> Json:
    """
    Your previous DF-based implementation; called if all_financial_data()
    is unavailable or too sparse. Uses .income_statement/.balance_sheet/.cash_flow
    and returns the same normalized shape.
    """
    # reuse the DF logic we discussed earlier; if you already have it defined as
    # a separate function, just call it here:
    try:
        freq = "q" if period == "quarterly" else "a"
        tq = Ticker(sym, asynchronous=False, formatted=False, validate=False)

        def _prep_df(df: Any) -> pd.DataFrame:
            if df is None:
                return pd.DataFrame()
            if isinstance(df, pd.Series):
                df = df.to_frame().T
            df = df.reset_index()
            for cand in ("asOfDate", "endDate", "date"):
                if cand in df.columns:
                    if cand != "date":
                        df = df.rename(columns={cand: "date"})
                    break
            if "date" not in df.columns:
                if "index" in df.columns:
                    df = df.rename(columns={"index": "date"})
                else:
                    return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date", ascending=False)
            return df

        def _val(row: pd.Series, *keys: str) -> Optional[float]:
            for k in keys:
                if k in row:
                    return _f(row[k])
            return None

        inc_df = _prep_df(tq.income_statement(frequency=freq))
        bal_df = _prep_df(tq.balance_sheet(frequency=freq))
        cfs_df = _prep_df(tq.cash_flow(frequency=freq))

        income = []
        for _, r in inc_df.iterrows():
            income.append({
                "date": r["date"].date().isoformat(),
                "revenue":           _val(r, "totalRevenue", "revenue"),
                "gross_profit":      _val(r, "grossProfit"),
                "operating_income":  _val(r, "operatingIncome"),
                "net_income":        _val(r, "netIncome"),
                "eps":               _val(r, "dilutedEps", "dilutedEPS", "eps"),
            })

        balance = []
        for _, r in bal_df.iterrows():
            balance.append({
                "date": r["date"].date().isoformat(),
                "total_assets":      _val(r, "totalAssets"),
                "total_liabilities": _val(r, "totalLiab", "totalLiabilities"),
                "total_equity":      _val(r, "totalStockholderEquity", "totalShareholderEquity"),
                "cash":              _val(r, "cash", "cashAndCashEquivalentsAtCarryingValue", "cashAndCashEquivalents"),
                "inventory":         _val(r, "inventory"),
                "long_term_debt":    _val(r, "longTermDebt"),
            })

        cash_flow = []
        for _, r in cfs_df.iterrows():
            ocf = _val(r, "totalCashFromOperatingActivities", "operatingCashFlow")
            icf = _val(r, "totalCashflowsFromInvestingActivities", "investingCashFlow")
            fcf = _val(r, "freeCashflow", "freeCashFlow")
            if fcf is None:
                capex = _val(r, "capitalExpenditures")
                if ocf is not None and capex is not None:
                    fcf = ocf - capex
            cash_flow.append({
                "date": r["date"].date().isoformat(),
                "operating_cash_flow":  ocf,
                "investing_cash_flow":  icf,
                "financing_cash_flow":  _val(r, "totalCashFromFinancingActivities", "financingCashFlow"),
                "free_cash_flow":       fcf,
            })

        return {
            "status": "ok",
            "symbol": sym,
            "period": "quarterly" if period == "quarterly" else "annual",
            "income_statement": income,
            "balance_sheet": balance,
            "cash_flow": cash_flow,
        }

    except Exception as e:
        return {"status": "error", "error_code": "YQ_FINANCIALS_FALLBACK_FAILED", "message": str(e)}

# --- new little helpers for calendarEvents quirks ---

import re
_dt_suffix_pat = re.compile(r"(:S|[ T]S)$", re.IGNORECASE)

def _parse_weird_cal_dt(val: Any) -> Optional[str]:
    """
    '2025-10-30 16:00:S' -> '2025-10-30' (ISO date)
    also accepts list[...] / epoch / dict{'raw':...}
    """
    # unwrap list
    if isinstance(val, list) and val:
        val = val[0]
    # epoch
    if isinstance(val, (int, float)):
        return _date_iso(val)
    # dict{'raw':...}
    if isinstance(val, dict) and "raw" in val:
        return _date_iso(val["raw"])
    # string with trailing ':S'
    if isinstance(val, str):
        s = _dt_suffix_pat.sub("", val.strip())  # drop weird suffix
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date().isoformat()
    return None

def _parse_epoch_list_to_iso(val: Any) -> Optional[str]:
    """
    [1753995600] -> '2025-08-31T12:00:00Z' (example)
    """
    if isinstance(val, list) and val:
        val = val[0]
    if isinstance(val, (int, float)):
        sec = int(val / 1000) if val > 1e12 else int(val)
        return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat()
    return None

def _quarter_label(iso_date: str) -> str:
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

# ---------- NEW: earnings ----------
def get_earnings(symbol: str) -> Json:
    """
    {
      status, symbol,
      next_earnings_date: 'YYYY-MM-DD' | null,
      next: {
        date: 'YYYY-MM-DD' | null,
        call_time_utc: 'YYYY-MM-DDTHH:MM:SSZ' | null,
        is_estimate: bool | null,
        eps_avg: float|null, eps_low: float|null, eps_high: float|null,
        revenue_avg: float|null, revenue_low: float|null, revenue_high: float|null
      },
      events: {
        ex_dividend_date: 'YYYY-MM-DD' | null,
        dividend_date: 'YYYY-MM-DD' | null
      },
      history: [{ date, period, actual, estimate, surprisePct }]
    }
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"status": "error", "error_code": "EMPTY_SYMBOL", "message": "Symbol is required"}

    try:
        tq = Ticker(sym, asynchronous=False, formatted=False, validate=False)
        res = tq.get_modules(["calendarEvents", "earningsHistory"])
        node = _get_symbol_node(res, sym)

        # ---- calendar / next earnings ----
        cal = node.get("calendarEvents") or {}
        nxt_date = None
        nxt_obj = None
        ex_div = None
        div_date = None

        # calendarEvents.earnings {...}
        earnings_block = cal.get("earnings") if isinstance(cal, dict) else None
        if isinstance(earnings_block, dict):
            nxt_date = _parse_weird_cal_dt(earnings_block.get("earningsDate"))
            call_iso = _parse_epoch_list_to_iso(earnings_block.get("earningsCallDate"))
            nxt_obj = {
                "date": nxt_date,
                "call_time_utc": call_iso,
                "is_estimate": bool(earnings_block.get("isEarningsDateEstimate")) if "isEarningsDateEstimate" in earnings_block else None,
                "eps_avg": _f(earnings_block.get("earningsAverage")),
                "eps_low": _f(earnings_block.get("earningsLow")),
                "eps_high": _f(earnings_block.get("earningsHigh")),
                "revenue_avg": _f(earnings_block.get("revenueAverage")),
                "revenue_low": _f(earnings_block.get("revenueLow")),
                "revenue_high": _f(earnings_block.get("revenueHigh")),
            }

        # ex-dividend & dividend dates (strings like '2025-08-10 20:00:00')
        ex_div = _date_iso(cal.get("exDividendDate"))
        div_date = _date_iso(cal.get("dividendDate"))

        # ---- earnings history ----
        history: List[Dict[str, Any]] = []
        eh = node.get("earningsHistory")
        hist_list = []
        if isinstance(eh, dict):
            hist_list = eh.get("history") or []
        elif isinstance(eh, list):
            hist_list = eh

        for r in hist_list or []:
            d = _date_iso(r.get("reportDate") or r.get("quarter") or r.get("date"))
            if not d:
                continue
            history.append({
                "date": d,
                "period": _quarter_label(d),
                "actual": _f(r.get("epsActual") or r.get("actual")),
                "estimate": _f(r.get("epsEstimate") or r.get("estimate")),
                "surprisePct": _f(r.get("surprisePercent") or r.get("surprisePct")),
            })

        # sort latest → oldest
        history.sort(key=lambda x: x["date"], reverse=True)

        return {
            "status": "ok",
            "symbol": sym,
            "next_earnings_date": nxt_date,
            "next": nxt_obj,
            "events": {
                "ex_dividend_date": ex_div,
                "dividend_date": div_date,
            },
            "history": history,
        }

    except Exception as e:
        return {"status": "error", "error_code": "YQ_EARNINGS_FAILED", "message": str(e)}

# ---------- NEW: analyst ----------
def get_analyst(symbol: str) -> Json:
    """
    Normalize to:
    {
      status, symbol,
      recommendation_key, recommendation,
      price_target_low, price_target_mean, price_target_high,
      trend: [{period, strongBuy, buy, hold, sell, strongSell}]
    }
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"status": "error", "error_code": "EMPTY_SYMBOL", "message": "Symbol is required"}

    try:
        tq = Ticker(sym, asynchronous=False, formatted=False, validate=False)
        res = tq.get_modules(["financialData", "recommendationTrend"])
        node = _get_symbol_node(res, sym)

        fin = node.get("financialData") or {}
        rec = node.get("recommendationTrend") or {}

        # recommendation score/key
        recommendation = _f(fin.get("recommendationMean"))
        recommendation_key = rec.get("trend", [{}])[0].get("recommendationKey") if isinstance(rec.get("trend"), list) and rec.get("trend") else fin.get("recommendationKey")
        if not recommendation_key:
            recommendation_key = fin.get("recommendationKey")

        # price targets
        price_target_low = _f(fin.get("targetLowPrice") or fin.get("priceTargetLow"))
        price_target_mean = _f(fin.get("targetMeanPrice") or fin.get("priceTargetMean"))
        price_target_high = _f(fin.get("targetHighPrice") or fin.get("priceTargetHigh"))

        trend_list = []
        t = rec.get("trend")
        if isinstance(t, list):
            for r in t:
                trend_list.append({
                    "period": r.get("period"),
                    "strongBuy": int(r.get("strongBuy") or 0),
                    "buy": int(r.get("buy") or 0),
                    "hold": int(r.get("hold") or 0),
                    "sell": int(r.get("sell") or 0),
                    "strongSell": int(r.get("strongSell") or 0),
                })

        return {
            "status": "ok",
            "symbol": sym,
            "recommendation_key": recommendation_key,
            "recommendation": recommendation,
            "price_target_low": price_target_low,
            "price_target_mean": price_target_mean,
            "price_target_high": price_target_high,
            "trend": trend_list,
        }

    except Exception as e:
        return {"status": "error", "error_code": "YQ_ANALYST_FAILED", "message": str(e)}

# ---------- NEW: company profile/overview ----------
def get_overview(symbol: str) -> Json:
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"status": "error", "error_code": "EMPTY_SYMBOL", "message": "Symbol is required"}

    try:
        tq = Ticker(sym, asynchronous=False, formatted=False, validate=False)

        res = tq.get_modules(["summaryProfile"]) or {}
        node = _get_symbol_node(res, sym)  # always dict from our helper

        # Some tickers return {summaryProfile: {...}}, others return the profile dict directly.
        if isinstance(node, dict) and isinstance(node.get("summaryProfile"), dict):
            prof = node["summaryProfile"]
        elif isinstance(node, dict):
            prof = node
        else:
            prof = {}

        def pick(*keys):
            for k in keys:
                v = prof.get(k)
                if v not in (None, "", []):
                    return v
            return None

        return {
            "status": "ok",
            "symbol": sym,
            "sector": pick("sector"),
            "industry": pick("industry"),
            "full_time_employees": pick("fullTimeEmployees", "full_time_employees"),
            "city": pick("city"),
            "state": pick("state"),
            "country": pick("country"),
            "zip": pick("zip"),
            "address1": pick("address1", "address"),
            "phone": pick("phone"),
            "website": pick("website"),
            "ir_website": pick("irWebsite"),
            "long_business_summary": pick("longBusinessSummary", "long_business_summary"),
        }

    except Exception as e:
        return {"status": "error", "error_code": "YQ_PROFILE_FAILED", "message": str(e)}
    
def _date_iso(d: Any) -> Optional[str]:
    t = _to_epoch_utc(d)
    if t is None:
        return None
    return datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat()

def _first_list(obj: Any) -> List[dict]:
    """
    Yahoo modules sometimes nest lists under varying keys. This returns
    the first list of dicts found within a module dict.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                lst = _first_list(v)
                if lst:
                    return lst
    return []


def _get_symbol_node(res: Any, sym: str) -> Dict[str, Any]:
    """
    get_modules(...) can return:
      - { "AAPL": { ...modules... } }
      - { ...modules... }  (already the node)
      - { "error": ... } / string / None
    Normalize to a dict or {}.
    """
    if isinstance(res, dict):
        # preferred: nested by symbol
        sub = res.get(sym)
        if isinstance(sub, dict):
            return sub
        # sometimes it's already the node (modules at top level)
        return res if any(isinstance(v, (dict, list)) for v in res.values()) else {}
    return {}


# --- plug your cache helpers here (no-ops shown) ---
def _cache_get_many(symbols: List[str]) -> Dict[str, Optional[Json]]:
    return {s: None for s in symbols}

def _cache_set_many(kv: Dict[str, Json], ttl_seconds: int = 60) -> None:
    pass

def _missing_fields(payload: Dict[str, Any]) -> List[str]:
    return [k for k in ("current_price", "currency", "previous_close") if payload.get(k) is None]

def get_full_stock_data_many(symbols: Iterable[str]) -> Dict[str, Json]:
    """
    Bulk version of get_full_stock_data:
      - Single yahooquery.Ticker call for many symbols
      - Fetch only minimal modules needed
      - Returns {symbol: payload}
    """
    syms = [s.upper().strip() for s in symbols if s and s.strip()]
    if not syms:
        return {}

    # cache first
    cached_map = _cache_get_many(syms)
    fresh_needed = [s for s, v in cached_map.items() if not v]

    out: Dict[str, Json] = {s: v for s, v in cached_map.items() if v}

    if not fresh_needed:
        return out

    # Minimal modules for your payload
    # price: name/currency/exchange/market times, previous close
    # summaryDetail: current price, 52w range, dividend_yield, trailing PE etc.
    # financialData: recommendation, targetMeanPrice, profitability metrics
    # defaultKeyStatistics: priceToBook, sometimes beta fallback
    modules = ["price", "summaryDetail", "financialData", "defaultKeyStatistics"]

    # One Ticker for all missing symbols
    tq = Ticker(fresh_needed, asynchronous=False, formatted=False, validate=False)

    # Single network round for each module set
    price_all = tq.get_modules("price") or {}
    sumdet_all = tq.get_modules("summaryDetail") or {}
    fin_all = tq.get_modules("financialData") or {}
    stats_all = tq.get_modules("defaultKeyStatistics") or {}

    now_iso = datetime.now(timezone.utc).isoformat()

    write_back: Dict[str, Json] = {}

    for sym in fresh_needed:
        try:
            price = _ensure_symbol_dict(price_all, sym)
            sd    = _ensure_symbol_dict(sumdet_all, sym)
            fd    = _ensure_symbol_dict(fin_all, sym)
            ks    = _ensure_symbol_dict(stats_all, sym)

            short_name = price.get("shortName") or price.get("longName")
            currency   = price.get("currency")
            exchange   = price.get("exchangeName") or price.get("fullExchangeName")

            current = _fnum(sd.get("regularMarketPrice") or price.get("regularMarketPrice"))
            previous = _fnum(price.get("regularMarketPreviousClose") or sd.get("previousClose"))

            # range & fundamentals
            high_52 = _fnum(sd.get("fiftyTwoWeekHigh"))
            low_52  = _fnum(sd.get("fiftyTwoWeekLow"))

            pe_ratio     = _fnum(sd.get("trailingPE"))
            forward_pe   = _fnum(sd.get("forwardPE") or fd.get("forwardPE"))
            price_to_book = _fnum(ks.get("priceToBook") or sd.get("priceToBook"))
            beta         = _fnum(sd.get("beta") or ks.get("beta"))
            dividend_yield = _fnum(sd.get("dividendYield"))
            market_cap   = _fnum(sd.get("marketCap") or price.get("marketCap"))

            return_on_equity = _fnum(fd.get("returnOnEquity"))
            profit_margins   = _fnum(fd.get("profitMargins"))
            earnings_growth  = _fnum(fd.get("earningsGrowth"))
            revenue_growth   = _fnum(fd.get("revenueGrowth"))
            recommendation   = _fnum(fd.get("recommendationMean"))
            recommendation_key = fd.get("recommendationKey")
            target_price     = _fnum(fd.get("targetMeanPrice"))

            quote_ts = (
                price.get("regularMarketTime")
                or price.get("postMarketTime")
                or price.get("preMarketTime")
            )
            quote_time_utc = _iso_utc_from_ts(quote_ts)

            day_change      = (current - previous) if (current is not None and previous is not None) else None
            day_change_pct  = _pct(current, previous)
            dist_high_pct   = _dist_pct(current, high_52)
            dist_low_pct    = _dist_pct(current, low_52)

            payload: Json = {
                "status": "ok",
                "symbol": sym,
                "name": short_name,
                "currency": currency,
                "exchange": exchange,
                "quote_time_utc": quote_time_utc,
                # Prices
                "current_price": current,
                "previous_close": previous,
                "day_change": day_change,
                "day_change_pct": day_change_pct,
                # Range
                "52_week_high": high_52,
                "52_week_low": low_52,
                "distance_from_52w_high_pct": dist_high_pct,
                "distance_from_52w_low_pct": dist_low_pct,
                # Fundamentals
                "market_cap": market_cap,
                "pe_ratio": pe_ratio,
                "forward_pe": forward_pe,
                "price_to_book": price_to_book,
                "beta": beta,
                "dividend_yield": dividend_yield,
                "return_on_equity": return_on_equity,
                "profit_margins": profit_margins,
                "earnings_growth": earnings_growth,
                "revenue_growth": revenue_growth,
                "recommendation": recommendation,
                "recommendation_key": recommendation_key,
                "target_price": target_price,
                # Quality & provenance
                "data_quality": {
                    "source": "Yahoo Finance via yahooquery",
                    "fetched_at_utc": now_iso,
                    "is_stale": False if not quote_time_utc else (
                        (datetime.now(timezone.utc) - datetime.fromisoformat(quote_time_utc.replace("Z", "+00:00"))) > timedelta(days=2)
                    ),
                    "missing_fields": _missing_fields({
                        "current_price": current,
                        "currency": currency,
                        "previous_close": previous,
                    }),
                },
            }

            out[sym] = payload
            write_back[sym] = payload

        except Exception as e:
            out[sym] = {
                "status": "error",
                "symbol": sym,
                "error_code": "YQ_MANY_FAILURE",
                "message": str(e),
            }

    if write_back:
        _cache_set_many(write_back, ttl_seconds=60)  # minute-bucket is typical

    return out