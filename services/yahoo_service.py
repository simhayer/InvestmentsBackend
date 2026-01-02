# services/yahoo_service.py
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from yahooquery import Ticker

from utils.common_helpers import safe_float as _fnum, retry
from services.helpers.yahoo_helpers import (
    pct,
    dist_pct,
    iso_utc_from_ts,
    to_epoch_utc,
    date_iso,
    parse_weird_cal_dt,
    quarter_label,
)

from services.cache.cache_backend import (
    cache_get_many,
    cache_set_many,
)
from services.cache.cache_utils import cacheable

Number = Optional[float]
Json = Dict[str, Any]

# -------------------------
# TTL policy
# -------------------------
TTL_QUOTES_SEC = 60
TTL_EARNINGS_SEC = 12 * 60 * 60        # 12h (6–24h)
TTL_FINANCIALS_SEC = 48 * 60 * 60      # 48h (24–72h)
TTL_PROFILE_SEC = 24 * 60 * 60         # 24h
TTL_HISTORY_SEC = 15 * 60              # 15m
TTL_ANALYST_SEC = 24 * 60 * 60         # daily-ish

def _sym(symbol: str) -> str:
    return (symbol or "").upper().strip()

def _ck(sym: str, kind: str) -> str:
    """Cache key helper to avoid collisions across endpoints."""
    s = _sym(sym)
    return f"{kind}:{s}"

def _extract_stock_payload(sym: str, price: dict, sd: dict, fd: dict, ks: dict) -> Json:
    short_name = price.get("shortName") or price.get("longName")
    currency = price.get("currency")
    exchange = price.get("exchangeName") or price.get("fullExchangeName")

    current = _fnum(sd.get("regularMarketPrice") or price.get("regularMarketPrice"))
    previous = _fnum(price.get("regularMarketPreviousClose") or sd.get("previousClose"))

    # 52-week range
    high_52 = _fnum(sd.get("fiftyTwoWeekHigh"))
    low_52 = _fnum(sd.get("fiftyTwoWeekLow"))

    # Fundamentals
    pe_ratio = _fnum(sd.get("trailingPE"))
    forward_pe = _fnum(sd.get("forwardPE") or fd.get("forwardPE"))
    price_to_book = _fnum(ks.get("priceToBook") or sd.get("priceToBook"))
    beta = _fnum(sd.get("beta") or ks.get("beta"))
    dividend_yield = _fnum(sd.get("dividendYield"))
    market_cap = _fnum(sd.get("marketCap") or price.get("marketCap"))

    return_on_equity = _fnum(fd.get("returnOnEquity"))
    profit_margins = _fnum(fd.get("profitMargins"))
    earnings_growth = _fnum(fd.get("earningsGrowth"))
    revenue_growth = _fnum(fd.get("revenueGrowth"))
    recommendation = _fnum(fd.get("recommendationMean"))
    recommendation_key = fd.get("recommendationKey")
    target_price = _fnum(fd.get("targetMeanPrice"))

    quote_ts = (
        price.get("regularMarketTime")
        or price.get("postMarketTime")
        or price.get("preMarketTime")
    )
    quote_time_utc = iso_utc_from_ts(quote_ts)

    day_change = (current - previous) if (current is not None and previous is not None) else None
    day_change_pct = pct(current, previous)
    dist_high_pct = dist_pct(current, high_52)
    dist_low_pct = dist_pct(current, low_52)

    is_stale = False
    if quote_time_utc:
        try:
            qt = datetime.fromisoformat(quote_time_utc.replace("Z", "+00:00"))
            is_stale = (datetime.now(timezone.utc) - qt) > timedelta(days=2)
        except Exception:
            pass

    missing = [
        k for k, v in {
            "current_price": current,
            "currency": currency,
            "previous_close": previous,
        }.items()
        if v is None
    ]

    return {
        "status": "ok",
        "symbol": sym,
        "name": short_name,
        "currency": currency,
        "exchange": exchange,
        "quote_time_utc": quote_time_utc,
        "current_price": current,
        "previous_close": previous,
        "day_change": day_change,
        "day_change_pct": day_change_pct,
        "52_week_high": high_52,
        "52_week_low": low_52,
        "distance_from_52w_high_pct": dist_high_pct,
        "distance_from_52w_low_pct": dist_low_pct,
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
        "data_quality": {
            "source": "Yahoo Finance",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "is_stale": is_stale,
            "missing_fields": missing,
        },
    }

# -------------------------
# Quotes / fundamentals
# -------------------------
@cacheable(
    ttl=TTL_QUOTES_SEC,
    key_fn=lambda symbol: _ck(_sym(symbol), "quote"),
)
def get_full_stock_data(symbol: str) -> Json:
    sym = _sym(symbol)
    if not sym:
        return {"status": "error", "error_code": "EMPTY_SYMBOL", "message": "Symbol required"}

    try:
        tq = Ticker(sym, asynchronous=False, formatted=False)
        raw = retry(lambda: tq.get_modules(["price", "summaryDetail", "financialData", "defaultKeyStatistics"])) or {}
        node = _get_symbol_node(raw, sym)

        if not node or "price" not in node:
            raise ValueError(f"No data found for {sym}")

        return _extract_stock_payload(
            sym,
            node.get("price", {}) or {},
            node.get("summaryDetail", {}) or {},
            node.get("financialData", {}) or {},
            node.get("defaultKeyStatistics", {}) or {},
        )
    except Exception as e:
        return {"status": "error", "error_code": "YAHOO_FETCH_ERROR", "message": str(e)}

# -------------------------
# Price history (optional cached)
# -------------------------
@cacheable(
    ttl=TTL_HISTORY_SEC,
    key_fn=lambda symbol, period="1y", interval="1d": _ck(_sym(symbol), f"history:{period}:{interval}"),
)
def get_price_history(symbol: str, period: str = "1y", interval: str = "1d") -> Json:
    sym = _sym(symbol)
    if not sym:
        return {"status": "error", "error_code": "EMPTY_SYMBOL", "message": "Symbol required"}

    try:
        tq = Ticker(sym, asynchronous=False, formatted=False)
        df = tq.history(period=period, interval=interval)

        if df is None or (hasattr(df, "empty") and df.empty):
            return {"status": "ok", "symbol": sym, "period": period, "interval": interval, "points": []}

        df = df.reset_index()

        if "symbol" in df.columns:
            df = df[df["symbol"].astype(str).str.upper() == sym]

        date_col = next((c for c in df.columns if c in ["date", "index", "asOfDate"]), None)
        if date_col and date_col != "date":
            df = df.rename(columns={date_col: "date"})

        if "date" not in df.columns:
            return {"status": "ok", "symbol": sym, "period": period, "interval": interval, "points": []}

        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        has_adj = "adjclose" in df.columns

        points: List[Dict[str, Any]] = []
        for row in df.itertuples(index=False):
            d = getattr(row, "date", None)
            t = int(d.timestamp()) if isinstance(d, pd.Timestamp) else to_epoch_utc(d)
            if t is None:
                continue

            points.append({
                "t": t,
                "o": _fnum(getattr(row, "open", None)),
                "h": _fnum(getattr(row, "high", None)),
                "l": _fnum(getattr(row, "low", None)),
                "c": _fnum(getattr(row, "close", None)),
                "v": _fnum(getattr(row, "volume", None)),
                "adjclose": _fnum(getattr(row, "adjclose", None)) if has_adj else None,
            })

        return {"status": "ok", "symbol": sym, "period": period, "interval": interval, "points": points}

    except Exception as e:
        return {"status": "error", "error_code": "YQ_HISTORY_FAILED", "message": str(e)}

# -------------------------
# Financials (cached 48h)
# -------------------------
@cacheable(
    ttl=TTL_FINANCIALS_SEC,
    key_fn=lambda symbol, period="annual": _ck(_sym(symbol), f"financials:{'quarterly' if period=='quarterly' else 'annual'}"),
)
def get_financials(symbol: str, period: str = "annual") -> Json:
    sym = _sym(symbol)
    if not sym:
        return {"status": "error", "error_code": "EMPTY_SYMBOL", "message": "Symbol required"}

    want_q = (period == "quarterly")
    acceptable = {"3M"} if want_q else {"12M", "FY"}

    try:
        tq = Ticker(sym, asynchronous=False, formatted=False)
        afd = tq.all_financial_data()

        if (
            afd is None
            or not isinstance(afd, pd.DataFrame)
            or afd.empty
            or not any(c in afd.columns for c in ("TotalRevenue", "OperatingRevenue"))
        ):
            return _financials_df_fallback(sym, period)

        df = pd.DataFrame(afd).reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"].astype(str).str.upper() == sym]

        date_col = "asOfDate" if "asOfDate" in df.columns else "endDate"
        if date_col not in df.columns:
            return _financials_df_fallback(sym, period)

        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        df = df.dropna(subset=[date_col])

        if "periodType" in df.columns:
            df = df[df["periodType"].isin(acceptable)]

        df = df.sort_values(date_col, ascending=False)

        def _get(r, *keys):
            for k in keys:
                if k in r and pd.notna(r[k]):
                    return _fnum(r[k])
            return None

        income: List[Dict[str, Any]] = []
        balance: List[Dict[str, Any]] = []
        cash: List[Dict[str, Any]] = []

        for _, r in df.iterrows():
            iso = r[date_col].date().isoformat()

            income.append({
                "date": iso,
                "revenue": _get(r, "TotalRevenue", "OperatingRevenue", "Revenue"),
                "gross_profit": _get(r, "GrossProfit"),
                "operating_income": _get(r, "OperatingIncome", "TotalOperatingIncomeAsReported"),
                "net_income": _get(r, "NetIncome", "NetIncomeCommonStockholders", "NetIncomeIncludingNoncontrollingInterests"),
                "eps": _get(r, "DilutedEPS", "BasicEPS", "EPS"),
            })

            balance.append({
                "date": iso,
                "total_assets": _get(r, "TotalAssets"),
                "total_liabilities": _get(r, "TotalLiabilitiesNetMinorityInterest", "TotalLiabilities"),
                "total_equity": _get(r, "StockholdersEquity", "CommonStockEquity", "TotalEquityGrossMinorityInterest"),
                "cash": _get(r, "CashAndCashEquivalents", "CashAndCashEquivalentsAtCarryingValue", "Cash"),
                "inventory": _get(r, "Inventory"),
                "long_term_debt": _get(r, "LongTermDebt", "LongTermDebtAndCapitalLeaseObligation"),
            })

            ocf = _get(r, "OperatingCashFlow")
            icf = _get(r, "InvestingCashFlow")
            fcf = _get(r, "FreeCashFlow")

            if fcf is None:
                capex = _get(r, "CapitalExpenditure", "CapitalExpenditures")
                if ocf is not None and capex is not None:
                    fcf = ocf - capex

            cash.append({
                "date": iso,
                "operating_cash_flow": ocf,
                "investing_cash_flow": icf,
                "financing_cash_flow": _get(r, "FinancingCashFlow", "CashFlowFromContinuingFinancingActivities"),
                "free_cash_flow": fcf,
            })

        return {
            "status": "ok",
            "symbol": sym,
            "period": "quarterly" if want_q else "annual",
            "income_statement": income,
            "balance_sheet": balance,
            "cash_flow": cash,
        }

    except Exception as e:
        return {"status": "error", "error_code": "YQ_FINANCIALS_FAILED", "message": str(e)}

def _financials_df_fallback(sym: str, period: str) -> Json:
    """Fallback using statement-specific DFs."""
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
                if k in row and pd.notna(row[k]):
                    return _fnum(row[k])
            return None

        inc_df = _prep_df(tq.income_statement(frequency=freq))
        bal_df = _prep_df(tq.balance_sheet(frequency=freq))
        cfs_df = _prep_df(tq.cash_flow(frequency=freq))

        income: List[Dict[str, Any]] = []
        for _, r in inc_df.iterrows():
            income.append({
                "date": r["date"].date().isoformat(),
                "revenue": _val(r, "totalRevenue", "revenue"),
                "gross_profit": _val(r, "grossProfit"),
                "operating_income": _val(r, "operatingIncome"),
                "net_income": _val(r, "netIncome"),
                "eps": _val(r, "dilutedEps", "dilutedEPS", "eps"),
            })

        balance: List[Dict[str, Any]] = []
        for _, r in bal_df.iterrows():
            balance.append({
                "date": r["date"].date().isoformat(),
                "total_assets": _val(r, "totalAssets"),
                "total_liabilities": _val(r, "totalLiab", "totalLiabilities"),
                "total_equity": _val(r, "totalStockholderEquity", "totalShareholderEquity"),
                "cash": _val(r, "cash", "cashAndCashEquivalentsAtCarryingValue", "cashAndCashEquivalents"),
                "inventory": _val(r, "inventory"),
                "long_term_debt": _val(r, "longTermDebt"),
            })

        cash_flow: List[Dict[str, Any]] = []
        for _, r in cfs_df.iterrows():
            ocf = _val(r, "totalCashFromOperatingActivities", "operatingCashFlow")
            icf = _val(r, "totalCashflowsFromInvestingActivities", "investingCashFlow")
            fcf = _val(r, "freeCashflow", "freeCashFlow")
            if fcf is None:
                capex = _val(r, "capitalExpenditures", "CapitalExpenditure", "capitalExpenditure")
                if ocf is not None and capex is not None:
                    fcf = ocf - capex

            cash_flow.append({
                "date": r["date"].date().isoformat(),
                "operating_cash_flow": ocf,
                "investing_cash_flow": icf,
                "financing_cash_flow": _val(r, "totalCashFromFinancingActivities", "financingCashFlow"),
                "free_cash_flow": fcf,
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

# -------------------------
# Earnings (cached 12h)
# -------------------------
@cacheable(
    ttl=TTL_EARNINGS_SEC,
    key_fn=lambda symbol: _ck(_sym(symbol), "earnings"),
)
def get_earnings(symbol: str) -> Json:
    sym = _sym(symbol)
    if not sym:
        return {"status": "error", "error_code": "EMPTY_SYMBOL", "message": "Symbol required"}

    try:
        tq = Ticker(sym, asynchronous=False, formatted=False)
        res = tq.get_modules(["calendarEvents", "earningsHistory"]) or {}
        node = _get_symbol_node(res, sym)

        cal = (node.get("calendarEvents") or {}).get("earnings") or {}
        nxt_date = parse_weird_cal_dt(cal.get("earningsDate"))

        history: List[Dict[str, Any]] = []
        hist_raw = (node.get("earningsHistory") or {}).get("history") or []
        for r in hist_raw:
            d = date_iso(r.get("reportDate") or r.get("quarter") or r.get("date"))
            if not d:
                continue
            history.append({
                "date": d,
                "period": quarter_label(d),
                "actual": _fnum(r.get("epsActual")),
                "estimate": _fnum(r.get("epsEstimate")),
                "surprisePct": _fnum(r.get("surprisePercent")),
            })

        return {
            "status": "ok",
            "symbol": sym,
            "next_earnings_date": nxt_date,
            "history": sorted(history, key=lambda x: x["date"], reverse=True),
        }

    except Exception as e:
        return {"status": "error", "error_code": "YQ_EARNINGS_FAILED", "message": str(e)}

# -------------------------
# Analyst (cached daily)
# -------------------------
@cacheable(
    ttl=TTL_ANALYST_SEC,
    key_fn=lambda symbol: _ck(_sym(symbol), "analyst"),
)
def get_analyst(symbol: str) -> Json:
    sym = _sym(symbol)
    if not sym:
        return {"status": "error", "error_code": "EMPTY_SYMBOL", "message": "Symbol is required"}

    try:
        tq = Ticker(sym, asynchronous=False, formatted=False, validate=False)
        res = tq.get_modules(["financialData", "recommendationTrend"]) or {}
        node = _get_symbol_node(res, sym)

        fin = node.get("financialData") or {}
        rec = node.get("recommendationTrend") or {}

        recommendation = _fnum(fin.get("recommendationMean"))

        recommendation_key = None
        t = rec.get("trend")
        if isinstance(t, list) and t:
            recommendation_key = t[0].get("recommendationKey")
        if not recommendation_key:
            recommendation_key = fin.get("recommendationKey")

        price_target_low = _fnum(fin.get("targetLowPrice") or fin.get("priceTargetLow"))
        price_target_mean = _fnum(fin.get("targetMeanPrice") or fin.get("priceTargetMean"))
        price_target_high = _fnum(fin.get("targetHighPrice") or fin.get("priceTargetHigh"))

        trend_list: List[Dict[str, Any]] = []
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

# -------------------------
# Company profile / overview (cached 24h)
# -------------------------
@cacheable(
    ttl=TTL_PROFILE_SEC,
    key_fn=lambda symbol: _ck(_sym(symbol), "profile"),
)
def get_overview(symbol: str) -> Json:
    sym = _sym(symbol)
    if not sym:
        return {"status": "error", "error_code": "EMPTY_SYMBOL", "message": "Symbol is required"}

    try:
        tq = Ticker(sym, asynchronous=False, formatted=False, validate=False)
        res = tq.get_modules(["summaryProfile"]) or {}
        node = _get_symbol_node(res, sym)

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

# -------------------------
# Helpers
# -------------------------
def _first_list(obj: Any) -> List[dict]:
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
    """Normalize yahooquery get_modules response to the dict for `sym`."""
    if not isinstance(res, dict):
        return {}

    if sym in res and isinstance(res[sym], dict):
        return res[sym]

    for k, v in res.items():
        if isinstance(k, str) and k.upper() == sym and isinstance(v, dict):
            return v

    return res if any(isinstance(v, (dict, list)) for v in res.values()) else {}

# -------------------------
# Bulk quotes (manual caching is best here)
# -------------------------
def get_full_stock_data_many(symbols: Iterable[str]) -> Dict[str, Json]:
    syms = sorted({_sym(s) for s in symbols if s and str(s).strip()})
    if not syms:
        return {}

    # cache keys are quote:<SYM>
    key_pairs = [(_ck(sym, "quote"), sym) for sym in syms]
    cached_map = cache_get_many([k for k, _ in key_pairs])

    out: Dict[str, Json] = {}
    fresh_needed: List[str] = []

    for cache_key, sym in key_pairs:
        hit = cached_map.get(cache_key)
        if hit is not None:
            out[sym] = hit
        else:
            fresh_needed.append(sym)

    if not fresh_needed:
        return out

    try:
        tq = Ticker(fresh_needed, asynchronous=False, formatted=False)
        raw = retry(lambda: tq.get_modules(["price", "summaryDetail", "financialData", "defaultKeyStatistics"])) or {}

        write_back: Dict[str, Json] = {}

        for sym in fresh_needed:
            try:
                node = _get_symbol_node(raw, sym)
                payload = _extract_stock_payload(
                    sym,
                    (node.get("price") or {}),
                    (node.get("summaryDetail") or {}),
                    (node.get("financialData") or {}),
                    (node.get("defaultKeyStatistics") or {}),
                )
                out[sym] = payload
                write_back[_ck(sym, "quote")] = payload
            except Exception as e:
                out[sym] = {
                    "status": "error",
                    "symbol": sym,
                    "error_code": "YQ_MANY_PARSE_FAILED",
                    "message": str(e),
                }

        if write_back:
            cache_set_many(write_back, ttl_seconds=TTL_QUOTES_SEC)

        return out

    except Exception as e:
        return {
            s: {"status": "error", "symbol": s, "error_code": "YQ_MANY_FETCH_FAILED", "message": str(e)}
            for s in fresh_needed
        }
