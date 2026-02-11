# services/yahoo_service.py
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from typing import Any, Dict, List, Optional
from yahooquery import Ticker
import pandas as pd
from utils.common_helpers import safe_float as _fnum, retry
from services.helpers.yahoo_helpers import pct, dist_pct, iso_utc_from_ts, to_epoch_utc, date_iso, parse_weird_cal_dt, quarter_label

Number = Optional[float]
Json = Dict[str, Any]

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

    # Quote time
    quote_ts = (
        price.get("regularMarketTime")
        or price.get("postMarketTime")
        or price.get("preMarketTime")
    )
    quote_time_utc = iso_utc_from_ts(quote_ts)

    # Computed deltas
    day_change = (current - previous) if (current is not None and previous is not None) else None
    day_change_pct = pct(current, previous)
    dist_high_pct = dist_pct(current, high_52)
    dist_low_pct = dist_pct(current, low_52)

    # Quality check
    is_stale = False
    if quote_time_utc:
        try:
            qt = datetime.fromisoformat(quote_time_utc.replace("Z", "+00:00"))
            is_stale = (datetime.now(timezone.utc) - qt) > timedelta(days=2)
        except Exception: pass

    missing = [k for k, v in {"price": current, "currency": currency}.items() if v is None]

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

def get_full_stock_data(symbol: str) -> Json:
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"status": "error", "error_code": "EMPTY_SYMBOL", "message": "Symbol required"}

    try:
        tq = Ticker(sym, asynchronous=False, formatted=False)
        raw = retry(lambda: tq.get_modules(["price", "summaryDetail", "financialData", "defaultKeyStatistics"]))
        node = _get_symbol_node(raw, sym)

        if not node or "price" not in node:
            raise ValueError(f"No data found for {sym}")

        payload = _extract_stock_payload(
            sym,
            node.get("price", {}),
            node.get("summaryDetail", {}),
            node.get("financialData", {}),
            node.get("defaultKeyStatistics", {})
        )

        return payload

    except Exception as e:
        return {"status": "error", "error_code": "YAHOO_FETCH_ERROR", "message": str(e)}


def get_price_history(symbol: str, period: str = "1y", interval: str = "1d") -> Json:
    sym = (symbol or "").upper().strip()
    try:
        tq = Ticker(sym, asynchronous=False, formatted=False)
        df = tq.history(period=period, interval=interval)

        if df is None or (hasattr(df, "empty") and df.empty):

            return {"status": "ok", "symbol": sym, "points": []}

        # Handle MultiIndex and flatten
        df = df.reset_index()
        if "symbol" in df.columns:
            df = df[df["symbol"].str.upper() == sym]
        
        # Rename 'index' or 'asOfDate' to 'date' if necessary
        date_col = next((c for c in df.columns if c in ["date", "index", "asOfDate"]), None)
        if date_col and date_col != "date":
            df = df.rename(columns={date_col: "date"})

        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        points = []
        for row in df.itertuples():

            points.append({
                "t": int(row.date.timestamp()) if isinstance(row.date, pd.Timestamp) else to_epoch_utc(row.date),
                "o": _fnum(getattr(row, "open", None)),
                "h": _fnum(getattr(row, "high", None)),
                "l": _fnum(getattr(row, "low", None)),
                "c": _fnum(getattr(row, "close", None)),
                "v": _fnum(getattr(row, "volume", None)),
                "adjclose": _fnum(getattr(row, "adjclose", None)),
            })
        return {"status": "ok", "symbol": sym, "points": points}
    except Exception as e:
        return {"status": "error", "message": str(e)}  

def get_financials(symbol: str, period: str = "annual") -> Json:
    """Uses all_financial_data for high-density statement building."""
    sym = (symbol or "").upper().strip()
    want_q = (period == "quarterly")
    # Yahoo uses different labels for periods
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
            df = df[df["symbol"].str.upper() == sym]
        
        date_col = "asOfDate" if "asOfDate" in df.columns else "endDate"
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        
        if "periodType" in df.columns:
            df = df[df["periodType"].isin(acceptable)]


        df = df.sort_values(date_col, ascending=False)


        def _get(r, *keys):
            for k in keys:
                if k in r and pd.notna(r[k]): return _fnum(r[k])

            return None

        income, balance, cash = [], [], []

        for _, r in df.iterrows():
            iso = r[date_col].date().isoformat()

            # --- Income statement ---
            income.append({
                "date": iso,
                "revenue": _get(r, "TotalRevenue", "OperatingRevenue", "Revenue"),
                # this is what you meant by gross_income
                "gross_profit": _get(r, "GrossProfit"),
                "operating_income": _get(r, "OperatingIncome", "TotalOperatingIncomeAsReported"),
                "net_income": _get(r, "NetIncome", "NetIncomeCommonStockholders", "NetIncomeIncludingNoncontrollingInterests"),
                "eps": _get(r, "DilutedEPS", "BasicEPS", "EPS"),
            })

            # --- Balance sheet ---
            balance.append({
                "date": iso,
                "total_assets": _get(r, "TotalAssets"),
                "total_liabilities": _get(r, "TotalLiabilitiesNetMinorityInterest", "TotalLiabilities"),
                "total_equity": _get(r, "StockholdersEquity", "CommonStockEquity", "TotalEquityGrossMinorityInterest"),
                "cash": _get(r, "CashAndCashEquivalents", "CashAndCashEquivalentsAtCarryingValue", "Cash"),
                "inventory": _get(r, "Inventory"),
                "long_term_debt": _get(r, "LongTermDebt", "LongTermDebtAndCapitalLeaseObligation"),
            })

            # --- Cash flow ---
            ocf = _get(r, "OperatingCashFlow")
            icf = _get(r, "InvestingCashFlow")
            fcf = _get(r, "FreeCashFlow")

            # compute FCF if missing: OCF - capex (capex often negative)
            if fcf is None:
                capex = _get(r, "CapitalExpenditure", "CapitalExpenditures")
                if ocf is not None and capex is not None:
                    fcf = ocf - capex  # if capex is negative, subtracting it adds its magnitude
                    # if you find capex is positive in your data, use: fcf = ocf - abs(capex)

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
            "income_statement": income, 
            "balance_sheet": balance, 
            "cash_flow": cash
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def _financials_df_fallback(sym: str, period: str) -> Json:
    """
    Previous DF-based implementation; called if all_financial_data()
    is unavailable or too sparse. Uses .income_statement/.balance_sheet/.cash_flow
    and returns the same normalized shape.
    """
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
                    return _fnum(row[k])
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
        print("Error in _financials_df_fallback:", str(e))
        return {"status": "error", "error_code": "YQ_FINANCIALS_FALLBACK_FAILED", "message": str(e)}

# ---------- NEW: earnings ----------
def get_earnings(symbol: str) -> Json:
    sym = (symbol or "").upper().strip()
    try:
        tq = Ticker(sym, asynchronous=False, formatted=False)
        res = tq.get_modules(["calendarEvents", "earningsHistory"])
        node = _get_symbol_node(res, sym)

        cal = node.get("calendarEvents", {}).get("earnings", {})
        nxt_date = parse_weird_cal_dt(cal.get("earningsDate"))

        history = []
        hist_raw = node.get("earningsHistory", {}).get("history", [])
        for r in hist_raw:
            d = date_iso(r.get("reportDate") or r.get("quarter"))
            if d:
                history.append({
                    "date": d,
                    "period": quarter_label(d),
                    "actual": _fnum(r.get("epsActual")),
                    "estimate": _fnum(r.get("epsEstimate")),
                    "surprisePct": _fnum(r.get("surprisePercent"))
                })

        return {
            "status": "ok",
            "symbol": sym,
            "next_earnings_date": nxt_date,
            "history": sorted(history, key=lambda x: x["date"], reverse=True)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
        recommendation = _fnum(fin.get("recommendationMean"))
        recommendation_key = rec.get("trend", [{}])[0].get("recommendationKey") if isinstance(rec.get("trend"), list) and rec.get("trend") else fin.get("recommendationKey")
        if not recommendation_key:
            recommendation_key = fin.get("recommendationKey")

        # price targets
        price_target_low = _fnum(fin.get("targetLowPrice") or fin.get("priceTargetLow"))
        price_target_mean = _fnum(fin.get("targetMeanPrice") or fin.get("priceTargetMean"))
        price_target_high = _fnum(fin.get("targetHighPrice") or fin.get("priceTargetHigh"))

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
    if isinstance(res, dict):
        # preferred: nested by symbol
        sub = res.get(sym)
        if isinstance(sub, dict):
            return sub
        # sometimes it's already the node (modules at top level)
        return res if any(isinstance(v, (dict, list)) for v in res.values()) else {}
    return {}