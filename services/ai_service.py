from langchain_openai import ChatOpenAI
from schemas.holding import HoldingInput
from services.helpers.ai.analyze_holding import analyze_holding
from schemas.ai_analysis import AnalysisOutput
from services.helpers.ai.analyze_portfolio import analyze_portfolio
from services.helpers.ai.analyze_portfolio_linkup import analyze_portfolio_linkup
from services.helpers.ai.analyze_holding_linkup import analyze_holding_linkup, analyze_holding_only_linkup
from services.helpers.ai.ai_forcaster import build_forecaster_payload, run_forecaster_analysis
from services.helpers.ai.forcast_post import normalize_forecast
from services.yahoo_service import get_full_stock_data
from services.helpers.ai.analyze_holding_perplexity import analyze_investment_holding_pplx

from typing import Any, Dict
import json

llm = ChatOpenAI(
    temperature=0.0,
    model="gpt-4o-mini",
    max_completion_tokens=1100,
    timeout=20,
    model_kwargs={"response_format": {"type": "json_object"}},
)

# ---------- Core ----------
def analyze_investment_holding(holding: HoldingInput) -> Dict[str, Any]:
    # return analyze_holding(holding, llm)
    return analyze_holding_only_linkup(holding)

def analyze_investment_portfolio(holdings: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Run a portfolio-level analysis on all holdings.
    Returns a dict (parsed JSON) with rating, rationale, exposures, etc.
    """
    analysis = analyze_portfolio_linkup(holdings)
    return analysis
    # raw = analyze_portfolio(holdings, llm)
    # try:
    #     return json.loads(raw)
    # except Exception:
    #     return {"error": "Could not parse portfolio analysis output", "raw": raw}

def analyze_investment_holding_forecast(holding: HoldingInput) -> Dict[str, Any]:
    symbol = holding.get("symbol","")
    if not symbol:
        return {"error": "Missing symbol"}
    yahooData = get_full_stock_data(symbol)
    linkupData = analyze_holding_only_linkup(holding)
    payload = build_forecaster_payload(symbol, yahooData, linkupData.get("answer"), linkupData.get("items"))

    # forecast = run_forecaster_analysis(llm, payload)
    # print(forecast)

    raw = run_forecaster_analysis(llm, payload)
    analysis = normalize_forecast(raw, payload["news"]["citations"], payload["yahoo"], payload.get("position"))
    return analysis
    # return forecast

def analyze_investment_holding_perplexity(holding: HoldingInput) -> Dict[str, Any]:
    return analyze_investment_holding_pplx(holding)