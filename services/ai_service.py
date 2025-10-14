from langchain_openai import ChatOpenAI
from schemas.holding import HoldingInput
from services.helpers.ai.analyze_holding import analyze_holding
from schemas.ai_analysis import AnalysisOutput
from services.helpers.ai.analyze_portfolio import analyze_portfolio
from services.helpers.ai.analyze_portfolio_linkup import analyze_portfolio_linkup
from services.helpers.ai.analyze_holding_linkup import analyze_holding_linkup, analyze_holding_only_linkup
from services.yahoo_service import get_full_stock_data
from services.helpers.ai.analyze_holding_perplexity import analyze_investment_symbol_pplx
from services.helpers.ai.analyze_portfolio_perplexity import analyze_portfolio_pplx
from services.helpers.ai.risk import compute_risk
from services.helpers.ai.riskai import generate_ai_risk_summary
from typing import Any, Dict
from models.holding import Holding

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

def analyze_portfolio_perplexity(holdings: list[Holding]) -> dict[str, Any]:
    result = analyze_portfolio_pplx(holdings)
    risk = compute_risk(result["summary"], result["positions"])
    risk["ai"] = generate_ai_risk_summary(risk, result["summary"], result["positions"])
    result["risk"] = risk

    return result

def analyze_investment_symbol_perplexity(symbol: str) -> Dict[str, Any]:
    # holding = {"symbol": symbol}
    return analyze_investment_symbol_pplx(symbol)