# services/ai/portfolio/portfolio_service.py
# (task runner + TTL constants, matches your symbol task pattern)

import os
import time
import logging
from sqlalchemy.orm import Session

from services.cache.cache_backend import cache_set
from services.finnhub.finnhub_service import FinnhubService

from services.ai.portfolio.types import ClassifyConfig
from services.ai.portfolio.holding_helper import classify_holdings
from services.ai.portfolio.facts_pack import build_portfolio_facts_pack
from services.ai.portfolio.portfolio_prompt import run_portfolio_core_llm

# IMPORTANT: import your existing holdings service
from services.holding_service import get_holdings_with_live_prices  # adjust path

logger = logging.getLogger("analysis_timing")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

TTL_PORTFOLIO_TASK_RESULT_SEC = int(os.getenv("TTL_PORTFOLIO_TASK_RESULT_SEC", "3600"))
TTL_PORTFOLIO_REPORT_SEC = int(os.getenv("TTL_PORTFOLIO_REPORT_SEC", "1800"))


def _ck_task(task_id: str) -> str:
    return f"PORTFOLIO:ANALYZE:TASK:{(task_id or '').strip()}"


def _ck_portfolio_report(user_id: int | str, currency: str) -> str:
    return f"PORTFOLIO:REPORT:{user_id}:{(currency or 'USD').strip().upper()}"


async def run_portfolio_analysis_task(
    *,
    user_id: str,
    task_id: str,
    currency: str,
    force: bool,
    db: Session,
    finnhub: FinnhubService,
):
    """
    Redis-backed background task runner.
    Stores progress under PORTFOLIO:ANALYZE:TASK:<task_id>
    Optionally caches final report under PORTFOLIO:REPORT:<user_id>:<currency>
    """
    t0_total = time.perf_counter()
    task_key = _ck_task(task_id)
    curr = (currency or "USD").strip().upper()

    cache_set(task_key, {"status": "processing", "data": None}, ttl_seconds=TTL_PORTFOLIO_TASK_RESULT_SEC)

    try:
        # 1) Fetch holdings in requested currency
        payload = await get_holdings_with_live_prices(
            user_id=str(user_id),
            db=db,
            finnhub=finnhub,
            currency=curr,
            top_only=False,
            include_weights=True,
        )
        holdings = payload.get("items", []) or []

        # 2) Classify holdings (core / risk / satellite)
        classified = classify_holdings(
            holdings,
            market_value=payload.get("market_value"),
            config=ClassifyConfig(
                top_n_core=5,
                core_weight_pct=10.0,
                big_loser_pct=-15.0,
                tiny_weight_pct=1.0,
                risk_types=("crypto",),
            ),
        )

        # 3) Build portfolio facts pack (compact)
        facts_pack = build_portfolio_facts_pack(
            holdings=holdings,
            classified_items=classified["items"],
            groups=classified["groups"],
            totals_payload=payload,
        )

        # 4) LLM reasoning (structured)
        core = await run_portfolio_core_llm(facts_pack.model_dump())

        # 5) Assemble final report
        report = {
            "user_id": user_id,
            "currency": curr,
            "as_of": payload.get("as_of"),
            "market_value": payload.get("market_value"),
            "classification_summary": classified.get("summary", {}),
            "facts_pack": facts_pack.model_dump(),
            "core_analysis": core,
        }

        total_dt = time.perf_counter() - t0_total
        payload_out = {
            "status": "complete",
            "data": {"report": report, "total_seconds": round(total_dt, 3)},
        }

        cache_set(task_key, payload_out, ttl_seconds=TTL_PORTFOLIO_TASK_RESULT_SEC)

        # Cache the latest report for quick reads
        try:
            cache_set(_ck_portfolio_report(user_id, curr), report, ttl_seconds=TTL_PORTFOLIO_REPORT_SEC)
        except Exception:
            pass

        logger.info("[%s] portfolio user=%s %s: total=%.2fs", task_id, user_id, curr, total_dt)

    except Exception as e:
        total_dt = time.perf_counter() - t0_total
        logger.exception("Portfolio analysis failed for user=%s (%s).", user_id, task_id)
        payload_out = {"status": "failed", "data": {"error": str(e), "total_seconds": round(total_dt, 3)}}
        cache_set(task_key, payload_out, ttl_seconds=TTL_PORTFOLIO_TASK_RESULT_SEC)
