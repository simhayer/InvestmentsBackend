# finnhub_analyst.py
from __future__ import annotations

"""
Analyst data endpoints for Finnhub: recommendations and price targets.
Add these methods to your FinnhubService class, or use standalone.
"""
import logging
import os

logger = logging.getLogger(__name__)
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from services.cache.cache_backend import cache_get, cache_set
import httpx

TTL_ANALYST_SEC = 3600  # 1 hour - analyst data doesn't change frequently
TTL_EMPTY_SEC = 300     # 5 min for empty results


def _ck_reco(symbol: str) -> str:
    return f"FINNHUB:RECO:{(symbol or '').strip().upper()}"


def _ck_target(symbol: str) -> str:
    return f"FINNHUB:TARGET:{(symbol or '').strip().upper()}"


@dataclass(frozen=True)
class AnalystRecommendation:
    """Aggregated analyst recommendation for a given period."""
    period: str  # e.g., "2024-01-01"
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int

    @property
    def total(self) -> int:
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell

    @property
    def consensus(self) -> str:
        """Returns consensus label based on weighted score."""
        if self.total == 0:
            return "N/A"
        # Score: Strong Buy=5, Buy=4, Hold=3, Sell=2, Strong Sell=1
        score = (
            self.strong_buy * 5 +
            self.buy * 4 +
            self.hold * 3 +
            self.sell * 2 +
            self.strong_sell * 1
        ) / self.total

        if score >= 4.5:
            return "Strong Buy"
        elif score >= 3.5:
            return "Buy"
        elif score >= 2.5:
            return "Hold"
        elif score >= 1.5:
            return "Sell"
        return "Strong Sell"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "strongBuy": self.strong_buy,
            "buy": self.buy,
            "hold": self.hold,
            "sell": self.sell,
            "strongSell": self.strong_sell,
            "total": self.total,
            "consensus": self.consensus,
        }


@dataclass(frozen=True)
class PriceTarget:
    """Analyst price target summary."""
    target_high: Optional[float]
    target_low: Optional[float]
    target_mean: Optional[float]
    target_median: Optional[float]
    last_updated: Optional[str]

    def upside_pct(self, current_price: float) -> Optional[float]:
        """Calculate upside to mean target."""
        if self.target_mean and current_price and current_price > 0:
            return ((self.target_mean - current_price) / current_price) * 100
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "targetHigh": self.target_high,
            "targetLow": self.target_low,
            "targetMean": self.target_mean,
            "targetMedian": self.target_median,
            "lastUpdated": self.last_updated,
        }


@dataclass(frozen=True)
class AnalystData:
    """Combined analyst data for a symbol."""
    symbol: str
    recommendations: List[AnalystRecommendation]
    price_target: Optional[PriceTarget]
    gaps: List[str]

    @property
    def latest_recommendation(self) -> Optional[AnalystRecommendation]:
        return self.recommendations[0] if self.recommendations else None

    def to_dict(self) -> Dict[str, Any]:
        latest = self.latest_recommendation
        return {
            "symbol": self.symbol,
            "latestRecommendation": latest.to_dict() if latest else None,
            "recommendationHistory": [r.to_dict() for r in self.recommendations],
            "priceTarget": self.price_target.to_dict() if self.price_target else None,
            "gaps": self.gaps,
        }


class FinnhubAnalystService:
    """
    Fetches analyst recommendations and price targets from Finnhub.
    
    Can be used standalone or integrated into your main FinnhubService.
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 5.0,
    ):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing FINNHUB_API_KEY")
        self.timeout = timeout

    def _auth_params(self, **params: Any) -> Dict[str, Any]:
        return {**params, "token": self.api_key}

    async def fetch_recommendations(
        self,
        symbol: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> List[AnalystRecommendation]:
        """
        Fetch analyst recommendation trends.
        Returns list sorted by period (most recent first).
        
        Finnhub endpoint: /stock/recommendation
        """
        sym = (symbol or "").strip().upper()
        if not sym:
            return []

        should_close = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)

        try:
            r = await client.get(
                f"{self.BASE_URL}/stock/recommendation",
                params=self._auth_params(symbol=sym),
            )
            r.raise_for_status()
            data = r.json()

            if not isinstance(data, list):
                return []

            recommendations = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                recommendations.append(AnalystRecommendation(
                    period=item.get("period", ""),
                    strong_buy=int(item.get("strongBuy", 0) or 0),
                    buy=int(item.get("buy", 0) or 0),
                    hold=int(item.get("hold", 0) or 0),
                    sell=int(item.get("sell", 0) or 0),
                    strong_sell=int(item.get("strongSell", 0) or 0),
                ))

            # Sort by period descending (most recent first)
            recommendations.sort(key=lambda x: x.period, reverse=True)
            return recommendations

        except Exception as e:
            logger.exception("fetch_recommendations failed for %s", sym)
            return []
        finally:
            if should_close:
                await client.aclose()

    async def fetch_price_target(
        self,
        symbol: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Optional[PriceTarget]:
        """
        Fetch analyst price target consensus.
        
        Finnhub endpoint: /stock/price-target
        """
        sym = (symbol or "").strip().upper()
        if not sym:
            return None

        should_close = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)

        try:
            r = await client.get(
                f"{self.BASE_URL}/stock/price-target",
                params=self._auth_params(symbol=sym),
            )
            r.raise_for_status()
            data = r.json()

            if not isinstance(data, dict):
                return None

            # Check if we have any meaningful data
            if not any(data.get(k) for k in ["targetHigh", "targetLow", "targetMean", "targetMedian"]):
                return None

            return PriceTarget(
                target_high=float(data["targetHigh"]) if data.get("targetHigh") else None,
                target_low=float(data["targetLow"]) if data.get("targetLow") else None,
                target_mean=float(data["targetMean"]) if data.get("targetMean") else None,
                target_median=float(data["targetMedian"]) if data.get("targetMedian") else None,
                last_updated=data.get("lastUpdated"),
            )

        except Exception as e:
            logger.exception("fetch_price_target failed for %s", sym)
            return None
        finally:
            if should_close:
                await client.aclose()

    async def fetch_analyst_data(
        self,
        symbol: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> AnalystData:
        """
        Fetch both recommendations and price targets in parallel.
        """
        sym = (symbol or "").strip().upper()
        if not sym:
            return AnalystData(symbol="", recommendations=[], price_target=None, gaps=["Missing symbol"])

        should_close = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout)

        gaps: List[str] = []

        try:
            reco_task = self.fetch_recommendations(sym, client=client)
            target_task = self.fetch_price_target(sym, client=client)

            recommendations, price_target = await asyncio.gather(
                reco_task, target_task, return_exceptions=True
            )

            if isinstance(recommendations, Exception):
                gaps.append("Recommendations unavailable")
                recommendations = []

            if isinstance(price_target, Exception):
                gaps.append("Price target unavailable")
                price_target = None

            if not recommendations:
                gaps.append("No analyst recommendations found")
            if not price_target:
                gaps.append("No price target data found")

            return AnalystData(
                symbol=sym,
                recommendations=recommendations if isinstance(recommendations, list) else [],
                price_target=price_target if isinstance(price_target, PriceTarget) else None,
                gaps=gaps,
            )

        finally:
            if should_close:
                await client.aclose()


# Cached versions

async def fetch_recommendations_cached(
    symbol: str,
    *,
    timeout: float = 5.0,
    ttl_seconds: int = TTL_ANALYST_SEC,
) -> List[Dict[str, Any]]:
    sym = (symbol or "").strip().upper()
    if not sym:
        return []

    key = _ck_reco(sym)
    cached = cache_get(key)
    if isinstance(cached, list):
        return cached

    svc = FinnhubAnalystService(timeout=timeout)
    recommendations = await svc.fetch_recommendations(sym)
    result = [r.to_dict() for r in recommendations]

    ttl = ttl_seconds if result else TTL_EMPTY_SEC
    cache_set(key, result, ttl_seconds=ttl)
    return result


async def fetch_price_target_cached(
    symbol: str,
    *,
    timeout: float = 5.0,
    ttl_seconds: int = TTL_ANALYST_SEC,
) -> Optional[Dict[str, Any]]:
    sym = (symbol or "").strip().upper()
    if not sym:
        return None

    key = _ck_target(sym)
    cached = cache_get(key)
    if isinstance(cached, dict):
        return cached

    svc = FinnhubAnalystService(timeout=timeout)
    target = await svc.fetch_price_target(sym)
    
    if target is None:
        cache_set(key, {}, ttl_seconds=TTL_EMPTY_SEC)
        return None

    result = target.to_dict()
    cache_set(key, result, ttl_seconds=ttl_seconds)
    return result


async def fetch_analyst_data_cached(
    symbol: str,
    *,
    timeout: float = 5.0,
    ttl_seconds: int = TTL_ANALYST_SEC,
) -> Dict[str, Any]:
    """
    Fetch analyst recommendations with caching.
    Note: Price target endpoint is premium - use Yahoo instead.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return {"symbol": "", "gaps": ["Missing symbol"]}

    # Only fetch recommendations (price target is premium on Finnhub)
    recommendations = await fetch_recommendations_cached(sym, timeout=timeout, ttl_seconds=ttl_seconds)

    gaps = []
    if not recommendations:
        gaps.append("No analyst recommendations found")

    return {
        "symbol": sym,
        "latestRecommendation": recommendations[0] if recommendations else None,
        "recommendationHistory": recommendations,
        "priceTarget": None,  # Use Yahoo for this instead
        "gaps": gaps,
    }