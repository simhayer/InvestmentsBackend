# services/finnhub/peer_benchmark_service.py
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from services.cache.cache_backend import cache_get, cache_set
from services.finnhub.client import FINNHUB_CLIENT
from services.finnhub.finnhub_service import FinnhubService, FinnhubServiceError
from services.finnhub.finnhub_fundamentals import fetch_fundamentals_cached
from utils.common_helpers import median, safe_float, fmt_pct, to_float

logger = logging.getLogger(__name__)

TTL_PEERS_LIST_SEC = 7 * 24 * 3600
TTL_PEER_BENCH_SEC = 24 * 3600


def _ck_peers(symbol: str, grouping: str) -> str:
    sym = (symbol or "").strip().upper()
    grp = (grouping or "").strip()
    return f"PEERS:{sym}:{grp}"


def _ck_peer_bench(symbol: str, grouping: str) -> str:
    sym = (symbol or "").strip().upper()
    grp = (grouping or "").strip()
    return f"PEER_BENCH:{sym}:{grp}"


@dataclass(frozen=True)
class PeerBenchmarkResult:
    data: Dict[str, Any]
    gaps: List[str]


def _percentile_rank(
    company_val: Optional[float],
    peer_vals: List[float],
    *,
    higher_is_better: bool,
) -> Optional[float]:
    """
    Returns percentile 0..100 where 100 = best relative to peers.
    Simple and explainable: % of peers the company beats (or is cheaper than).
    """
    if company_val is None:
        return None

    vals = [v for v in peer_vals if isinstance(v, (int, float))]
    if not vals:
        return None

    if higher_is_better:
        better_or_equal = sum(1 for v in vals if company_val >= v)
    else:
        better_or_equal = sum(1 for v in vals if company_val <= v)

    return 100.0 * better_or_equal / max(1, len(vals))


def _avg(nums: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in nums if isinstance(x, (int, float))]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _extract_norm(snapshot: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Uses fundamentals_service normalized block.
    Keep this small and stable; add more later as needed.
    """
    n = snapshot.get("normalized") or {}
    return {
        "market_cap": safe_float(n.get("market_cap")),
        "pe_ttm": safe_float(n.get("pe_ttm")),
        "revenue_growth_yoy": safe_float(n.get("revenue_growth_yoy")),
        "gross_margin": safe_float(n.get("gross_margin")),
        "operating_margin": safe_float(n.get("operating_margin")),
        "free_cash_flow": safe_float(n.get("free_cash_flow")),
        "debt_to_equity": safe_float(n.get("debt_to_equity")),
    }


def _snapshot_meta(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    prof = snapshot.get("profile") or {}
    return {
        "name": prof.get("name") or prof.get("ticker"),
        "industry": prof.get("finnhubIndustry") or prof.get("industry"),
        "sector": prof.get("sector") or None,
        "country": prof.get("country"),
    }


async def fetch_peers_list_cached(
    symbol: str,
    *,
    timeout_s: float = 5.0,
    ttl_seconds: int = TTL_PEERS_LIST_SEC,
    grouping: str = "industry",
) -> List[str]:
    """
    Fetch peers list and cache it.
    Uses grouping='industry' by default.
    """
    sym = (symbol or "").strip().upper()
    grp = (grouping or "industry").strip()
    if not sym:
        return []

    key = _ck_peers(sym, grp)
    cached = cache_get(key)
    if isinstance(cached, list) and all(isinstance(x, str) for x in cached):
        return cached

    try:
        svc = FinnhubService(timeout=timeout_s)
    except FinnhubServiceError:
        return []

    peers: List[str] = []
    try:
        peers = await svc.fetch_peers(sym, grouping=grp, client=FINNHUB_CLIENT)
        if not isinstance(peers, list):
            peers = []
    except Exception as e:
        logger.info("fetch_peers failed (%s, grouping=%s): %s", sym, grp, e)
        peers = []

    # normalize + remove self + dedupe (preserve order)
    peers = [p.strip().upper() for p in peers if isinstance(p, str) and p.strip()]
    out: List[str] = []
    seen: set[str] = set()
    for p in peers:
        if p == sym:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)

    cache_set(key, out, ttl_seconds=ttl_seconds)
    return out


def _filter_peers_soft(
    company_norm: Dict[str, Optional[float]],
    peers: List[Tuple[str, Dict[str, Optional[float]], Dict[str, Any]]],
) -> List[Tuple[str, Dict[str, Optional[float]], Dict[str, Any]]]:
    """
    Soft peer filtering:
    - require market cap (to keep percentile math meaningful)
    - rank by size similarity to company
    """
    c_mc = company_norm.get("market_cap") or 0.0
    if not c_mc:
        # if missing, just keep peers with market cap
        return [(s, n, m) for (s, n, m) in peers if n.get("market_cap")]

    def size_score(p_mc: float) -> float:
        # Higher is better. Use log-ratio closeness.
        # score ~ 0 when far, closer to 1 when similar.
        try:
            ratio = p_mc / c_mc
            if ratio <= 0:
                return 0.0
            # clamp extremes to reduce weirdness
            if ratio < 1e-6:
                ratio = 1e-6
            if ratio > 1e6:
                ratio = 1e6
            # log distance
            import math
            dist = abs(math.log(ratio))
            return 1.0 / (1.0 + dist)
        except Exception:
            return 0.0

    scored: List[Tuple[float, str, Dict[str, Optional[float]], Dict[str, Any]]] = []
    for sym, norm, meta in peers:
        p_mc = norm.get("market_cap")
        if not p_mc:
            continue
        scored.append((size_score(float(p_mc)), sym, norm, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [(sym, norm, meta) for _, sym, norm, meta in scored]


def _needs_sector_fallback(
    company_norm: Dict[str, Optional[float]],
    peer_rows: List[Tuple[str, Dict[str, Optional[float]], Dict[str, Any]]],
) -> bool:
    """
    Decide whether industry peers are too noisy or too small.
    """
    c_mc = company_norm.get("market_cap") or 0.0
    if not c_mc:
        return True

    if len(peer_rows) < 6:
        return True

    # Too many tiny peers => noisy set
    tiny = 0
    total = 0
    for _, norm, _ in peer_rows:
        p_mc = norm.get("market_cap")
        if not p_mc:
            continue
        total += 1
        if p_mc < max(20_000, 0.01 * c_mc):  # < $20B OR <1% of company
            tiny += 1

    if total == 0:
        return True

    return (tiny / total) > 0.30


def _build_summary(benchmarks: Dict[str, Any]) -> List[str]:
    """
    Deterministic (non-LLM) summary bullets to stabilize outputs.
    """
    def pct_str(x: Optional[float]) -> Optional[str]:
        if x is None:
            return None
        return str(int(round(x)))

    out: List[str] = []

    pe_pct = pct_str((benchmarks.get("pe_ttm") or {}).get("company_percentile"))
    if pe_pct is not None:
        out.append(f"Valuation: P/E is cheaper than about {pe_pct}% of peers.")

    opm_pct = pct_str((benchmarks.get("operating_margin") or {}).get("company_percentile"))
    if opm_pct is not None:
        out.append(f"Profitability: operating margin ranks around the {opm_pct}th percentile vs peers.")

    revg_pct = pct_str((benchmarks.get("revenue_growth_yoy") or {}).get("company_percentile"))
    if revg_pct is not None:
        out.append(f"Growth: revenue growth ranks around the {revg_pct}th percentile vs peers.")

    dte_pct = pct_str((benchmarks.get("debt_to_equity") or {}).get("company_percentile"))
    if dte_pct is not None:
        out.append(f"Balance sheet: leverage (debt-to-equity) looks better than about {dte_pct}% of peers.")

    return out[:4]


async def _build_peer_rows(
    peers: List[str],
    *,
    timeout_s: float,
) -> List[Tuple[str, Dict[str, Optional[float]], Dict[str, Any]]]:
    """
    Fetch peer fundamentals concurrently and transform into (sym, norm, meta).
    """
    tasks = [fetch_fundamentals_cached(p, timeout_s=timeout_s) for p in peers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    peer_rows: List[Tuple[str, Dict[str, Optional[float]], Dict[str, Any]]] = []
    for p, r in zip(peers, results):
        if isinstance(r, BaseException):
            continue
        data = getattr(r, "data", None)
        if not isinstance(data, dict) or not data:
            continue
        peer_rows.append((p, _extract_norm(data), _snapshot_meta(data)))

    return peer_rows


async def fetch_peer_benchmark_cached(
    symbol: str,
    *,
    timeout_s: float = 5.0,
    ttl_seconds: int = TTL_PEER_BENCH_SEC,
    max_peers: int = 20,
    peers_grouping: str = "industry",
) -> PeerBenchmarkResult:
    sym = (symbol or "").strip().upper()
    if not sym:
        return PeerBenchmarkResult({}, ["Missing symbol for peer benchmark"])

    # 1) company snapshot (needed to decide grouping strategy)
    company_res = await fetch_fundamentals_cached(sym, timeout_s=timeout_s)
    if not getattr(company_res, "data", None):
        return PeerBenchmarkResult({}, ["Company fundamentals unavailable"])

    company_norm = _extract_norm(company_res.data)
    company_meta = _snapshot_meta(company_res.data)
    c_mc = company_norm.get("market_cap") or 0.0

    # Mega-cap override: default to sector peers for better comps
    effective_grouping = (peers_grouping or "industry").strip()
    if c_mc >= 500_000 and effective_grouping != "sector":
        effective_grouping = "sector"

    key = _ck_peer_bench(sym, effective_grouping)
    cached = cache_get(key)
    if isinstance(cached, dict) and "data" in cached and "gaps" in cached:
        data = cached.get("data") or {}
        gaps = cached.get("gaps") or []
        if isinstance(data, dict) and isinstance(gaps, list):
            return PeerBenchmarkResult(data=data, gaps=gaps)

    gaps: List[str] = []

    # 2) peers list (effective grouping)
    peers = await fetch_peers_list_cached(sym, timeout_s=timeout_s, grouping=effective_grouping)
    if not peers:
        gaps.append("Peers list unavailable")
        out = PeerBenchmarkResult(
            data={
                "symbol": sym,
                "company_meta": company_meta,
                "company": company_norm,
                "peers_used": [],
                "benchmarks": {},
                "scores": {},
                "summary": [],
                "meta": {"grouping": effective_grouping, "max_peers": max_peers},
            },
            gaps=gaps,
        )
        cache_set(key, {"data": out.data, "gaps": out.gaps}, ttl_seconds=ttl_seconds)
        return out

    peers = peers[: max_peers * 2]

    # 3) fetch peer rows
    peer_rows = await _build_peer_rows(peers, timeout_s=timeout_s)
    if not peer_rows:
        gaps.append("Peer fundamentals unavailable")
        out = PeerBenchmarkResult(
            data={
                "symbol": sym,
                "company_meta": company_meta,
                "company": company_norm,
                "peers_used": [],
                "benchmarks": {},
                "scores": {},
                "summary": [],
                "meta": {"grouping": effective_grouping, "max_peers": max_peers},
            },
            gaps=gaps,
        )
        cache_set(key, {"data": out.data, "gaps": out.gaps}, ttl_seconds=ttl_seconds)
        return out

    # 4) soft size ranking + cap
    peer_rows = _filter_peers_soft(company_norm, peer_rows)
    peer_rows = peer_rows[:max_peers]

    if len(peer_rows) < 6 and effective_grouping != "sector":
        # last resort fallback
        gaps.append("Peer set is small; fell back to sector peers")
        peers2 = await fetch_peers_list_cached(sym, timeout_s=timeout_s, grouping="sector")
        peers2 = peers2[: max_peers * 2]
        peer_rows2 = await _build_peer_rows(peers2, timeout_s=timeout_s)
        peer_rows2 = _filter_peers_soft(company_norm, peer_rows2)[:max_peers]
        if peer_rows2:
            peer_rows = peer_rows2
            effective_grouping = "sector"

    peers_used = [p for p, _, _ in peer_rows]

    if len(peers_used) < 5:
        gaps.append("Peer set is small after filtering")

    # 5) distributions
    def dist(key: str) -> List[float]:
        xs: List[float] = []
        for _, n, _ in peer_rows:
            v = n.get(key)
            if isinstance(v, (int, float)):
                xs.append(float(v))
        return xs

    metric_cfg: Dict[str, bool] = {
        "pe_ttm": False,
        "revenue_growth_yoy": True,
        "gross_margin": True,
        "operating_margin": True,
        "debt_to_equity": False,
    }

    benchmarks: Dict[str, Any] = {}
    for m, hib in metric_cfg.items():
        d = dist(m)
        benchmarks[m] = {
            "company": company_norm.get(m),
            "peer_median": median(d),
            "company_percentile": _percentile_rank(company_norm.get(m), d, higher_is_better=hib),
            "peer_count": len(d),
            "higher_is_better": hib,
        }

    # 6) scores (0–100)
    valuation_score = _avg([benchmarks["pe_ttm"]["company_percentile"]])
    growth_score = _avg([benchmarks["revenue_growth_yoy"]["company_percentile"]])
    quality_score = _avg([
        benchmarks["gross_margin"]["company_percentile"],
        benchmarks["operating_margin"]["company_percentile"],
    ])
    health_score = _avg([benchmarks["debt_to_equity"]["company_percentile"]])

    scores = {
        "valuation": valuation_score,
        "growth": growth_score,
        "quality": quality_score,
        "financial_health": health_score,
        "overall": _avg([valuation_score, growth_score, quality_score, health_score]),
    }

    data = {
        "symbol": sym,
        "company_meta": company_meta,
        "company": company_norm,
        "peers_used": peers_used,
        "benchmarks": benchmarks,
        "scores": scores,
        "summary": _build_summary(benchmarks),
        "meta": {"grouping": effective_grouping, "max_peers": max_peers},
    }

    out = PeerBenchmarkResult(data=data, gaps=gaps)
    cache_set(key, {"data": out.data, "gaps": out.gaps}, ttl_seconds=ttl_seconds)
    return out

def _line(label, company, median, pctile, higher_is_better):
    company_f = to_float(company)
    median_f = to_float(median)
    p = to_float(pctile)

    if company_f is None or median_f is None:
        return None

    # No percentile available → simple comparison
    if p is None:
        return f"{label}: {company_f:.2f} vs peer median {median_f:.2f}."

    p = max(0.0, min(100.0, p))

    if higher_is_better is False:
        # Your percentile already behaves like "better-than %" for lower-is-better metrics
        return (
            f"{label} (lower is better): {company_f:.2f} vs {median_f:.2f} "
            f"(better than ~{p:.0f}% of peers)."
        )

    return (
        f"{label}: {company_f:.2f} vs {median_f:.2f} "
        f"(better than ~{p:.0f}% of peers)."
    )


def build_peer_summary(pc):
    ks = (pc or {}).get("key_stats") or {}
    out = []

    pe = ks.get("pe_ttm") or {}
    s = _line("Valuation P/E", pe.get("company"), pe.get("peer_median"),
              pe.get("company_percentile"), pe.get("higher_is_better"))
    if s: out.append(s)

    g = ks.get("revenue_growth_yoy") or {}
    s = _line("Growth YoY", g.get("company"), g.get("peer_median"),
              g.get("company_percentile"), g.get("higher_is_better"))
    if s: out.append(s)

    om = ks.get("operating_margin") or {}
    s = _line("Operating margin", om.get("company"), om.get("peer_median"),
              om.get("company_percentile"), om.get("higher_is_better"))
    if s: out.append(s)

    d = ks.get("debt_to_equity") or {}
    s = _line("Leverage D/E", d.get("company"), d.get("peer_median"),
              d.get("company_percentile"), d.get("higher_is_better"))
    if s: out.append(s)

    return out[:4]

def build_peer_positioning(peer_ready: Dict[str, Any]) -> Dict[str, str]:
    ks = peer_ready.get("key_stats", {})
    out = {}

    pe = ks.get("pe_ttm", {})
    if pe:
        out["valuation"] = "cheaper than most peers" if pe.get("company") < pe.get("peer_median") else "more expensive than peers"

    g = ks.get("revenue_growth_yoy", {})
    if g:
        out["growth"] = "bottom-quintile growth vs peers" if g.get("company_percentile", 50) < 25 else "above-average growth"

    om = ks.get("operating_margin", {})
    if om:
        out["profitability"] = "above-median margins" if om.get("company") > om.get("peer_median") else "below-median margins"

    return out


def build_peer_comparison_ready(peer_benchmark: Dict[str, Any]) -> Dict[str, Any]:
    peer_benchmark = peer_benchmark or {}
    bench = peer_benchmark.get("benchmarks") or {}

    key_stats = {}
    for k in ("pe_ttm", "revenue_growth_yoy", "operating_margin", "debt_to_equity", "gross_margin"):
        if k in bench and bench.get(k) is not None:
            key_stats[k] = bench.get(k)

    return {
        "peers_used": (peer_benchmark.get("peers_used") or [])[:12],
        "scores": peer_benchmark.get("scores") or {},
        "key_stats": key_stats,
    }