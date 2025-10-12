import os, datetime as dt
from typing import Any, Dict, Optional, List, cast, TypedDict
import httpx

# ---- Config ----
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")
LINKUP_API_URL = os.getenv("LINKUP_API_URL", "https://api.linkup.so/v1/search")

SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "8"))


LINKUP_DEPTH = os.getenv("LINKUP_DEPTH", "standard")
LINKUP_OUTPUT = os.getenv("LINKUP_OUTPUT", "sourcedAnswer")  # default to match docs example
INCLUDE_DOMAINS = [d.strip() for d in os.getenv("LINKUP_INCLUDE_DOMAINS", "").split(",") if d.strip()]
EXCLUDE_DOMAINS = [d.strip() for d in os.getenv("LINKUP_EXCLUDE_DOMAINS", "").split(",") if d.strip()]

class LinkupItem(TypedDict):
    title: str
    url: str
    snippet: Optional[str]
    published_at: Optional[str]

class LinkupResult(TypedDict):
    items: List[LinkupItem]
    answer: Optional[str]
    raw: Optional[Dict[str, Any]]  # for debugging if you want to peek

def linkup_search(query: str, *, limit: int = 8, freshness_days: int = 7) -> LinkupResult:
    """
    POST /v1/search → normalize common shapes
    Returns:
      {
        "items": [{title, url, snippet?, published_at?}, ...],
        "answer": "..." | None
      }
    Prints helpful debug lines so you can confirm wiring.
    """
    print("[LINKUP] POST search starting…")
    if not LINKUP_API_KEY:
        print("[LINKUP] Missing LINKUP_API_KEY; skipping search.")
        return {"items": [], "answer": None, "raw": {}}

    today = dt.date.today()
    payload: Dict[str, Any] = {
        "q": query,
        "depth": LINKUP_DEPTH,               # "standard" or "deep"
        "outputType": LINKUP_OUTPUT,        # "sources" or "sourcedAnswer"
        "includeImages": False,
        "fromDate": (today - dt.timedelta(days=freshness_days)).isoformat(),
        "toDate": today.isoformat(),
    }
    if INCLUDE_DOMAINS:
        payload["includeDomains"] = INCLUDE_DOMAINS
    if EXCLUDE_DOMAINS:
        payload["excludeDomains"] = EXCLUDE_DOMAINS

    headers = {
        "Authorization": f"Bearer {LINKUP_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as cx:
            r = cx.post(LINKUP_API_URL, headers=headers, json=payload)
            if r.status_code == 401:
                print("[LINKUP] 401 Unauthorized. Check your API key.")
            if r.status_code == 404:
                print("[LINKUP] 404 Not Found. Ensure POST + correct /v1/search path.")

            r.raise_for_status()
            data = r.json() or {}

            # --- Normalize sources ---
            # 1) Docs example: {"answer": "...", "sources": [{name,url,snippet,...}]}
            # 2) Other shapes sometimes: "results", "data", or "response"
            raw_sources = (
                data.get("sources")
                or data.get("results")
                or data.get("data")
                or data.get("response")
                or []
            )
            items: List[LinkupItem] = []
            for it in raw_sources:
                url = it.get("url")
                if not url:
                    continue
                items.append({
                    "title": it.get("title") or it.get("name") or it.get("headline") or url,
                    "url": url,
                    "snippet": it.get("snippet") or it.get("summary") or it.get("description"),
                    "published_at": it.get("published_at") or it.get("publishedAt") or it.get("date"),
                })
                if len(items) >= limit:
                    break

            answer = data.get("answer")
            if answer:
                print("[LINKUP] Answer:", answer)
            print(f"[LINKUP] {len(items)} source(s) normalized. depth={LINKUP_DEPTH} output={LINKUP_OUTPUT}")

            return {"items": items, "answer": answer, "raw": data}

    except httpx.HTTPStatusError as e:
        body = e.response.text[:300] if e.response is not None else str(e)
        print(f"[LINKUP] HTTP {e.response.status_code if e.response else ''}: {body}")
    except httpx.ConnectError as e:
        print(f"[LINKUP] ConnectError: {e} (proxy/VPN/firewall?)")
    except Exception as e:
        print(f"[LINKUP] Unexpected error: {type(e).__name__}: {e}")

    return {"items": [], "answer": None, "raw": {}}

# async def fetch_latest_news_for_holdings(holdings: List[str] | None) -> List[LinkupItem]:
#     """
#     Fetch latest news articles for demo holdings.
#     """
#     all_news: List[LinkupItem] = []
#     if not holdings:
#         return all_news

#     for symbol in holdings:
#         # print(f"[LINKUP] Searching news for {symbol}…")
#         result = linkup_search(f"{symbol} stock news", limit=2, freshness_days=7)
#         all_news.extend(result["items"])

#     # Deduplicate by URL
#     seen_urls = set()
#     unique_news = []
#     for item in all_news:
#         if item["url"] not in seen_urls:
#             seen_urls.add(item["url"])
#             unique_news.append(item)

#     # Sort by published_at if available
#     def parse_date(date_str: Optional[str]) -> dt.datetime:
#         if not date_str:
#             return dt.datetime.min
#         for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
#             try:
#                 return dt.datetime.strptime(date_str, fmt)
#             except ValueError:
#                 continue
#         return dt.datetime.min

#     unique_news.sort(key=lambda x: parse_date(x.get("published_at")), reverse=True)

#     return unique_news[:SEARCH_LIMIT]
import asyncio
from typing import Iterable


def _auth_header() -> dict:
    if not LINKUP_API_KEY:
        raise RuntimeError("LINKUP_API_KEY is missing.")
    return {"Authorization": f"Bearer {LINKUP_API_KEY}"}

def _coerce_item(doc: dict) -> LinkupItem:
    """
    Normalize a single Linkup search result into LinkupItem.
    Handles both camelCase and snake_case keys defensively.
    """
    # print("[LINKUP] Raw doc:", doc)
    print("[LINKUP] Raw doc keys:", doc.keys())  # defensive: ensure it's a dict
    title = doc.get("title") or doc.get("name") or ""
    url = doc.get("url") or doc.get("link") or ""
    # snippet = doc.get("snippet") or doc.get("description")
    snippet = doc.get("snippet") or doc.get("summary") or doc.get("description") or doc.get("content")
    # Linkup commonly uses publishedAt; normalize to published_at.
    published_at = doc.get("publishedAt") or doc.get("published_at")
    return LinkupItem(title=title, url=url, snippet=snippet, published_at=published_at)

def _news_query(symbol: str, company_name: Optional[str]) -> str:
    """
    Build a precise-yet-flexible query for 'latest news' around a ticker.
    If a company name is provided, include it to reduce ambiguity (e.g., 'SBSI' vs unrelated acronyms).
    """
    # Keep it neutral and news-focused; Linkup is prompt-sensitive.
    base = f"Latest credible news headlines about {symbol} stock"
    if company_name:
        base += f" ({company_name})"
    base += " with links and publication dates"
    return base

def _today_iso() -> str:
    return dt.date.today().isoformat()

def _days_ago_iso(days: int) -> str:
    return (dt.date.today() - dt.timedelta(days=days)).isoformat()

# ----------------- Core (sync) -----------------

def get_latest_news_for_symbols(
    symbols: List[str],
    *,
    company_names: Optional[Dict[str, str]] = None,
    days_back: int = 14,
    limit: int = SEARCH_LIMIT,
    depth: str = LINKUP_DEPTH,
    include_domains: Optional[Iterable[str]] = None,
    exclude_domains: Optional[Iterable[str]] = None,
    timeout: float = 15.0,
) -> Dict[str, List[LinkupItem]]:
    """
    Fetch recent news for each ticker symbol using Linkup's /v1/search.

    Args:
        symbols: e.g. ["AAPL", "SBSI", "BTC"].
        company_names: optional mapping (e.g. {"SBSI": "Southside Bancshares Inc."}) to disambiguate acronyms.
        days_back: restrict results to the last N days via fromDate (YYYY-MM-DD per Linkup).
        limit: max results per symbol.
        depth: "standard" or "deep" per Linkup.
        include/exclude_domains: optional domain filters from your env or per-call.
        timeout: HTTP timeout (seconds).

    Returns:
        { "SBSI": [LinkupItem, ...], "BTC": [LinkupItem, ...], ... }
    """
    include_domains = list(include_domains or INCLUDE_DOMAINS)
    exclude_domains = list(exclude_domains or EXCLUDE_DOMAINS)

    if not symbols:
        return {}

    headers = {**_auth_header(), "Content-Type": "application/json"}
    cutoff = _days_ago_iso(days_back)

    out: Dict[str, List[LinkupItem]] = {}
    with httpx.Client(timeout=timeout) as client:
        for sym in symbols:
            payload: Dict[str, Any] = {
                "q": _news_query(sym, (company_names or {}).get(sym)),
                "depth": depth,
                # For news feeds, we prefer documents; override OUTPUT to "searchResults".
                "outputType": "searchResults",
                "fromDate": cutoff,  # YYYY-MM-DD per docs
                "toDate": _today_iso(),
                "includeImages": False,
            }
            if include_domains:
                payload["includeDomains"] = include_domains
            if exclude_domains:
                payload["excludeDomains"] = exclude_domains

            # Make the call
            r = client.post(LINKUP_API_URL, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

            # Linkup 'searchResults' returns an object with a 'results' list (plus possible metadata)
            raw_results = data.get("results") or data.get("sources") or data.get("documents") or []
            # Trim to limit and normalize shape
            items = [_coerce_item(doc) for doc in raw_results][:limit]
            out[sym] = items
    return out

# ----------------- Core (async + concurrent) -----------------

async def fetch_latest_news_for_holdings(
    symbols: List[str],
    *,
    company_names: Optional[Dict[str, str]] = None,
    days_back: int = 14,
    limit: int = SEARCH_LIMIT,
    depth: str = LINKUP_DEPTH,
    include_domains: Optional[Iterable[str]] = None,
    exclude_domains: Optional[Iterable[str]] = None,
    concurrency: int = 6,
    timeout: float = 15.0,
) -> Dict[str, List[LinkupItem]]:
    """
    Async version with bounded concurrency (great for 10–50 tickers).
    """
    include_domains = list(include_domains or INCLUDE_DOMAINS)
    exclude_domains = list(exclude_domains or EXCLUDE_DOMAINS)

    if not symbols:
        return {}

    headers = {**_auth_header(), "Content-Type": "application/json"}
    cutoff = _days_ago_iso(days_back)

    sem = asyncio.Semaphore(concurrency)
    results: Dict[str, List[LinkupItem]] = {}

    async def _one(sym: str, client: httpx.AsyncClient):
        nonlocal results
        payload: Dict[str, Any] = {
            "q": _news_query(sym, (company_names or {}).get(sym)),
            "depth": depth,
            "outputType": "searchResults",  # documents for news feeds
            "fromDate": cutoff,
            "toDate": _today_iso(),
            "includeImages": False,
        }
        if include_domains:
            payload["includeDomains"] = include_domains
        if exclude_domains:
            payload["excludeDomains"] = exclude_domains

        async with sem:
            resp = await client.post(LINKUP_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_results = data.get("results") or data.get("sources") or data.get("documents") or []
            items = [_coerce_item(doc) for doc in raw_results][:limit]
            results[sym] = items

    async with httpx.AsyncClient(timeout=timeout) as client:
        await asyncio.gather(*[_one(s, client) for s in symbols])

    return results
