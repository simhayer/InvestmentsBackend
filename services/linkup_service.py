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
