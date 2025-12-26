import os
from linkup import LinkupClient
import json, time, random
from datetime import datetime, timedelta, timezone

LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "")
if not LINKUP_API_KEY:
    raise RuntimeError("LINKUP_API_KEY is missing")
client = LinkupClient(api_key=LINKUP_API_KEY)

def get_linkup_client() -> LinkupClient:
    return client

def linkup_structured_search(
    *,
    query_obj: dict,
    schema: dict,
    days: int = 7,
    include_sources: bool = True,
    depth: str = "standard",
    max_retries: int = 2,
) -> dict:
    if depth not in ("standard", "deep"):
        depth = "standard"
    t0 = time.perf_counter()
    now = datetime.now(timezone.utc)
    to_date = now.date()
    from_date = (now - timedelta(days=days)).date()

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.search(
                query=json.dumps(query_obj),
                depth=depth,
                output_type="structured",
                structured_output_schema=json.dumps(schema),
                include_images=False,
                include_sources=include_sources,
                from_date=from_date,
                to_date=to_date,
            )
            return {
                "ok": True,
                "data": resp,
                "error": None,
                "meta": {
                    "from_date": str(from_date),
                    "to_date": str(to_date),
                    "duration_ms": int((time.perf_counter() - t0) * 1000),
                },
            }
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep((0.6 * (2 ** attempt)) + random.random() * 0.3)
            else:
                return {
                    "ok": False,
                    "data": None,
                    "error": str(last_err),
                    "meta": {
                        "from_date": str(from_date),
                        "to_date": str(to_date),
                        "duration_ms": int((time.perf_counter() - t0) * 1000),
                    },
                }
            
    return {
        "ok": False,
        "data": None,
        "error": str(last_err),
        "meta": {
            "from_date": str(from_date),
            "to_date": str(to_date),
            "duration_ms": int((time.perf_counter() - t0) * 1000),
        },
    }
