#services/finnhub/client.py
import httpx

FINNHUB_CLIENT = httpx.AsyncClient(
    timeout=httpx.Timeout(5.0, connect=2.0),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    http2=True,
)
