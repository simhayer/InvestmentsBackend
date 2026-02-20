from __future__ import annotations

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator
import httpx
from sqlalchemy.orm import Session

from services.finnhub.finnhub_service import FinnhubService
from services.holding_service import get_holdings_with_live_prices


AssetType = Literal["stock", "cryptocurrency"]


class ToolResult(BaseModel):
    ok: bool
    tool_name: str
    data: Dict[str, Any] = Field(default_factory=dict)
    data_gaps: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class QuoteArgs(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    asset_type: AssetType = "stock"

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, v: str) -> str:
        return (v or "").strip().upper()


class SymbolArgs(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, v: str) -> str:
        return (v or "").strip().upper()


class PortfolioArgs(BaseModel):
    currency: str = Field(default="USD", max_length=8)
    top_n: int = Field(default=8, ge=1, le=25)

    @field_validator("currency")
    @classmethod
    def _normalize_currency(cls, v: str) -> str:
        out = (v or "USD").strip().upper()
        return out if out in {"USD", "CAD"} else "USD"


class PortfolioPositionArgs(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    currency: str = Field(default="USD", max_length=8)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, v: str) -> str:
        return (v or "").strip().upper()

    @field_validator("currency")
    @classmethod
    def _normalize_currency(cls, v: str) -> str:
        out = (v or "USD").strip().upper()
        return out if out in {"USD", "CAD"} else "USD"


class FinnhubToolRegistry:
    """Small, validated facade for chat-safe Finnhub tool calls."""

    def __init__(
        self,
        service: Optional[FinnhubService] = None,
        *,
        db: Optional[Session] = None,
        user_id: Optional[str] = None,
    ):
        self._service = service or FinnhubService()
        self._db = db
        self._user_id = user_id
        self._http = httpx.AsyncClient(timeout=6.0)

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        try:
            if tool_name == "get_quote":
                parsed = QuoteArgs(**(arguments or {}))
                return await self.get_quote(parsed)
            if tool_name == "get_company_profile":
                parsed = SymbolArgs(**(arguments or {}))
                return await self.get_company_profile(parsed)
            if tool_name == "get_basic_financials":
                parsed = SymbolArgs(**(arguments or {}))
                return await self.get_basic_financials(parsed)
            if tool_name == "get_peers":
                parsed = SymbolArgs(**(arguments or {}))
                return await self.get_peers(parsed)
            if tool_name == "get_portfolio_overview":
                parsed = PortfolioArgs(**(arguments or {}))
                return await self.get_portfolio_overview(parsed)
            if tool_name == "get_portfolio_position":
                parsed = PortfolioPositionArgs(**(arguments or {}))
                return await self.get_portfolio_position(parsed)
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                error=f"Unsupported tool: {tool_name}",
                data_gaps=["Unsupported tool"],
            )
        except ValidationError as exc:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                error=f"Invalid tool arguments: {exc.errors()}",
                data_gaps=["Invalid tool arguments"],
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                error=f"Tool execution failed: {exc}",
                data_gaps=["Tool execution failed"],
            )

    async def get_quote(self, args: QuoteArgs) -> ToolResult:
        payload = await self._service.get_price(args.symbol, typ=args.asset_type)
        data = {
            "symbol": payload.get("symbol"),
            "formattedSymbol": payload.get("formattedSymbol"),
            "currentPrice": payload.get("currentPrice"),
            "previousClose": payload.get("previousClose"),
            "high": payload.get("high"),
            "low": payload.get("low"),
            "open": payload.get("open"),
            "currency": payload.get("currency", "USD"),
        }
        return ToolResult(ok=True, tool_name="get_quote", data=data)

    async def get_company_profile(self, args: SymbolArgs) -> ToolResult:
        profile = await self._service.fetch_profile(args.symbol, client=self._http)
        if not profile:
            return ToolResult(
                ok=False,
                tool_name="get_company_profile",
                data={},
                data_gaps=["Company profile unavailable"],
                error="Company profile unavailable",
            )
        safe = {
            "symbol": args.symbol,
            "name": profile.get("name"),
            "country": profile.get("country"),
            "currency": profile.get("currency"),
            "exchange": profile.get("exchange"),
            "finnhubIndustry": profile.get("finnhubIndustry"),
            "ipo": profile.get("ipo"),
            "marketCapitalization": profile.get("marketCapitalization"),
        }
        return ToolResult(ok=True, tool_name="get_company_profile", data=safe)

    async def get_basic_financials(self, args: SymbolArgs) -> ToolResult:
        metrics = await self._service.fetch_basic_financials(args.symbol, client=self._http)
        metric_obj = metrics.get("metric", {}) if isinstance(metrics, dict) else {}
        if not isinstance(metric_obj, dict) or not metric_obj:
            return ToolResult(
                ok=False,
                tool_name="get_basic_financials",
                data={},
                data_gaps=["Basic financial metrics unavailable"],
                error="Basic financial metrics unavailable",
            )
        keep = {
            "peTTM": metric_obj.get("peTTM"),
            "pbAnnual": metric_obj.get("pbAnnual"),
            "psTTM": metric_obj.get("psTTM"),
            "roeTTM": metric_obj.get("roeTTM"),
            "roaTTM": metric_obj.get("roaTTM"),
            "currentRatioAnnual": metric_obj.get("currentRatioAnnual"),
            "debtToEquityAnnual": metric_obj.get("totalDebt/totalEquityAnnual"),
            "netMargin": metric_obj.get("netMargin"),
            "operatingMarginAnnual": metric_obj.get("operatingMarginAnnual"),
            "52WeekHigh": metric_obj.get("52WeekHigh"),
            "52WeekLow": metric_obj.get("52WeekLow"),
        }
        return ToolResult(
            ok=True,
            tool_name="get_basic_financials",
            data={"symbol": args.symbol, "metrics": keep},
        )

    async def get_peers(self, args: SymbolArgs) -> ToolResult:
        peers = await self._service.fetch_peers(args.symbol, client=self._http)
        if not peers:
            return ToolResult(
                ok=False,
                tool_name="get_peers",
                data={"symbol": args.symbol, "peers": []},
                data_gaps=["Peers unavailable"],
                error="Peers unavailable",
            )
        return ToolResult(
            ok=True,
            tool_name="get_peers",
            data={"symbol": args.symbol, "peers": peers[:12]},
        )

    def _require_portfolio_context(self, tool_name: str) -> Optional[ToolResult]:
        if self._db is None or not self._user_id:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                data={},
                data_gaps=["Missing portfolio context"],
                error="Portfolio tool unavailable without authenticated user context",
            )
        return None

    async def get_portfolio_overview(self, args: PortfolioArgs) -> ToolResult:
        missing = self._require_portfolio_context("get_portfolio_overview")
        if missing:
            return missing

        payload = await get_holdings_with_live_prices(
            user_id=str(self._user_id),
            db=self._db,  # type: ignore[arg-type]
            finnhub=self._service,
            currency=args.currency,
            top_only=False,
            top_n=args.top_n,
            include_weights=True,
        )
        items = payload.get("items", []) if isinstance(payload, dict) else []
        top_items = payload.get("top_items", []) if isinstance(payload, dict) else []

        normalized_top = []
        for h in top_items[: args.top_n]:
            normalized_top.append(
                {
                    "symbol": getattr(h, "symbol", None),
                    "name": getattr(h, "name", None),
                    "type": getattr(h, "type", None),
                    "weight": getattr(h, "weight", None),
                    "value": getattr(h, "value", None),
                    "current_price": getattr(h, "current_price", None),
                    "unrealized_pl": getattr(h, "unrealized_pl", None),
                    "unrealized_pl_pct": getattr(h, "unrealized_pl_pct", None),
                }
            )

        unique_symbols = sorted({getattr(h, "symbol", "").upper() for h in items if getattr(h, "symbol", None)})
        data = {
            "currency": payload.get("currency") if isinstance(payload, dict) else args.currency,
            "market_value": payload.get("market_value") if isinstance(payload, dict) else None,
            "holdings_count": len(items),
            "symbols": unique_symbols[:100],
            "top_holdings": normalized_top,
            "as_of": payload.get("as_of") if isinstance(payload, dict) else None,
        }
        return ToolResult(ok=True, tool_name="get_portfolio_overview", data=data)

    async def get_portfolio_position(self, args: PortfolioPositionArgs) -> ToolResult:
        missing = self._require_portfolio_context("get_portfolio_position")
        if missing:
            return missing

        payload = await get_holdings_with_live_prices(
            user_id=str(self._user_id),
            db=self._db,  # type: ignore[arg-type]
            finnhub=self._service,
            currency=args.currency,
            top_only=False,
            top_n=25,
            include_weights=True,
        )
        items = payload.get("items", []) if isinstance(payload, dict) else []
        matches = [h for h in items if getattr(h, "symbol", "").upper() == args.symbol]
        if not matches:
            return ToolResult(
                ok=False,
                tool_name="get_portfolio_position",
                data={"symbol": args.symbol, "positions": []},
                data_gaps=[f"No portfolio position found for {args.symbol}"],
                error=f"No portfolio position found for {args.symbol}",
            )

        positions = []
        total_value = 0.0
        total_quantity = 0.0
        for h in matches:
            qty = float(getattr(h, "quantity", 0.0) or 0.0)
            val = float(getattr(h, "value", 0.0) or 0.0)
            total_quantity += qty
            total_value += val
            positions.append(
                {
                    "symbol": getattr(h, "symbol", None),
                    "name": getattr(h, "name", None),
                    "account_name": getattr(h, "account_name", None),
                    "type": getattr(h, "type", None),
                    "quantity": getattr(h, "quantity", None),
                    "current_price": getattr(h, "current_price", None),
                    "value": getattr(h, "value", None),
                    "weight": getattr(h, "weight", None),
                    "unrealized_pl": getattr(h, "unrealized_pl", None),
                    "unrealized_pl_pct": getattr(h, "unrealized_pl_pct", None),
                }
            )

        return ToolResult(
            ok=True,
            tool_name="get_portfolio_position",
            data={
                "symbol": args.symbol,
                "positions": positions,
                "aggregate": {
                    "total_quantity": round(total_quantity, 8),
                    "total_value": round(total_value, 8),
                    "currency": payload.get("currency") if isinstance(payload, dict) else args.currency,
                },
            },
        )
