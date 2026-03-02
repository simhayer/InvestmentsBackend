# Market Monitor Frontend Handoff

## Endpoints

- `GET /api/market/monitor-panel`
  - Public global monitor payload.
- `GET /api/market/monitor-panel/personalized`
  - Authenticated personalized payload.
  - Query params:
    - `watchlist_id`: optional integer. Uses a persisted watchlist for personalization.
    - `refresh`: optional boolean.
    - `currency`: optional string.
- `GET /api/watchlists`
- `POST /api/watchlists`
- `GET /api/watchlists/{watchlist_id}`
- `PATCH /api/watchlists/{watchlist_id}`
- `POST /api/watchlists/{watchlist_id}/items`
- `DELETE /api/watchlists/{watchlist_id}/items/{symbol}`
- `DELETE /api/watchlists/{watchlist_id}`

## Personalized Contract

Top-level response:

```ts
type PersonalizedMarketMonitorEnvelope = {
  message: string
  data: PersonalizedMarketMonitorPayload
}
```

Personalization block:

```ts
type PersonalizationBlockProps = {
  scope: "portfolio" | "watchlist" | "global_fallback"
  currency: string
  symbols: string[]
  watchlist: {
    id: number
    name: string
    is_default: boolean
  } | null
  top_positions: Array<{
    symbol: string | null
    name: string | null
    weight: number | null
    current_value: number | null
    unrealized_pl_pct: number | null
    current_price: number | null
    currency: string | null
  }>
  portfolio_snapshot: {
    as_of?: number | null
    currency?: string | null
    price_status?: string | null
    positions_count?: number | null
    market_value?: number | null
    cost_basis?: number | null
    unrealized_pl?: number | null
    unrealized_pl_pct?: number | null
    day_pl?: number | null
    day_pl_pct?: number | null
    allocations?: Record<string, unknown> | null
    connections?: Array<Record<string, unknown>> | null
  } | null
  inline_insights: {
    healthBadge?: string | null
    performanceNote?: string | null
    riskFlag?: string | null
    topPerformer?: string | null
    actionNeeded?: string | null
    disclaimer?: string | null
  } | null
  insight_cards: Array<{
    title: string
    summary: string
    signal: "bullish" | "bearish" | "neutral"
    time_horizon: string
  }>
  focus_news: Array<{
    symbol: string
    items: Array<{
      title: string | null
      url: string | null
      source: string | null
      published_at: string | null
      snippet: string | null
      image: string | null
    }>
  }>
  empty_state: string | null
}
```

## Rendering Rules

- `scope === "portfolio"`
  - Show `portfolio_snapshot`, `top_positions`, `inline_insights`, `insight_cards`, `focus_news`.
- `scope === "watchlist"`
  - Show `watchlist`, `insight_cards`, `focus_news`.
  - Hide `portfolio_snapshot` and `top_positions`.
- `scope === "global_fallback"`
  - Show `empty_state`.
  - Render the global monitor sections only.

## Migration Notes

- Frontend should stop using `symbols=` as the primary personalization path.
- Use persisted watchlists via `/api/watchlists` and pass `watchlist_id` into `/api/market/monitor-panel/personalized`.
- The original global sections from the previous monitor work remain unchanged.
