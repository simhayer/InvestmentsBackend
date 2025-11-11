METRICS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "per_symbol": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "symbol": {"type": "string"},
                    "name": {"type": "string"},
                    "asset_class": {
                        "type": "string",
                        "enum": ["equity", "etf", "fund", "crypto", "bond", "cash", "other"],
                    },
                    "sector": {"type": "string"},
                    "region": {"type": "string"},
                    "weight_pct": {"type": "number"},        # % of portfolio
                    "market_value": {"type": "number"},      # in base_currency
                    "cost_basis": {"type": "number"},        # in base_currency
                    "unrealized_pnl_abs": {"type": "number"},
                    "unrealized_pnl_pct": {"type": "number"},  # %
                    # Timeframe returns (optional but recommended)
                    "return_1D_pct": {"type": "number"},
                    "return_1W_pct": {"type": "number"},
                    "return_1M_pct": {"type": "number"},
                    "return_3M_pct": {"type": "number"},
                    "return_1Y_pct": {"type": "number"},
                    # Risk metrics (optional)
                    "vol_30D_pct": {"type": "number"},
                    "max_drawdown_1Y_pct": {"type": "number"},
                    "beta_1Y": {"type": "number"},
                    "is_leveraged": {"type": "boolean"},
                },
                "required": [
                    "symbol",
                    "asset_class",
                    "weight_pct",
                    "market_value",
                    "cost_basis",
                    "unrealized_pnl_abs",
                    "unrealized_pnl_pct",
                ],
            },
        },
        "portfolio": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "total_value": {"type": "number"},
                "cash_value": {"type": "number"},
                "num_positions": {"type": "integer"},
                "concentration_top_5_pct": {"type": "number"},
                "core_weight_pct": {"type": "number"},
                "speculative_weight_pct": {"type": "number"},
                "hedge_weight_pct": {"type": "number"},
                # Allocation breakdowns (optional)
                "sector_weights_pct": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "asset_class_weights_pct": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "region_weights_pct": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                # Risk metrics (optional)
                "vol_30D_pct": {"type": "number"},
                "max_drawdown_1Y_pct": {"type": "number"},
            },
            "required": [
                "total_value",
                "cash_value",
                "num_positions",
                "concentration_top_5_pct",
                "core_weight_pct",
                "speculative_weight_pct",
                "hedge_weight_pct",
            ],
        },
    },
    "required": ["per_symbol", "portfolio"],
}
