# backend/schemas/metrics_schema.py

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
                        "enum": [
                            "equity",
                            "etf",
                            "fund",
                            "cryptocurrency",
                            "bond",
                            "cash",
                            "other",
                        ],
                    },
                    "sector": {"type": "string"},
                    "region": {"type": "string"},

                    "weight_pct": {"type": "number"},
                    "market_value": {"type": "number"},
                    "cost_basis": {"type": "number"},

                    # can be null if no cost basis
                    "unrealized_pnl_abs": {"type": ["number", "null"]},
                    "unrealized_pnl_pct": {"type": ["number", "null"]},

                    # optional performance
                    "return_1D_pct": {"type": ["number", "null"]},
                    "return_1W_pct": {"type": ["number", "null"]},
                    "return_1M_pct": {"type": ["number", "null"]},
                    "return_3M_pct": {"type": ["number", "null"]},
                    "return_1Y_pct": {"type": ["number", "null"]},

                    # optional risk
                    "vol_30D_pct": {"type": ["number", "null"]},
                    "max_drawdown_1Y_pct": {"type": ["number", "null"]},
                    "beta_1Y": {"type": ["number", "null"]},
                    "is_leveraged": {"type": ["boolean", "null"]},

                    # fundamentals / quote fields you are returning now
                    "pe_ratio": {"type": ["number", "null"]},
                    "forward_pe": {"type": ["number", "null"]},
                    "market_cap": {"type": ["number", "null"]},
                    "price_to_book": {"type": ["number", "null"]},
                    "dividend_yield": {"type": ["number", "null"]},
                    "quote_currency": {"type": ["string", "null"]},
                    "price_status": {"type": ["string", "null"]},
                },
                "required": [
                    "symbol",
                    "name",
                    "asset_class",
                    "sector",
                    "region",
                    "weight_pct",
                    "market_value",
                    "cost_basis",
                    "unrealized_pnl_abs",
                    "unrealized_pnl_pct",
                    "price_status",
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

                "core_weight_pct": {"type": ["number", "null"]},
                "speculative_weight_pct": {"type": ["number", "null"]},
                "hedge_weight_pct": {"type": ["number", "null"]},

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

                "vol_30D_pct": {"type": ["number", "null"]},
                "max_drawdown_1Y_pct": {"type": ["number", "null"]},

                # your extra portfolio fields
                "base_currency": {"type": "string"},
                "usd_to_cad": {"type": ["number", "null"]},
            },
            "required": [
                "total_value",
                "cash_value",
                "num_positions",
                "concentration_top_5_pct",
                "asset_class_weights_pct",
                "base_currency",
            ],
        },
    },
    "required": ["per_symbol", "portfolio"],
}
