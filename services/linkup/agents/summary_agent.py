from __future__ import annotations
from services.linkup.linkup_search import linkup_structured_search

SUMMARY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "disclaimer": {"type": "string"},
        "explainability": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "confidence_overall": {"type": "number", "minimum": 0, "maximum": 1},
                "section_confidence": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "news": {"type": "number"},
                        "catalysts": {"type": "number"},
                        "sentiment": {"type": "number"},
                        "risks_list": {"type": "number"},
                        "performance_analysis": {"type": "number"},
                        "predictions": {"type": "number"},
                        "scenarios": {"type": "number"},
                        "rebalance_paths": {"type": "number"},
                        "market_outlook": {"type": "number"},
                        "actions": {"type": "number"},
                    },
                },
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    "required": ["summary", "disclaimer"],
}

def build_summary_query(
    news_sentiment_json,
    performance_predictions_json,
    scenarios_rebalance_json,
):
    return {
        "role": (
            "You are a professional portfolio analyst summarizer. "
            "You do NOT fetch new data. You ONLY read the three JSON objects provided "
            "and write a concise, user-facing overview and disclaimer. "
            "Output ONLY valid JSON matching SUMMARY_SCHEMA."
        ),
        "step_1_task": [
            "Synthesize the key ideas from prior sections into a 2–5 paragraph natural-language summary "
            "that a non-expert can understand.",
            "Highlight major themes: concentration, diversification, recent news, sentiment, and key risks.",
            "Do NOT recommend any specific actions like buying, selling, adding, trimming, or holding assets.",
        ],
        "step_2_context": [
            "Treat all prior JSON as already quality-checked: do not re-interpret or change the factual content.",
            "You may reorder and compress ideas, but not introduce new facts or predictions.",
        ],
        "step_3_references": [
            "You are not required to include citations here; the detailed sections already hold them.",
            "If you mention numbers or probabilities, take them directly from the input JSON.",
        ],
        "step_4_evaluate": [
            "Check that the summary does not contradict any of the source JSONs.",
            "Ensure language is clearly informational and encourages users to seek professional advice "
            "for decisions.",
        ],
        "step_5_iterate": [
            "Simplify wording where possible while preserving accuracy.",
            "Make limitations and uncertainty explicit in the summary where helpful.",
        ],
        "constraints": [
            "STRICTLY conform to SUMMARY_SCHEMA.",
            "Do NOT use words: 'buy', 'sell', 'add', 'trim', 'hold', "
            "or concrete transaction instructions.",
        ],
        "inputs": {
            "news_sentiment": news_sentiment_json,
            "performance_predictions": performance_predictions_json,
            "scenarios_rebalance": scenarios_rebalance_json,
        },
        # You can hard-code or template this disclaimer if you want, but the agent can also generate it:
        "disclaimer_template_hint": (
            "Emphasize that the information is general, may be incomplete, is based on historical data "
            "and public sources, and is not personalized investment, tax, or legal advice. "
            "Encourage the user to consult a qualified professional before making decisions."
        ),
    }

def call_link_up_for_summary(news_sentiment_json,
    performance_predictions_json,
    scenarios_rebalance_json):
    return linkup_structured_search(
        query_obj=build_summary_query(
            news_sentiment_json=news_sentiment_json,
            performance_predictions_json=performance_predictions_json,
            scenarios_rebalance_json=scenarios_rebalance_json,
        ),
        schema=SUMMARY_SCHEMA,
        days=7,
        include_sources=False,
    )
