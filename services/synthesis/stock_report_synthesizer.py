from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from agents import Agent, ModelBehaviorError, ModelSettings, Runner
from agents.models.default_models import get_default_model_settings
from openai.types.shared import Reasoning

from schemas.stock_report import StockReport

SYSTEM_PROMPT = """
You are a senior equity research analyst.
You produce a concise, current stock report based ONLY on the provided inputs.
Return ONLY valid JSON that matches the StockReport schema.

Rules:
- Use only the provided news, filings, and fundamentals. If missing, say so.
- Do not invent facts, dates, or numbers.
- No financial advice or recommendation language.
- Every claim about current events must cite source IDs in the "sources" fields.
- Use the provided source IDs exactly (e.g., "news_2", "filing_1").
- Populate "citations" from the provided items; do not add new IDs.
- If information is insufficient, keep sections short and note the limitation.
""".strip()


def _build_model_settings(model: str) -> ModelSettings:
    base_settings = get_default_model_settings(model)
    return base_settings.resolve(
        ModelSettings(
            tool_choice="none",
            reasoning=Reasoning(effort="low"),
        )
    )


def _build_prompt(inputs: Dict[str, Any]) -> str:
    payload = json.dumps(inputs, default=str)
    return (
        "Use the following JSON inputs to produce a StockReport.\n"
        "INPUT_JSON:\n"
        f"{payload}\n"
        "Make sure every current event claim includes a source id in sources arrays.\n"
        "Return ONLY JSON that matches the StockReport schema."
    )


async def synthesize_stock_report(
    inputs: Dict[str, Any],
    *,
    model: str | None = None,
    timeout_s: float = 16.0,
) -> StockReport:
    model_name = model or "gpt-5-mini"
    model_settings = _build_model_settings(model_name)

    agent = Agent(
        name="Stock report synthesizer",
        instructions=SYSTEM_PROMPT,
        model=model_name,
        model_settings=model_settings,
        tools=[],
        output_type=StockReport,
    )

    prompt = _build_prompt(inputs)
    try:
        result = await asyncio.wait_for(
            Runner.run(agent, input=prompt, max_turns=1),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError("LLM synthesis timed out") from exc
    except ModelBehaviorError as exc:
        raise RuntimeError(f"LLM synthesis failed: {exc}") from exc

    output = result.final_output
    if isinstance(output, StockReport):
        return output
    if isinstance(output, dict):
        return StockReport.model_validate(output)
    raise RuntimeError("Unexpected synthesis output type")
