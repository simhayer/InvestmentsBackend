from __future__ import annotations

import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Tuple

from sqlalchemy.orm import Session

from agent.decisioning_graph import DecisioningGraph
from agent.memory_cache import append_chat_history, load_chat_history
from agent.state_models import GraphState, ToolResult

logger = logging.getLogger("chat_agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

_graph_instance: DecisioningGraph | None = None


def _get_graph() -> DecisioningGraph:
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = DecisioningGraph()
    return _graph_instance


def _tools_used(tool_results: List[ToolResult]) -> List[str]:
    return [res.source for res in tool_results if res.ok]


def _tool_errors(tool_results: List[ToolResult]) -> List[Dict[str, Any]]:
    errors = []
    for res in tool_results:
        if res.ok or not res.error:
            continue
        errors.append({"tool": res.source, "type": res.error.type, "message": res.error.message})
    return errors


def _chunk_text(text: str, size: int = 160) -> List[str]:
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


def build_stream_events(
    state: GraphState,
    answer: str,
    response_ms: float,
    session_id: str,
    chunk_size: int = 160,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    plan_payload = DecisioningGraph.build_plan_event(state)
    plan_payload["session_id"] = session_id
    events.append({"event": "plan", "data": plan_payload})

    for status in state.tool_statuses:
        payload = dict(status)
        payload["trace_id"] = state.trace_id
        payload["turn_id"] = state.turn_id
        events.append({"event": "tool_status", "data": payload})

    for chunk in _chunk_text(answer, size=chunk_size):
        events.append({"event": "delta", "data": chunk})

    events.append(
        {
            "event": "final",
            "data": {
                "trace_id": state.trace_id,
                "turn_id": state.turn_id,
                "recency_insufficient": state.recency_insufficient,
                "tools_used": _tools_used(state.tool_results),
                "response_ms": response_ms,
            },
        }
    )
    return events


async def run_chat_turn(
    *,
    message: str,
    user_id: Any,
    user_currency: str,
    session_id: str,
    db: Session,
    finnhub: Any,
) -> Tuple[str, Dict[str, Any]]:
    graph = _get_graph()
    state = DecisioningGraph.new_state(
        message=message,
        user_id=user_id,
        user_currency=user_currency,
        session_id=session_id,
        db=db,
        finnhub=finnhub,
    )
    final_state = await graph.run(state)

    debug = {
        "trace_id": final_state.trace_id,
        "turn_id": final_state.turn_id,
        "recency_insufficient": final_state.recency_insufficient,
        "tools_used": _tools_used(final_state.tool_results),
        "tool_errors": _tool_errors(final_state.tool_results),
    }
    return final_state.answer, debug


async def run_chat_turn_stream(
    *,
    message: str,
    user_id: Any,
    user_currency: str,
    session_id: str,
    db: Session,
    finnhub: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    graph = _get_graph()
    state = DecisioningGraph.new_state(
        message=message,
        user_id=user_id,
        user_currency=user_currency,
        session_id=session_id,
        db=db,
        finnhub=finnhub,
    )
    t0 = time.perf_counter()
    try:
        pre_state = await graph.run_pre_synthesis(state)
        pre_state = await graph.run_synthesis(pre_state)
        await graph.run_postprocess(pre_state)
        response_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        for event in build_stream_events(pre_state, pre_state.answer, response_ms, session_id):
            yield event
    except Exception as exc:
        logger.exception("streaming chat turn failed")
        yield {"event": "final", "data": {"error": str(exc)}}


__all__ = ["run_chat_turn", "run_chat_turn_stream", "load_chat_history", "append_chat_history"]
