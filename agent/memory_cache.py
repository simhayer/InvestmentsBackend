from __future__ import annotations

import os
from typing import Any, Dict, List

from services.cache.cache_backend import cache_get, cache_set
from .state_models import MemorySnapshot

CHAT_HISTORY_TTL_SEC = int(os.getenv("CHAT_HISTORY_TTL_SEC", "21600"))  # 6h
CHAT_MAX_HISTORY = int(os.getenv("CHAT_MAX_HISTORY", "14"))
CHAT_SUMMARY_MAX_CHARS = int(os.getenv("CHAT_SUMMARY_MAX_CHARS", "1200"))
CHAT_ENTITIES_MAX = int(os.getenv("CHAT_ENTITIES_MAX", "20"))


def _session_key(user_id: Any, session_id: str) -> str:
    return f"CHAT:SESSION:{str(user_id)}:{(session_id or '').strip()}"


def _summary_key(user_id: Any, session_id: str) -> str:
    return f"CHAT:SUMMARY:{str(user_id)}:{(session_id or '').strip()}"


def _entities_key(user_id: Any, session_id: str) -> str:
    return f"CHAT:ENTITIES:{str(user_id)}:{(session_id or '').strip()}"


def load_chat_history(user_id: Any, session_id: str) -> List[Dict[str, str]]:
    payload = cache_get(_session_key(user_id, session_id))
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list):
            return [m for m in messages if isinstance(m, dict)]
    return []


def save_chat_history(user_id: Any, session_id: str, messages: List[Dict[str, str]]) -> None:
    cache_set(
        _session_key(user_id, session_id),
        {"messages": messages},
        ttl_seconds=CHAT_HISTORY_TTL_SEC,
    )


def append_chat_history(
    user_id: Any,
    session_id: str,
    new_messages: List[Dict[str, str]],
    max_items: int = CHAT_MAX_HISTORY,
) -> List[Dict[str, str]]:
    history = load_chat_history(user_id, session_id)
    history.extend(new_messages)
    if len(history) > max_items:
        history = history[-max_items:]
    save_chat_history(user_id, session_id, history)
    return history


def load_thread_summary(user_id: Any, session_id: str) -> str:
    payload = cache_get(_summary_key(user_id, session_id))
    if isinstance(payload, dict):
        summary = payload.get("summary")
        if isinstance(summary, str):
            return summary
    return ""


def save_thread_summary(user_id: Any, session_id: str, summary: str) -> None:
    cache_set(
        _summary_key(user_id, session_id),
        {"summary": summary},
        ttl_seconds=CHAT_HISTORY_TTL_SEC,
    )


def load_recent_entities(user_id: Any, session_id: str) -> List[str]:
    payload = cache_get(_entities_key(user_id, session_id))
    if isinstance(payload, dict):
        items = payload.get("entities")
        if isinstance(items, list):
            out = []
            for item in items:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
            return out
    return []


def save_recent_entities(user_id: Any, session_id: str, entities: List[str]) -> None:
    cache_set(
        _entities_key(user_id, session_id),
        {"entities": entities},
        ttl_seconds=CHAT_HISTORY_TTL_SEC,
    )


def update_thread_summary(
    previous_summary: str,
    user_message: str,
    assistant_message: str,
    max_chars: int = CHAT_SUMMARY_MAX_CHARS,
) -> str:
    parts = []
    if previous_summary:
        parts.append(previous_summary.strip())
    if user_message:
        parts.append(f"User: {user_message.strip()}")
    if assistant_message:
        parts.append(f"Assistant: {assistant_message.strip()}")
    combined = "\n".join([p for p in parts if p])
    if len(combined) <= max_chars:
        return combined
    return combined[-max_chars:]


def load_memory_snapshot(user_id: Any, session_id: str, max_turns: int = CHAT_MAX_HISTORY) -> MemorySnapshot:
    summary = load_thread_summary(user_id, session_id)
    entities = load_recent_entities(user_id, session_id)
    recent_turns = load_chat_history(user_id, session_id)
    if max_turns > 0:
        recent_turns = recent_turns[-max_turns:]
    return MemorySnapshot(
        thread_summary=summary,
        recent_entities=entities,
        recent_turns=recent_turns,
    )
