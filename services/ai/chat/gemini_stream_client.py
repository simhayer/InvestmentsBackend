from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)

def _trace_info(msg: str, *args: Any) -> None:
    text = msg % args if args else msg
    print(text, flush=True)
    logger.info(msg, *args)


def _trace_warning(msg: str, *args: Any) -> None:
    text = msg % args if args else msg
    print(text, flush=True)
    logger.warning(msg, *args)


def _trace_exception(msg: str, *args: Any) -> None:
    text = msg % args if args else msg
    print(text, flush=True)
    logger.exception(msg, *args)


@dataclass
class GeminiConfig:
    api_key: str
    model: str
    use_web: bool
    temperature: float
    thinking_level: str


class GeminiStreamClient:
    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or self._from_env()

    def _from_env(self) -> GeminiConfig:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")
        return GeminiConfig(
            api_key=api_key,
            model=os.getenv("GEMINI_MODEL") or "gemini-3-flash-preview",
            use_web=os.getenv("GEMINI_USE_WEB", "1") == "1",
            temperature=float(os.getenv("AI_TEMPERATURE", "0.3")),
            thinking_level=(os.getenv("GEMINI_THINKING_LEVEL") or "MEDIUM").upper(),
        )

    @staticmethod
    def _normalize_thinking(model: str, level: str) -> str:
        lvl = (level or "MEDIUM").upper()
        m = (model or "").lower()
        if "flash" in m:
            return lvl if lvl in ("LOW", "MEDIUM", "HIGH") else "MEDIUM"
        return lvl if lvl in ("LOW", "HIGH") else "HIGH"

    def _build_config(self, *, use_web: bool, model: Optional[str] = None):
        from google.genai import types

        tools = [types.Tool(googleSearch=types.GoogleSearch())] if use_web else None
        thinking_config = None
        try:
            fields = set(getattr(types.ThinkingConfig, "model_fields", {}).keys())
            normalized = self._normalize_thinking(model or self.config.model, self.config.thinking_level)
            if "thinking_level" in fields:
                thinking_config = types.ThinkingConfig(thinking_level=normalized)
            elif "include_thoughts" in fields:
                # Newer SDK versions replaced level controls with a boolean switch.
                thinking_config = types.ThinkingConfig(include_thoughts=(normalized != "LOW"))
        except Exception:
            thinking_config = None

        return types.GenerateContentConfig(
            thinking_config=thinking_config,
            tools=tools,
            temperature=self.config.temperature,
        )

    async def decide_tool_action(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        chosen_model = model_override or self.config.model
        _trace_info("gemini.decide.start model=%s prompt_len=%s", chosen_model, len(user_prompt or ""))
        raw = await asyncio.to_thread(self._sync_generate_text, system_prompt, user_prompt, False, chosen_model)
        text = (raw or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines and lines[-1].strip().startswith("```") else lines[1:]).strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                _trace_info("gemini.decide.done elapsed_ms=%s is_json=true", elapsed_ms)
                return parsed
        except Exception:
            pass
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _trace_warning("gemini.decide.done elapsed_ms=%s is_json=false fallback=answer", elapsed_ms)
        return {"use_web": False, "action": "answer", "tool_name": None, "arguments": {}, "reason": "non_json_tool_decision"}

    def _sync_generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        use_web: bool,
        model_override: Optional[str] = None,
    ) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.config.api_key)
        model = model_override or self.config.model
        combined = f"{system_prompt}\n\n{user_prompt}"
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=combined)])]
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=self._build_config(use_web=use_web, model=model),
        )
        return getattr(resp, "text", None) or str(resp)

    async def stream_answer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        allow_web_search: Optional[bool] = None,
    ) -> AsyncIterator[str]:
        use_web = self.config.use_web if allow_web_search is None else bool(allow_web_search)
        _trace_info(
            "gemini.stream.start model=%s use_web=%s prompt_len=%s",
            self.config.model,
            use_web,
            len(user_prompt or ""),
        )
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[str]] = asyncio.Queue()
        started = time.perf_counter()
        worker_state: Dict[str, Any] = {"error": None, "chunks_emitted": 0}

        def _worker() -> None:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.config.api_key)
            combined = f"{system_prompt}\n\n{user_prompt}"
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=combined)])]
            for attempt in (1, 2):
                try:
                    stream = client.models.generate_content_stream(
                        model=self.config.model,
                        contents=contents,
                        config=self._build_config(use_web=use_web),
                    )
                    emitted = 0
                    for chunk in stream:
                        text = getattr(chunk, "text", None)
                        if text:
                            emitted += 1
                            worker_state["chunks_emitted"] = int(worker_state["chunks_emitted"]) + 1
                            loop.call_soon_threadsafe(q.put_nowait, text)
                    _trace_info("gemini.stream.worker.done chunks=%s attempt=%s", emitted, attempt)
                    worker_state["error"] = None
                    break
                except Exception as exc:
                    worker_state["error"] = exc
                    chunks_emitted = int(worker_state["chunks_emitted"])
                    if chunks_emitted > 0:
                        # Non-fatal: stream produced content, then transport parse failed.
                        # Avoid noisy traceback spam for this frequent SDK chunk-decoding issue.
                        _trace_warning(
                            "gemini.stream.worker.partial_decode_error attempt=%s chunks_emitted=%s err=%s",
                            attempt,
                            chunks_emitted,
                            type(exc).__name__,
                        )
                        break
                    if attempt == 2:
                        _trace_exception(
                            "gemini.stream.worker.error attempt=%s chunks_emitted=%s",
                            attempt,
                            chunks_emitted,
                        )
                        break
                    _trace_warning("gemini.stream.worker.retry attempt=%s reason=stream_decode_error", attempt + 1)
                    time.sleep(0.15)
            loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=_worker, daemon=True).start()

        emitted_tokens = 0
        while True:
            item = await q.get()
            if item is None:
                break
            emitted_tokens += 1
            yield item

        # Fallback only if stream failed before any token was emitted.
        if worker_state["error"] is not None and emitted_tokens == 0:
            fallback_timeout_s = float(os.getenv("GEMINI_SYNC_FALLBACK_TIMEOUT_S", "12"))
            _trace_warning("gemini.stream.fallback.start timeout_s=%s", fallback_timeout_s)
            try:
                fallback = await asyncio.wait_for(
                    asyncio.to_thread(self._sync_generate_text, system_prompt, user_prompt, use_web),
                    timeout=fallback_timeout_s,
                )
                # Emit moderate chunks to reduce SSE overhead.
                text = (fallback or "").strip()
                chunk_size = 180
                for i in range(0, len(text), chunk_size):
                    yield text[i:i + chunk_size]
                if text:
                    emitted_tokens += (len(text) + chunk_size - 1) // chunk_size
                _trace_info("gemini.stream.fallback.done emitted_chunks=%s", emitted_tokens)
            except Exception:
                _trace_exception("gemini.stream.fallback.error")

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _trace_info("gemini.stream.done elapsed_ms=%s emitted_events=%s", elapsed_ms, emitted_tokens)
