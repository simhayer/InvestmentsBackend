from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


def _trace_info(msg: str, *args: Any) -> None:
    logger.info(msg, *args)


def _trace_warning(msg: str, *args: Any) -> None:
    logger.warning(msg, *args)


def _trace_exception(msg: str, *args: Any) -> None:
    logger.exception(msg, *args)


@dataclass
class GeminiConfig:
    model: str
    use_web: bool
    temperature: float
    thinking_budget: int
    project_id: str
    location: str


class GeminiStreamClient:
    _thinking_unsupported_models: set = set()

    def __init__(self, config: Optional[GeminiConfig] = None):
        self.config = config or self._from_env()
        from google import genai
        self._client = genai.Client(
            vertexai=True,
            project=self.config.project_id,
            location=self.config.location,
        )
        self._last_citations: List[Dict[str, str]] = []

    def _from_env(self) -> GeminiConfig:
        project_id = (
            os.getenv("GCP_PROJECT_ID")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or ""
        ).strip()
        if not project_id:
            raise ValueError("Missing GCP_PROJECT_ID")
        return GeminiConfig(
            model=os.getenv("GEMINI_MODEL") or "gemini-2.5-flash",
            use_web=os.getenv("GEMINI_USE_WEB", "1") == "1",
            temperature=float(os.getenv("AI_TEMPERATURE", "0.3")),
            thinking_budget=int(os.getenv("GEMINI_THINKING_BUDGET", "1024")),
            project_id=project_id,
            location=(
                os.getenv("GCP_LOCATION")
                or os.getenv("GOOGLE_CLOUD_LOCATION")
                or "us-central1"
            ).strip(),
        )

    @classmethod
    def _model_supports_thinking(cls, model: str) -> bool:
        m = (model or "").lower()
        if m in cls._thinking_unsupported_models:
            return False
        if "lite" in m:
            return False
        if "gemini-3" in m or "gemini-2.5" in m:
            return True
        return False

    @classmethod
    def _blacklist_thinking(cls, model: str) -> None:
        cls._thinking_unsupported_models.add((model or "").lower())

    @staticmethod
    def _is_thinking_unsupported_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return ("thinking" in text) and ("not supported" in text or "unsupported" in text)

    @staticmethod
    def _is_http_url(value: Any) -> bool:
        return isinstance(value, str) and re.match(r"^https?://", value.strip(), re.IGNORECASE) is not None

    @classmethod
    def _extract_citations_from_payload(cls, payload: Any) -> List[Dict[str, str]]:
        if payload is None:
            return []

        root: Any = payload
        if not isinstance(root, (dict, list)):
            if hasattr(root, "model_dump"):
                try:
                    root = root.model_dump(mode="python")
                except Exception:
                    root = payload
            elif hasattr(root, "to_dict"):
                try:
                    root = root.to_dict()
                except Exception:
                    root = payload

        citations: List[Dict[str, str]] = []
        seen_urls: set[str] = set()

        def _add(url: str, title: Optional[str] = None) -> None:
            normalized = (url or "").strip()
            if not normalized or normalized in seen_urls:
                return
            seen_urls.add(normalized)
            entry: Dict[str, str] = {"url": normalized}
            if title and title.strip():
                entry["title"] = title.strip()
            citations.append(entry)

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                url_value = None
                for key in ("url", "uri", "link", "href"):
                    candidate = node.get(key)
                    if cls._is_http_url(candidate):
                        url_value = str(candidate).strip()
                        break
                if url_value:
                    title_value = None
                    for title_key in ("title", "name", "display_name", "source"):
                        t = node.get(title_key)
                        if isinstance(t, str) and t.strip():
                            title_value = t
                            break
                    _add(url_value, title_value)

                for value in node.values():
                    _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(root)
        return citations

    def pop_last_citations(self) -> List[Dict[str, str]]:
        out = list(self._last_citations)
        self._last_citations = []
        return out

    def _build_config(
        self,
        *,
        use_web: bool,
        model: Optional[str] = None,
        disable_thinking: bool = False,
        system_instruction: Optional[str] = None,
    ):
        from google.genai import types

        resolved_model = model or self.config.model
        tools = [types.Tool(googleSearch=types.GoogleSearch())] if use_web else None
        thinking_config = None

        if (not disable_thinking) and self._model_supports_thinking(resolved_model):
            try:
                thinking_config = types.ThinkingConfig(
                    thinking_budget=self.config.thinking_budget,
                )
            except Exception:
                thinking_config = None

        config_kwargs: Dict[str, Any] = {
            "thinking_config": thinking_config,
            "tools": tools,
            "temperature": self.config.temperature,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        return types.GenerateContentConfig(**config_kwargs)

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
        from google.genai import types

        model = model_override or self.config.model
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])]
        try:
            resp = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=self._build_config(use_web=use_web, model=model, system_instruction=system_prompt),
            )
        except Exception as exc:
            if not self._is_thinking_unsupported_error(exc):
                raise
            self._blacklist_thinking(model)
            _trace_warning(
                "gemini.generate.retry_without_thinking model=%s (blacklisted)",
                model,
            )
            resp = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=self._build_config(use_web=use_web, model=model, disable_thinking=True, system_instruction=system_prompt),
            )
        self._last_citations = self._extract_citations_from_payload(resp)
        return getattr(resp, "text", None) or str(resp)

    async def quick_generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_override: Optional[str] = None,
        timeout_s: float = 3.5,
    ) -> str:
        started = time.perf_counter()
        chosen_model = model_override or self.config.model
        _trace_info("gemini.quick.start model=%s prompt_len=%s", chosen_model, len(user_prompt or ""))
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(
                    self._sync_generate_text,
                    system_prompt,
                    user_prompt,
                    False,
                    chosen_model,
                ),
                timeout=timeout_s,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _trace_info("gemini.quick.done elapsed_ms=%s chars=%s", elapsed_ms, len(text or ""))
            return (text or "").strip()
        except asyncio.TimeoutError:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            _trace_warning("gemini.quick.timeout elapsed_ms=%s", elapsed_ms)
            return ""
        except Exception:
            _trace_exception("gemini.quick.error")
            return ""

    async def stream_answer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        allow_web_search: Optional[bool] = None,
    ) -> AsyncIterator[str]:
        use_web = self.config.use_web if allow_web_search is None else bool(allow_web_search)
        self._last_citations = []
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
            from google.genai import types

            contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])]
            disable_thinking = False
            citations: List[Dict[str, str]] = []
            citation_urls: set[str] = set()
            for attempt in (1, 2):
                try:
                    stream = self._client.models.generate_content_stream(
                        model=self.config.model,
                        contents=contents,
                        config=self._build_config(
                            use_web=use_web,
                            disable_thinking=disable_thinking,
                            system_instruction=system_prompt,
                        ),
                    )
                    emitted = 0
                    for chunk in stream:
                        for item in self._extract_citations_from_payload(chunk):
                            url = item.get("url", "")
                            if not url or url in citation_urls:
                                continue
                            citation_urls.add(url)
                            citations.append(item)
                        text = getattr(chunk, "text", None)
                        if text:
                            emitted += 1
                            worker_state["chunks_emitted"] = int(worker_state["chunks_emitted"]) + 1
                            loop.call_soon_threadsafe(q.put_nowait, text)
                    _trace_info("gemini.stream.worker.done chunks=%s attempt=%s", emitted, attempt)
                    worker_state["error"] = None
                    break
                except Exception as exc:
                    if self._is_thinking_unsupported_error(exc) and not disable_thinking:
                        disable_thinking = True
                        self._blacklist_thinking(self.config.model)
                        _trace_warning(
                            "gemini.stream.worker.retry_without_thinking attempt=%s model=%s (blacklisted)",
                            attempt,
                            self.config.model,
                        )
                        continue
                    worker_state["error"] = exc
                    chunks_emitted = int(worker_state["chunks_emitted"])
                    if chunks_emitted > 0:
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
            self._last_citations = citations
            loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=_worker, daemon=True).start()

        emitted_tokens = 0
        while True:
            item = await q.get()
            if item is None:
                break
            emitted_tokens += 1
            yield item

        if worker_state["error"] is not None and emitted_tokens == 0:
            fallback_timeout_s = float(os.getenv("GEMINI_SYNC_FALLBACK_TIMEOUT_S", "12"))
            _trace_warning("gemini.stream.fallback.start timeout_s=%s", fallback_timeout_s)
            try:
                fallback = await asyncio.wait_for(
                    asyncio.to_thread(self._sync_generate_text, system_prompt, user_prompt, use_web),
                    timeout=fallback_timeout_s,
                )
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


# ── Singleton ────────────────────────────────────────────────────────────

_shared_client: Optional[GeminiStreamClient] = None
_shared_client_lock = threading.Lock()


def get_shared_gemini_client() -> GeminiStreamClient:
    global _shared_client
    if _shared_client is not None:
        return _shared_client
    with _shared_client_lock:
        if _shared_client is None:
            _shared_client = GeminiStreamClient()
    return _shared_client
