# services/llm/llm_service.py
from __future__ import annotations

import os
import json
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Literal


# ============================================================================
# PUBLIC INTERFACE
# ============================================================================

class LLMClient(Protocol):
    async def generate_json(self, *, system: str, user: str) -> str:
        """Return raw text that should be JSON."""


@dataclass
class LLMConfig:
    provider: str = "gemini"  # openai | anthropic | gemini | cloud
    temperature: float = 0.3

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_timeout_s: float = 60.0

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-latest"
    anthropic_timeout_s: float = 60.0

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_use_web: bool = True
    gemini_thinking_level: str = "HIGH"  # auto-normalized inside client

    # Cloud gateway (optional)
    cloud_base_url: str = ""
    cloud_api_key: str = ""
    cloud_timeout_s: float = 60.0

    @staticmethod
    def from_env() -> "LLMConfig":
        provider = (os.getenv("AI_PROVIDER") or "gemini").lower()
        return LLMConfig(
            provider=provider,
            temperature=float(os.getenv("AI_TEMPERATURE", "0.3")),

            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
            openai_timeout_s=float(os.getenv("OPENAI_TIMEOUT_S", "60")),

            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            anthropic_model=os.getenv("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest",
            anthropic_timeout_s=float(os.getenv("ANTHROPIC_TIMEOUT_S", "60")),

            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL") or "gemini-2.5-flash",
            gemini_use_web=(os.getenv("GEMINI_USE_WEB", "1") == "1"),
            gemini_thinking_level=(os.getenv("GEMINI_THINKING_LEVEL") or "HIGH").upper(),

            cloud_base_url=os.getenv("CLOUD_LLM_BASE_URL", ""),
            cloud_api_key=os.getenv("CLOUD_LLM_API_KEY", ""),
            cloud_timeout_s=float(os.getenv("CLOUD_LLM_TIMEOUT_S", "60")),
        )


# ============================================================================
# PROVIDER CLIENTS
# ============================================================================

class OpenAIClient:
    def __init__(self, api_key: str, model: str, temperature: float, timeout_s: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s

    async def generate_json(self, *, system: str, user: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": self.temperature,
                    "response_format": {"type": "json_object"},
                },
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]


class AnthropicClient:
    def __init__(self, api_key: str, model: str, temperature: float, timeout_s: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s

    async def generate_json(self, *, system: str, user: str) -> str:
        import httpx
        # Note: Anthropic "temperature" is supported, but some models ignore it.
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.model,
                    "max_tokens": 2000,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "temperature": self.temperature,
                },
            )
            r.raise_for_status()
            data = r.json()
            return data["content"][0]["text"]


GeminiThinking = Literal["LOW", "MEDIUM", "HIGH"]

class GeminiClient:
    def __init__(
        self,
        model: str,
        *,
        use_web: bool = True,
        thinking_level: GeminiThinking = "HIGH",
        temperature: float = 0.3,
    ):
        self.model = model
        self.use_web = use_web
        self.thinking_level = thinking_level
        self.temperature = temperature

    def _normalize_thinking(self) -> str:
        lvl = (self.thinking_level or "HIGH").upper()
        m = (self.model or "").lower()

        # pro: LOW/HIGH only (per your note)
        if "pro" in m:
            return lvl if lvl in ("LOW", "HIGH") else "HIGH"
        # flash: LOW/MEDIUM/HIGH
        if "flash" in m:
            return lvl if lvl in ("LOW", "MEDIUM", "HIGH") else "MEDIUM"
        return lvl if lvl in ("LOW", "HIGH") else "HIGH"

    async def generate_json(self, *, system: str, user: str) -> str:
        # google-genai SDK is sync-ish; run in thread.
        return await asyncio.to_thread(self._sync_call, system, user)

    def _sync_call(self, system: str, user: str) -> str:
        from google import genai
        from google.genai import types
        import os

        client = genai.Client(
            vertexai=True,
            project=os.getenv("GCP_PROJECT_ID"),
            location="us-central1",  # important
        )

        combined = f"{system}\n\n{user}"
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=combined)]
            )
        ]

        tools = None
        if self.use_web:
            tools = [types.Tool(googleSearch=types.GoogleSearch())]

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=self._normalize_thinking()
            ),
            tools=tools,
            temperature=self.temperature,
        )

        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            if "thinking_level" in str(e):
                config = types.GenerateContentConfig(
                    tools=tools,
                    temperature=self.temperature,
                )
                resp = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            else:
                raise

        return resp.text if getattr(resp, "text", None) else str(resp)



class CloudLLMClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    async def generate_json(self, *, system: str, user: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(
                f"{self.base_url}/v1/generate",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"system": system, "user": user},
            )
            r.raise_for_status()
            data = r.json()
            return data["text"]


# ============================================================================
# LLM SERVICE (reusable everywhere)
# ============================================================================

class LLMService:
    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig.from_env()
        self.client: LLMClient = self._resolve_client(self.cfg)

    def _resolve_client(self, cfg: LLMConfig) -> LLMClient:
        p = (cfg.provider or "gemini").lower()

        if p == "openai":
            if not cfg.openai_api_key:
                raise ValueError("Missing OPENAI_API_KEY")
            return OpenAIClient(
                api_key=cfg.openai_api_key,
                model=cfg.openai_model,
                temperature=cfg.temperature,
                timeout_s=cfg.openai_timeout_s,
            )

        if p == "anthropic":
            if not cfg.anthropic_api_key:
                raise ValueError("Missing ANTHROPIC_API_KEY")
            return AnthropicClient(
                api_key=cfg.anthropic_api_key,
                model=cfg.anthropic_model,
                temperature=cfg.temperature,
                timeout_s=cfg.anthropic_timeout_s,
            )

        if p == "cloud":
            if not cfg.cloud_base_url or not cfg.cloud_api_key:
                raise ValueError("Missing CLOUD_LLM_BASE_URL or CLOUD_LLM_API_KEY")
            return CloudLLMClient(
                base_url=cfg.cloud_base_url,
                api_key=cfg.cloud_api_key,
                timeout_s=cfg.cloud_timeout_s,
            )

        # default: gemini (Vertex AI — authenticates via GOOGLE_APPLICATION_CREDENTIALS)
        return GeminiClient(
            model=cfg.gemini_model,
            use_web=cfg.gemini_use_web,
            thinking_level=cfg.gemini_thinking_level,
            temperature=cfg.temperature,
        )

    # ---- helpers used by many callers ----

    @staticmethod
    def strip_code_fences(text: str) -> str:
        t = (text or "").strip()
        if t.startswith("```"):
            lines = t.split("\n")
            # drop first fence line
            lines = lines[1:]
            # drop last fence line if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t = "\n".join(lines).strip()
        return t

    def parse_json(self, text: str) -> Dict[str, Any]:
        t = self.strip_code_fences(text)
        return json.loads(t)

    async def generate_json(self, *, system: str, user: str) -> Dict[str, Any]:
        raw = await self.client.generate_json(system=system, user=user)
        return self.parse_json(raw)


# Optional: shared singleton
_llm_singleton: Optional[LLMService] = None

def get_llm_service() -> LLMService:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMService()
    return _llm_singleton
