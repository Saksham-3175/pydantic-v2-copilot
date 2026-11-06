"""Unified Multi-Provider LLM Client with automatic failover (Groq primary, Gemini secondary)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import typing
from pydantic import BaseModel

logger = logging.getLogger(__name__)

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass

T = typing.TypeVar("T", bound=BaseModel)


def _format_clean_error(provider: str, exc: Exception) -> str:
    err_str = str(exc)
    if "429" in err_str or "rate_limit" in err_str.lower():
        return f"{provider} rate limit reached (429)"
    elif "503" in err_str or "unavailable" in err_str.lower():
        return f"{provider} service unavailable (503)"
    elif "TimeoutError" in type(exc).__name__ or "timeout" in err_str.lower():
        return f"{provider} request timed out (>5s)"
    elif "connection" in err_str.lower() or "connect" in err_str.lower() or "name resolution" in err_str.lower():
        return f"{provider} connection offline"
    return f"{provider} API unavailable"


class UnifiedLLMClient:
    """Manages LLM generation with primary Groq (qwen/qwen3.8-27b) and automatic Gemini fallback."""

    def __init__(
        self,
        groq_api_key: typing.Optional[str] = None,
        gemini_api_key: typing.Optional[str] = None,
        primary_model: str = "qwen/qwen3.8-27b",
        secondary_model: str = "gemini-2.5-flash",
    ) -> None:
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.primary_model = primary_model
        self.secondary_model = secondary_model

        self._groq_client: typing.Any = None
        if self.groq_api_key:
            try:
                from groq import AsyncGroq

                self._groq_client = AsyncGroq(api_key=self.groq_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")

        self._gemini_client: typing.Any = None
        if self.gemini_api_key:
            try:
                from google import genai

                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")

    @property
    def groq_client(self) -> typing.Any:
        return self._groq_client

    @groq_client.setter
    def groq_client(self, client: typing.Any) -> None:
        self._groq_client = client

    @property
    def gemini_client(self) -> typing.Any:
        return self._gemini_client

    @gemini_client.setter
    def gemini_client(self, client: typing.Any) -> None:
        self._gemini_client = client

    async def generate_structured(
        self,
        prompt: str,
        schema: typing.Type[T],
        system_instruction: typing.Optional[str] = None,
    ) -> typing.Tuple[T, str]:
        """Generate structured Pydantic model with automatic cross-provider failover.

        Tries Primary (Groq) first. Upon failure/rate-limit, falls back to Secondary (Gemini).
        Returns tuple of (validated_model_instance, provider_name_used).
        """
        errors: list[str] = []

        # 1. Primary: Groq API
        if self._groq_client and (self.groq_api_key or os.environ.get("GROQ_API_KEY")):
            try:
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})

                schema_json = json.dumps(schema.model_json_schema())
                user_msg = (
                    f"{prompt}\n\nYou MUST respond with a valid JSON object matching this schema:\n{schema_json}"
                )
                messages.append({"role": "user", "content": user_msg})

                resp = await asyncio.wait_for(
                    self._groq_client.chat.completions.create(
                        model=self.primary_model,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=0.1,
                    ),
                    timeout=5.0,
                )
                content = resp.choices[0].message.content or "{}"
                return schema.model_validate_json(content), "groq"
            except Exception as e:
                clean_msg = _format_clean_error("Groq", e)
                print(f"\033[2minfo: {clean_msg}, failing over to Gemini...\033[0m")
                errors.append(clean_msg)

        # 2. Secondary Fallback: Gemini API
        if self._gemini_client and (self.gemini_api_key or os.environ.get("GEMINI_API_KEY")):
            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1,
                )
                if system_instruction:
                    config.system_instruction = system_instruction

                resp = await asyncio.wait_for(
                    self._gemini_client.aio.models.generate_content(
                        model=self.secondary_model,
                        contents=prompt,
                        config=config,
                    ),
                    timeout=5.0,
                )
                if resp.text:
                    return schema.model_validate_json(resp.text), "gemini"
            except Exception as e:
                clean_msg = _format_clean_error("Gemini", e)
                print(f"\033[2minfo: {clean_msg}, failing over to local AST engine...\033[0m")
                errors.append(clean_msg)

        raise RuntimeError(f"All LLM providers failed or unconfigured: {'; '.join(errors)}")
