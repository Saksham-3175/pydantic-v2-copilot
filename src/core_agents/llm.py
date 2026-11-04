"""Unified Multi-Provider LLM Client with automatic failover (Groq primary, Gemini secondary)."""

from __future__ import annotations

import json
import logging
import os
import typing
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = typing.TypeVar("T", bound=BaseModel)


class UnifiedLLMClient:
    """Manages LLM generation with primary Groq (qwen/qwen3.8-27b) and automatic Gemini fallback."""

    def __init__(
        self,
        groq_api_key: typing.Optional[str] = None,
        gemini_api_key: typing.Optional[str] = None,
        primary_model: str = "qwen/qwen3.8-27b",
        secondary_model: str = "gemini-3.8-flash",
    ) -> None:
        groq_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        gemini_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not (groq_key and gemini_key):
            try:
                from dotenv import find_dotenv, load_dotenv

                load_dotenv(find_dotenv(usecwd=True))
                groq_key = groq_key or os.environ.get("GROQ_API_KEY")
                gemini_key = gemini_key or os.environ.get("GEMINI_API_KEY")
            except Exception:
                pass

        self.groq_api_key = groq_key
        self.gemini_api_key = gemini_key
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

                resp = await self._groq_client.chat.completions.create(
                    model=self.primary_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                content = resp.choices[0].message.content or "{}"
                return schema.model_validate_json(content), "groq"
            except Exception as e:
                logger.warning(f"Primary provider (Groq) failed: {e}. Falling back to Gemini...")
                errors.append(f"Groq error: {e}")

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

                resp = await self._gemini_client.aio.models.generate_content(
                    model=self.secondary_model,
                    contents=prompt,
                    config=config,
                )
                if resp.text:
                    return schema.model_validate_json(resp.text), "gemini"
            except Exception as e:
                logger.warning(f"Secondary provider (Gemini) failed: {e}.")
                errors.append(f"Gemini error: {e}")

        raise RuntimeError(f"All LLM providers failed or unconfigured: {'; '.join(errors)}")
