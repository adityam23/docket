"""Select a provider from settings. All three branches build the SAME
OpenAICompatProvider with a different base_url/key (parameterise, don't fork)."""

from __future__ import annotations

from ..config import Provider as P
from ..config import Settings
from .base import Provider
from .openai_compat import OpenAICompatProvider
from .presets import CEREBRAS, GROQ


def get_provider(settings: Settings) -> Provider:
    if settings.provider == P.local:
        return OpenAICompatProvider(
            name="local",
            base_url=settings.backend_url,
            chat_model=settings.chat_model,
            embed_url=settings.embed_url,
            embed_model=settings.embed_model,
            rerank_url=settings.rerank_url,
            rerank_model=settings.rerank_model,
            timeout_s=settings.request_timeout_s,
        )
    if settings.provider == P.cerebras:
        return OpenAICompatProvider(
            name="cerebras",
            base_url=CEREBRAS["base_url"],
            chat_model=settings.chat_model or CEREBRAS["default_model"],
            api_key=settings.cerebras_api_key,
            timeout_s=settings.request_timeout_s,
        )
    if settings.provider == P.groq:
        return OpenAICompatProvider(
            name="groq",
            base_url=GROQ["base_url"],
            chat_model=settings.chat_model or GROQ["default_model"],
            api_key=settings.groq_api_key,
            timeout_s=settings.request_timeout_s,
        )
    raise ValueError(f"unknown provider: {settings.provider}")
