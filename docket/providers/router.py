"""Build the ONE provider: mettle's /v1. mettle is the only supported engine
(cutover contract, 2026-09-10) — parameterise the client, never fork it."""

from __future__ import annotations

from ..config import Settings
from ..engine import resolve_model
from .base import Provider
from .openai_compat import OpenAICompatProvider


def get_provider(settings: Settings) -> Provider:
    return OpenAICompatProvider(
        name="mettle",
        base_url=settings.backend_url,
        chat_model=settings.chat_model,
        embed_model=settings.embed_model,
        rerank_model=settings.rerank_model,
        api_key=settings.engine_key,
        timeout_s=settings.request_timeout_s,
        resolve_model=lambda m: resolve_model(settings, m),
    )
