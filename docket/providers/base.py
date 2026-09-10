"""Provider contract + capability tiers. Capabilities drive graceful degradation
of the trust layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, auto
from typing import Protocol, runtime_checkable


class Capability(Flag):
    NONE = 0
    CHAT = auto()
    EMBED = auto()
    RERANK = auto()         # cross-encoder /v1/rerank -> retrieval reordering (T26)
    LOGPROBS = auto()       # token logprobs   -> Tier-1 trust score (entropy)
    HIDDEN_STATES = auto()  # white-box states  -> Tier-2 probe (roadmap)
    SAE = auto()            # Gemma Scope feats -> Tier-3 lab mode (roadmap)


@dataclass
class ChatResult:
    text: str
    logprobs: list[float] | None = None
    raw: dict | None = None


class SlotUnavailable(Exception):
    """A requested model slot is not resident — the engine's active profile
    dropped or shrank it (``slot_unavailable`` 503). Carries the structured
    payload so callers surface the reason and offer a profile switch, instead of
    a bare HTTP error. See ``docket/engine.py`` (mettle `/engine/*` consumer)."""
    def __init__(
        self,
        *,
        reason: str = "vram",
        missing: list[str] | None = None,
        profile_degraded: bool = False,
        status_code: int = 503,
    ) -> None:
        self.reason = reason
        self.missing = missing or []
        self.profile_degraded = profile_degraded
        self.status_code = status_code
        super().__init__(f"slot unavailable ({reason}): {', '.join(self.missing)}")


@runtime_checkable
class Provider(Protocol):
    name: str
    capabilities: Capability

    def health(self) -> dict: ...
    def chat(self, messages: list[dict], **kw) -> ChatResult: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...
