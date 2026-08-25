"""Provider contract + capability tiers. Capabilities drive graceful degradation
of the trust/interpretability layer (docs/decisions.md Q9)."""

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


@runtime_checkable
class Provider(Protocol):
    name: str
    capabilities: Capability

    def health(self) -> dict: ...
    def chat(self, messages: list[dict], **kw) -> ChatResult: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...
