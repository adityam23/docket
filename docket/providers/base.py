"""Provider contract + capability tiers.

``Capability`` is roadmap bookkeeping for the trust layer's future tiers
(HIDDEN_STATES/SAE, lab mode) -- it is NOT read anywhere in the live request
path today: mettle is the only backend (cutover contract) and serves chat,
embed, rerank and logprobs unconditionally, so there is nothing to degrade
gracefully between. Keep the enum for the roadmap tiers; don't wire it back
into request-shaping until a second tier actually needs branching on it."""

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
    """The real surface callers rely on -- chat/embed/rerank for the agent and
    retrieval, health for the probe, the collection lifecycle + search for
    ingest/retrieval, and ocr for scanned-page ingest. (``name`` and the old
    ``capabilities`` attribute were plumbing nothing outside the provider
    read -- dropped; see ``Capability``'s docstring.)"""

    def health(self) -> dict: ...
    def chat(self, messages: list[dict], **kw) -> ChatResult: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...
    def ensure_collection(self, collection: str) -> None: ...
    def post_document(
        self, collection: str, doc_id: str, text: str, metadata: dict[str, str] | None = None
    ) -> list[str]: ...
    def delete_document(self, collection: str, doc_id: str) -> bool: ...
    def collection_search(
        self,
        collection: str,
        query: str,
        *,
        mode: str = "hybrid",
        k: int = 10,
        allowlist: list[str] | None = None,
    ) -> list[dict]: ...
    def ocr(self, image_b64: str, instruction: str | None = None) -> str: ...
