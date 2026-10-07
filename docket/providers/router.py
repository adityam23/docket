"""Build the ONE provider: mettle's /v1. mettle is the only supported engine
(cutover contract, 2026-09-10) — parameterise the client, never fork it.

Every call is also wrapped with the engine's phase coordinator (residency
contract D5/D6, see ``docket/phase.py``): generation enters CHAT; every
retrieval/ingest call (embed, rerank, collection search, document post/
delete, collection ensure/delete) enters RETRIEVAL. This is the ONE place
that wrapping happens (CLAUDE.md reusability) -- every caller through
``get_provider`` gets phase-correctness and ingest-preemption at chunk
granularity for free, with no call site responsible for remembering to ask
for it. ``health``/``ocr`` enter no phase: health must answer regardless of
the active profile (it's a liveness probe), and OCR's slot isn't pinned in
either phase profile today -- an open question posed to mettle (see
``QUESTIONS from mettle.md``), not something docket can resolve unilaterally."""

from __future__ import annotations

from ..config import Settings
from ..engine import resolve_model
from ..phase import Phase, PhaseCoordinator, get_phase_coordinator
from .base import Provider
from .openai_compat import OpenAICompatProvider


class PhasedProvider:
    """Wrap a :class:`Provider` so every call enters the engine phase it
    needs (via ``coordinator.enter``) before running. Anything not listed
    below (``name``, future provider methods) delegates straight to the
    inner provider -- same transparent-delegation shape as benchmarks/
    financebench's ``_ResilientProvider`` -- so a provider attribute is never
    silently hidden by the wrapper, it just isn't phase-gated until it is
    explicitly added here."""

    def __init__(self, inner: Provider, coordinator: PhaseCoordinator) -> None:
        self._inner = inner
        self._coordinator = coordinator

    def __getattr__(self, key):
        return getattr(self._inner, key)

    def phase(self, phase: Phase):
        """Hold ``phase`` open across SEVERAL engine calls instead of paying
        a switch between each one (D5 contention fix): a thread already
        holding a phase open re-enters it for free (same-thread reentrancy,
        see ``docket/phase.py``), so e.g. ``Retriever.retrieve``'s search+
        rerank pair, or the agent's per-hop chat-answer+refine pair, becomes
        ONE held block instead of releasing and re-acquiring (and risking a
        competing thread's phase winning the gap) between the two calls.
        Call via ``docket.phase.provider_phase(provider, phase)``, which
        no-ops for a provider that doesn't implement this (test doubles)."""
        return self._coordinator.enter(phase)

    def health(self) -> dict:
        return self._inner.health()

    def ocr(self, image_b64: str, instruction: str | None = None) -> str:
        return self._inner.ocr(image_b64, instruction)

    def chat(self, messages: list[dict], **kw):
        with self._coordinator.enter(Phase.CHAT):
            return self._inner.chat(messages, **kw)

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._coordinator.enter(Phase.RETRIEVAL):
            return self._inner.embed(texts)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        with self._coordinator.enter(Phase.RETRIEVAL):
            return self._inner.rerank(query, documents)

    def ensure_collection(self, collection: str) -> None:
        with self._coordinator.enter(Phase.RETRIEVAL):
            return self._inner.ensure_collection(collection)

    def delete_collection(self, collection: str) -> bool:
        with self._coordinator.enter(Phase.RETRIEVAL):
            return self._inner.delete_collection(collection)

    def post_document(
        self, collection: str, doc_id: str, text: str, metadata: dict[str, str] | None = None
    ) -> list[str]:
        with self._coordinator.enter(Phase.RETRIEVAL):
            return self._inner.post_document(collection, doc_id, text, metadata)

    def delete_document(self, collection: str, doc_id: str) -> bool:
        with self._coordinator.enter(Phase.RETRIEVAL):
            return self._inner.delete_document(collection, doc_id)

    def collection_search(
        self,
        collection: str,
        query: str,
        *,
        mode: str = "hybrid",
        k: int = 10,
        allowlist: list[str] | None = None,
    ) -> list[dict]:
        with self._coordinator.enter(Phase.RETRIEVAL):
            return self._inner.collection_search(
                collection, query, mode=mode, k=k, allowlist=allowlist
            )


def get_provider(settings: Settings) -> Provider:
    inner = OpenAICompatProvider(
        name="mettle",
        base_url=settings.backend_url,
        chat_model=settings.chat_model,
        embed_model=settings.embed_model,
        rerank_model=settings.rerank_model,
        api_key=settings.engine_key,
        timeout_s=settings.request_timeout_s,
        resolve_model=lambda m: resolve_model(settings, m),
    )
    return PhasedProvider(inner, get_phase_coordinator(settings))
