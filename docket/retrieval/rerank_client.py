"""Cross-encoder rerank through the configured provider (the ``Retriever``
reranker seam, T26).

Mirrors ``ingest/embed.py``: a thin wrapper over the ONE ``/v1`` provider so the
reranker is reached exactly like chat/embed (one concept, one implementation —
CLAUDE.md). Requires a backend serving ``/v1/rerank`` (``DK_RERANK_URL``); when
unconfigured, callers degrade to the fused RRF order rather than fail — the
retriever does exactly that when no ``reranker`` is passed.
"""

from __future__ import annotations

from ..config import Settings, load_settings
from ..providers.router import get_provider
from .retriever import Reranker


def rerank_available(settings: Settings | None = None) -> bool:
    """True when a rerank endpoint is configured (drives whether the retriever
    reorders fused candidates with a cross-encoder or keeps the RRF order)."""
    return bool((settings or load_settings()).rerank_url)


def make_reranker(settings: Settings | None = None) -> Reranker:
    """Build the ``Reranker`` callable ``(query, docs) -> scores`` for the seam.

    One score per document, in the input order, higher = more relevant (the
    provider scatters the endpoint's sorted results back to request order)."""
    s = settings or load_settings()
    provider = get_provider(s)
    return lambda query, docs: provider.rerank(query, docs)
