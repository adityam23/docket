"""Cross-encoder rerank through the ONE provider (the ``Retriever`` reranker
seam, T26). A thin wrapper so the reranker is reached exactly like chat/embed —
mettle serves ``/v1/rerank`` unconditionally on the same base (cutover
contract); an engine without it fails loudly, like every engine feature."""

from __future__ import annotations

from ..config import Settings, load_settings
from ..providers.router import get_provider


def make_reranker(settings: Settings | None = None):
    """Build the ``Reranker`` callable ``(query, docs) -> scores`` for the seam.

    One score per document, in the input order, higher = more relevant (the
    provider scatters the endpoint's sorted results back to request order)."""
    s = settings or load_settings()
    provider = get_provider(s)
    return lambda query, docs: provider.rerank(query, docs)
