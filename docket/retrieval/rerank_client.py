"""Cross-encoder rerank through the ONE provider (the ``Retriever`` reranker
seam, T26). A thin wrapper so the reranker is reached exactly like chat/embed —
mettle serves ``/v1/rerank`` unconditionally on the same base (cutover
contract); an engine without it fails loudly, like every engine feature."""

from __future__ import annotations

from ..providers.base import Provider


def make_reranker(provider: Provider):
    """Build the ``Reranker`` callable ``(query, docs) -> scores`` for the seam.

    One score per document, in the input order, higher = more relevant (the
    provider scatters the endpoint's sorted results back to request order).
    Takes the ALREADY-BUILT provider rather than ``Settings`` — one provider
    per retriever (``load_retriever`` previously resolved it twice: once here,
    once for the retriever itself)."""
    return lambda query, docs: provider.rerank(query, docs)
