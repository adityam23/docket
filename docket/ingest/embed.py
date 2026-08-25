"""Embed text through the configured provider.

Requires a DEDICATED embedding endpoint (``DK_EMBED_URL``) — the shared chat
server is not an embedding server (docs/architecture.md). When unconfigured,
callers should degrade to sparse-only retrieval rather than fail; the retriever
does exactly that.
"""

from __future__ import annotations

from ..config import Settings, load_settings
from ..providers.router import get_provider


def embed_texts(texts: list[str], *, batch: int = 64, settings: Settings | None = None) -> list[list[float]]:
    """Embed a list of texts, batched to keep request bodies reasonable."""
    provider = get_provider(settings or load_settings())
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        out.extend(provider.embed(texts[i : i + batch]))
    return out


def embeddings_available(settings: Settings | None = None) -> bool:
    """True when a dedicated embedding endpoint is configured (drives whether
    retrieval runs hybrid dense+sparse or degrades to sparse-only)."""
    return bool((settings or load_settings()).embed_url)
