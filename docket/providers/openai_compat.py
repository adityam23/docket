"""THE OpenAI-/v1 client, pointed at mettle — the only supported engine. Chat,
embeddings, rerank, OCR and the collection surface (search / document ingest /
collection lifecycle) all ride this one base URL; parameterise, never fork
(CLAUDE.md 'Reusability')."""

from __future__ import annotations

import re
from typing import Callable
from urllib.parse import quote

import httpx

from ..auth import bearer_headers
from .base import ChatResult, SlotUnavailable

# Control/NUL bytes (keep \t \n \r) — PDF text extraction (pypdf) leaves stray
# NULs in real filings, and the backend rejects them ("nul byte found in provided
# data") with a 500 on /v1/embeddings and /v1/rerank. Strip them at the single
# point every backend text request flows through, so no caller has to remember to.
_CTRL_BYTES = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(text: str) -> str:
    return _CTRL_BYTES.sub(" ", text)


def _path(*segments: str) -> str:
    """Join URL path segments, percent-encoding each one individually (``safe=""``,
    so a literal ``/``, ``#`` or ``?`` inside a segment gets encoded too) — the
    ONE place every collection/document call builds its path. Without this, a
    chunk id like ``acme#3`` gets treated by httpx as a URL fragment (DELETE
    silently hits the wrong path), and a hostile collection name like
    ``../../engine/x`` can escape the collections route onto another engine
    route carrying the same bearer token."""
    return "/".join(quote(seg, safe="") for seg in segments)


def _extract_logprobs(choice: dict) -> list[float] | None:
    """Pull per-token top-choice logprobs out of an OpenAI-/v1 chat choice.
    Returns None when the backend didn't emit any (→ trust label 'unknown')."""
    lp = choice.get("logprobs")
    if not isinstance(lp, dict):
        return None
    content = lp.get("content")
    if not content:
        return None
    out = [tok["logprob"] for tok in content if tok.get("logprob") is not None]
    return out or None


def _raise_on_slot_unavailable(r: httpx.Response) -> None:
    """A structured 503 from the engine: a model slot was dropped/shrunk by the
    active profile (VRAM pressure / profile switch). Raise the structured
    exception the service layer turns into a helpful answer; any other status
    error is left for ``raise_for_status``."""
    if r.status_code != 503:
        return
    try:
        err = r.json().get("error", {})
    except ValueError:
        return
    if isinstance(err, dict) and err.get("code") == "slot_unavailable":
        raise SlotUnavailable(
            reason=err.get("reason", "unknown"),
            missing=err.get("missing") or [],
            profile_degraded=bool(err.get("profile_degraded")),
        )


class OpenAICompatProvider:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        chat_model: str,
        api_key: str | None = None,
        embed_model: str | None = None,
        rerank_model: str | None = None,
        timeout_s: float = 120.0,
        resolve_model: Callable[[str], str] | None = None,
    ) -> None:
        self.name = name
        self._base = base_url.rstrip("/")
        self._chat_model = chat_model
        self._embed_model = embed_model
        self._rerank_model = rerank_model
        self._resolve = resolve_model or (lambda m: m)
        self._client = httpx.Client(timeout=timeout_s, headers=bearer_headers(api_key))

    def _url(self, *segments: str) -> str:
        return f"{self._base}/{_path(*segments)}"

    def _send(
        self,
        method: str,
        *segments: str,
        json: dict | None = None,
        ok: tuple[int, ...] = (),
    ) -> httpx.Response:
        """The ONE request path for every /v1 call: build the (encoded) URL,
        send, check for the structured slot-unavailable 503, then raise on any
        other error status -- unless the caller declared it expected via ``ok``
        (e.g. 404 for an idempotent delete, 409 for create-if-absent), in which
        case the response is handed back as-is for the caller to interpret."""
        r = self._client.request(method, self._url(*segments), json=json)
        _raise_on_slot_unavailable(r)
        if r.status_code in ok:
            return r
        r.raise_for_status()
        return r

    def health(self) -> dict:
        r = self._send("GET", "models")
        data = r.json()
        rows = data.get("data") or data.get("models") or []
        ids = [row.get("id") or row.get("name") for row in rows]
        return {"base_url": self._base, "models": ids}

    def chat(self, messages: list[dict], **kw) -> ChatResult:
        body: dict = {"model": self._resolve(self._chat_model), "messages": messages,
                      "stream": False}
        # Always request per-token logprobs so the Tier-1 trust layer has a
        # real signal to score (feeds trust/reliability) -- mettle is the only
        # backend and always supports this, so there is no capability to check.
        body.setdefault("logprobs", True)
        body.update(kw)
        r = self._send("POST", "chat", "completions", json=body)
        data = r.json()
        choice = data["choices"][0]
        # mettle guarantees `message.content` is always a string (""  when a
        # reasoning model hits max_tokens with finish_reason "length"); `or ""`
        # is defence-in-depth only, not expected to ever trigger.
        text = choice["message"]["content"] or ""
        return ChatResult(text=text, logprobs=_extract_logprobs(choice), raw=data)

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self._send(
            "POST", "embeddings",
            json={"model": self._resolve(self._embed_model),
                  "input": [_sanitize(t) for t in texts]},
        )
        return [row["embedding"] for row in r.json()["data"]]

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score each document against the query via ``/v1/rerank`` (Cohere shape).

        The endpoint returns results **sorted by score** carrying the ``index``
        into the request; the retriever's ``Reranker`` seam expects one score per
        document in the ORIGINAL order, so we scatter each ``relevance_score`` back
        to its ``index``. Only rank order is used downstream, so the raw
        (query-dependent, uncalibrated) score is fine — no normalisation. Empty
        input short-circuits (the backend returns ``results: []`` for it anyway)."""
        if not documents:
            return []
        r = self._send(
            "POST", "rerank",
            json={"model": self._resolve(self._rerank_model), "query": _sanitize(query),
                  "documents": [_sanitize(d) for d in documents]},
        )
        scores = [0.0] * len(documents)
        for row in r.json().get("results", []):
            idx = row.get("index")
            if isinstance(idx, int) and 0 <= idx < len(documents):
                scores[idx] = row.get("relevance_score", 0.0)
        return scores

    def ocr(self, image_b64: str, instruction: str | None = None) -> str:
        """Scanned-page OCR via mettle's ``/v1/ocr`` -- the same shared client,
        bearer auth and configured timeout as every other /v1 call (no bare,
        unauthenticated ``httpx.post`` off to the side). Returns the raw text;
        callers decide what an empty result means."""
        body: dict = {"image": image_b64}
        if instruction is not None:
            body["instruction"] = instruction
        r = self._send("POST", "ocr", json=body)
        return r.json().get("text") or ""

    # --- collection surface (mettle /v1/collections; retrieval lives engine-side)

    def ensure_collection(self, collection: str) -> None:
        """Create the collection if absent. 409 (already exists) is success —
        the collection's embedder/dim are the engine's defaults, not ours."""
        self._send(
            "POST", "collections",
            json={"name": collection, "embedding_model": "default"},
            ok=(409,),
        )

    def delete_collection(self, collection: str) -> bool:
        r = self._send("DELETE", "collections", collection, ok=(404,))
        return r.status_code != 404

    def post_document(
        self, collection: str, doc_id: str, text: str, metadata: dict[str, str] | None = None
    ) -> list[str]:
        """POST one docket chunk as an engine document. Returns the engine's
        chunk ids (one per posted document under the engine's own chunker)."""
        r = self._send(
            "POST", "collections", collection, "documents",
            json={"id": doc_id, "content": _sanitize(text), "metadata": metadata or {}},
        )
        return r.json().get("chunk_ids", [])

    def delete_document(self, collection: str, doc_id: str) -> bool:
        r = self._send("DELETE", "collections", collection, "documents", doc_id, ok=(404,))
        return r.status_code != 404

    def collection_search(
        self,
        collection: str,
        query: str,
        *,
        mode: str = "hybrid",
        k: int = 10,
        allowlist: list[str] | None = None,
    ) -> list[dict]:
        """Search a collection engine-side. ``mode`` is dense | keyword | hybrid —
        anything else is a client bug and raises LOUDLY before the wire (no
        silent fallback to another mode: a wrong-number answer is worse than an
        error). Response rows are ``ScoredChunk``: chunk_id, document_id, text,
        score (desc), optional source_span."""
        if mode not in ("dense", "keyword", "hybrid"):
            raise ValueError(f"unknown search mode: {mode!r}")
        body: dict = {"query": query, "k": k, "mode": mode}
        if allowlist:
            body["allowlist"] = allowlist
        r = self._send("POST", "collections", collection, "search", json=body)
        return r.json()
