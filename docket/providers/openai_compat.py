"""ONE OpenAI-/v1 client, parameterised by base_url / api_key / model — reused
for every backend (local llama-server / infengine / Ollama, plus Cerebras and
Groq). One concept, one implementation (see CLAUDE.md 'Reusability')."""

from __future__ import annotations

import re

import httpx

from .base import Capability, ChatResult

# Control/NUL bytes (keep \t \n \r) — PDF text extraction (pypdf) leaves stray
# NULs in real filings, and the backend rejects them ("nul byte found in provided
# data") with a 500 on /v1/embeddings and /v1/rerank. Strip them at the single
# point every backend text request flows through, so no caller has to remember to.
_CTRL_BYTES = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(text: str) -> str:
    return _CTRL_BYTES.sub(" ", text)


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


class OpenAICompatProvider:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        chat_model: str,
        api_key: str | None = None,
        embed_url: str | None = None,
        embed_model: str | None = None,
        rerank_url: str | None = None,
        rerank_model: str | None = None,
        timeout_s: float = 120.0,
        capabilities: Capability | None = None,
    ) -> None:
        self.name = name
        self.capabilities = capabilities or (Capability.CHAT | Capability.LOGPROBS)
        self._base = base_url.rstrip("/")
        self._chat_model = chat_model
        self._embed_url = (embed_url or "").rstrip("/") or None
        self._embed_model = embed_model
        self._rerank_url = (rerank_url or "").rstrip("/") or None
        self._rerank_model = rerank_model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout_s, headers=headers)
        if self._embed_url:
            self.capabilities |= Capability.EMBED
        if self._rerank_url:
            self.capabilities |= Capability.RERANK

    def health(self) -> dict:
        r = self._client.get(f"{self._base}/models")
        r.raise_for_status()
        data = r.json()
        rows = data.get("data") or data.get("models") or []
        ids = [row.get("id") or row.get("name") for row in rows]
        return {"base_url": self._base, "models": ids}

    def chat(self, messages: list[dict], **kw) -> ChatResult:
        body: dict = {"model": self._chat_model, "messages": messages, "stream": False}
        # Request per-token logprobs by default so the Tier-1 trust layer has a
        # real signal to score (feeds trust/reliability). Backends that ignore
        # the flag simply return no `logprobs` block and we degrade to unknown.
        if Capability.LOGPROBS in self.capabilities:
            body.setdefault("logprobs", True)
        body.update(kw)
        r = self._client.post(f"{self._base}/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        return ChatResult(text=text, logprobs=_extract_logprobs(choice), raw=data)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._embed_url:
            raise RuntimeError(
                "embeddings unconfigured — set DK_EMBED_URL to a dedicated "
                "`llama-server --embeddings` endpoint (see docs/architecture.md)"
            )
        r = self._client.post(
            f"{self._embed_url}/embeddings",
            json={"model": self._embed_model, "input": [_sanitize(t) for t in texts]},
        )
        r.raise_for_status()
        return [row["embedding"] for row in r.json()["data"]]

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score each document against the query via ``/v1/rerank`` (Cohere shape:
        Cerebras/Groq don't serve it — this is the all-in-one local backend path).

        The endpoint returns results **sorted by score** carrying the ``index``
        into the request; the retriever's ``Reranker`` seam expects one score per
        document in the ORIGINAL order, so we scatter each ``relevance_score`` back
        to its ``index``. Only rank order is used downstream, so the raw
        (query-dependent, uncalibrated) score is fine — no normalisation. Empty
        input short-circuits (the backend returns ``results: []`` for it anyway)."""
        if not self._rerank_url:
            raise RuntimeError(
                "rerank unconfigured — set DK_RERANK_URL to a backend serving "
                "`/v1/rerank` (see docs/architecture.md)"
            )
        if not documents:
            return []
        r = self._client.post(
            f"{self._rerank_url}/rerank",
            json={"model": self._rerank_model, "query": _sanitize(query),
                  "documents": [_sanitize(d) for d in documents]},
        )
        r.raise_for_status()
        scores = [0.0] * len(documents)
        for row in r.json().get("results", []):
            idx = row.get("index")
            if isinstance(idx, int) and 0 <= idx < len(documents):
                scores[idx] = row.get("relevance_score", 0.0)
        return scores
