"""Offline tests for the ONE OpenAI-/v1 client plumbing itself (not the
higher-level rerank/retrieval seams covered in ``test_rerank.py``): path
encoding, the shared ``_send`` status handling, the bearer header, and OCR.
Uses a real ``OpenAICompatProvider`` over ``httpx.MockTransport`` so the
assertions are against the exact bytes that would hit the wire -- no
hand-rolled fake standing in for httpx itself."""

from __future__ import annotations

import httpx

from docket.providers.openai_compat import OpenAICompatProvider


def _provider_over(handler, **kw) -> OpenAICompatProvider:
    p = OpenAICompatProvider(name="mettle", base_url="http://x/v1", chat_model="m", **kw)
    # Keep the headers the real client built (bearer auth) but swap the
    # transport for the mock, so auth-header assertions stay meaningful.
    p._client = httpx.Client(transport=httpx.MockTransport(handler), headers=p._client.headers)
    return p


def test_delete_document_percent_encodes_hash_and_slash_in_the_path():
    """A chunk id like `acme#3` must not be read as a URL fragment -- httpx
    would otherwise silently drop everything from `#` onward, hitting the
    wrong path and 404ing in a way that looks like "already deleted" when it
    is actually a client bug (the engine never saw the real DELETE). A literal
    `/` inside an id must likewise stay part of ONE segment, not become an
    extra path component."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["raw_path"] = request.url.raw_path.decode()
        return httpx.Response(200)

    p = _provider_over(handler)

    assert p.delete_document("acme", "acme#3") is True
    assert captured["raw_path"] == "/v1/collections/acme/documents/acme%233"

    assert p.delete_document("acme", "sub/doc#9") is True
    assert captured["raw_path"] == "/v1/collections/acme/documents/sub%2Fdoc%239"


def test_collection_path_traversal_is_encoded_not_escaped():
    """A hostile/garbled collection name must stay ONE opaque path segment --
    it must never widen into a different /v1 or /engine/* route carrying the
    same bearer token."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["raw_path"] = request.url.raw_path.decode()
        return httpx.Response(200)

    p = _provider_over(handler)
    p.delete_document("../../engine/x", "doc1")
    assert captured["raw_path"] == "/v1/collections/..%2F..%2Fengine%2Fx/documents/doc1"


def test_delete_document_404_returns_false_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    p = _provider_over(handler)
    assert p.delete_document("acme", "ghost#1") is False


def test_ensure_collection_treats_409_as_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409)

    p = _provider_over(handler)
    p.ensure_collection("acme")  # must not raise


def test_ensure_collection_raises_on_other_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad name"})

    p = _provider_over(handler)
    try:
        p.ensure_collection("acme")
    except httpx.HTTPStatusError:
        pass
    else:
        raise AssertionError("expected a 400 to raise")


def test_chat_always_requests_logprobs():
    """No capability gate: mettle is the only backend and always supports
    per-token logprobs, so the flag is unconditional."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    p = _provider_over(handler)
    result = p.chat([{"role": "user", "content": "hi"}])
    assert captured["body"]["logprobs"] is True
    assert result.text == "hi"


def test_chat_empty_content_falls_back_to_empty_string():
    """mettle guarantees a string always, but `or ""` is defence-in-depth for
    a reasoning model hitting max_tokens before any content streamed."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
        )

    p = _provider_over(handler)
    result = p.chat([{"role": "user", "content": "hi"}])
    assert result.text == ""


def test_bearer_header_sent_on_every_request():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"text": "hello"})

    p = _provider_over(handler, api_key="s3cr3t")
    p.ocr("base64bytes==")
    assert captured["auth"] == "Bearer s3cr3t"


def test_ocr_posts_image_and_returns_text():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        captured["raw_path"] = request.url.raw_path.decode()
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"text": "Total revenue $10M", "model": "ocr-1"})

    p = _provider_over(handler)
    text = p.ocr("aGVsbG8=", instruction="read the table")
    assert captured["raw_path"] == "/v1/ocr"
    assert captured["body"] == {"image": "aGVsbG8=", "instruction": "read the table"}
    assert text == "Total revenue $10M"


def test_ocr_omits_instruction_when_not_given():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"text": ""})

    p = _provider_over(handler)
    assert p.ocr("aGVsbG8=") == ""
    assert captured["body"] == {"image": "aGVsbG8="}
