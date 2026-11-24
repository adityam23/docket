"""Offline tests for the ONE OpenAI-/v1 client plumbing itself (not the
higher-level rerank/retrieval seams covered in ``test_rerank.py``): path
encoding, the shared ``_send`` status handling, the bearer header, and OCR.
Uses a real ``OpenAICompatProvider`` over ``httpx.MockTransport`` so the
assertions are against the exact bytes that would hit the wire -- no
hand-rolled fake standing in for httpx itself."""

from __future__ import annotations

import json

import httpx
import pytest

from docket.providers import openai_compat as oc
from docket.providers.base import EngineDocument
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


def _batch_engine(fail_on_batch: int | None = None, status: int = 503):
    """A mock ``documents:batch`` endpoint recording each request; the batch
    numbered ``fail_on_batch`` (1-based) answers ``status`` and writes nothing."""
    log: dict = {"batches": [], "deleted": [], "raw_paths": []}

    def handler(request: httpx.Request) -> httpx.Response:
        log["raw_paths"].append(request.url.raw_path.decode())
        if request.method == "DELETE":
            log["deleted"].append(request.url.raw_path.decode().rsplit("/", 1)[-1])
            return httpx.Response(200)
        docs = json.loads(request.content)["documents"]
        log["batches"].append([d["id"] for d in docs])
        if len(log["batches"]) == fail_on_batch:
            return httpx.Response(status, json={"error": {"code": "conflict"}})
        return httpx.Response(201, json={"results": [
            {"document_id": d["id"], "chunks": 1, "chunk_ids": [f"{d['id']}/c0"]} for d in docs
        ]})

    return handler, log


def _docs(n: int, text: str = "x") -> list[EngineDocument]:
    return [EngineDocument(f"f#{i}", text, {"page": "1"}) for i in range(n)]


def test_post_documents_sends_one_batch_to_the_literal_batch_route():
    handler, log = _batch_engine()
    p = _provider_over(handler)
    assert p.post_documents("acme", _docs(3)) == ["f#0/c0", "f#1/c0", "f#2/c0"]
    assert log["batches"] == [["f#0", "f#1", "f#2"]]
    assert log["raw_paths"] == ["/v1/collections/acme/documents:batch"]


def test_post_documents_splits_over_the_count_cap(monkeypatch):
    monkeypatch.setattr(oc, "_BATCH_MAX_DOCS", 2)
    handler, log = _batch_engine()
    p = _provider_over(handler)
    p.post_documents("acme", _docs(5))
    assert [len(b) for b in log["batches"]] == [2, 2, 1]


def test_wire_batches_splits_over_the_byte_cap(monkeypatch):
    monkeypatch.setattr(oc, "_BATCH_MAX_BYTES", 300 + oc._BATCH_ENVELOPE_BYTES)
    docs = [{"id": f"d{i}", "content": "y" * 100, "metadata": {}} for i in range(5)]
    batches = oc._wire_batches(docs)
    assert [len(b) for b in batches] == [2, 2, 1]
    assert [d for b in batches for d in b] == docs  # order preserved


def test_wire_batches_rejects_a_single_document_over_the_byte_cap(monkeypatch):
    monkeypatch.setattr(oc, "_BATCH_MAX_BYTES", 100 + oc._BATCH_ENVELOPE_BYTES)
    with pytest.raises(ValueError, match="over the batch cap"):
        oc._wire_batches([{"id": "big", "content": "z" * 200, "metadata": {}}])


def test_post_documents_rolls_back_committed_batches_when_a_later_one_fails(monkeypatch):
    monkeypatch.setattr(oc, "_BATCH_MAX_DOCS", 2)
    handler, log = _batch_engine(fail_on_batch=2, status=409)
    p = _provider_over(handler)
    with pytest.raises(httpx.HTTPStatusError):
        p.post_documents("acme", _docs(5))
    # Batch 1 committed and is rolled back; batch 2 wrote nothing (and on a 409
    # its ids may belong to an existing document) so it is left alone; batch 3
    # was never sent.
    assert log["deleted"] == ["f%230", "f%231"]
    assert len(log["batches"]) == 2


def test_post_documents_failure_on_first_batch_deletes_nothing():
    handler, log = _batch_engine(fail_on_batch=1)
    p = _provider_over(handler)
    with pytest.raises(httpx.HTTPStatusError):
        p.post_documents("acme", _docs(3))
    assert log["deleted"] == []


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


def test_batch_post_uses_the_batch_timeout_and_other_calls_the_shared_one():
    # A filing's documents:batch embeds every chunk before answering, so it
    # gets its own (longer) ceiling; every other call keeps the shared one.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.method] = request.extensions["timeout"]["read"]
        if request.method == "POST":
            return httpx.Response(201, json={"results": []})
        return httpx.Response(200)

    p = _provider_over(handler, timeout_s=120.0, batch_timeout_s=900.0)
    p._client.timeout = httpx.Timeout(120.0)  # the real client's shared timeout
    p.post_documents("acme", _docs(1))
    p.delete_document("acme", "f#0")
    assert seen == {"POST": 900.0, "DELETE": 120.0}


def test_list_documents_gets_the_collection_listing():
    rows = [{"id": "acme#0", "chunk_count": 1, "metadata": {"page": "1"}}]
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["call"] = (request.method, request.url.raw_path.decode())
        return httpx.Response(200, json=rows)

    p = _provider_over(handler)
    assert p.list_documents("acme") == rows
    assert captured["call"] == ("GET", "/v1/collections/acme/documents")
