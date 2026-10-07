"""Offline test of the /ask web endpoint with the service layer faked out, plus
the logprobs extraction that feeds the Tier-1 trust label."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from docket.agent.graph import Answer
from docket.ingest.index import Corpus
from docket.providers.openai_compat import _extract_logprobs
from docket.report.schema import Citation
import docket.web.app as webapp


def _client() -> TestClient:
    # A real loopback Host: the app's TrustedHostMiddleware rejects anything
    # else (incl. TestClient's default "testserver").
    return TestClient(webapp.app, base_url="http://127.0.0.1")


def test_ask_endpoint(monkeypatch):
    fake = Answer(
        question="What was revenue?",
        answer="Revenue was $10M [1].",
        reliability="high",
        surprisal=0.03,
        citations=[Citation(doc_id="acme-10k", page=1, quote="Total revenue ...")],
        hits=[{"doc_id": "acme-10k"}],
    )
    trace = {"id": "trace-x", "surprisal": 0.03, "elapsed_ms": 5, "steps": [], "chat_id": None}
    monkeypatch.setattr(
        webapp, "ask_traced_service", lambda q, on_step=None, session_id=None: (fake, trace)
    )
    client = _client()
    r = client.post("/ask", json={"question": "What was revenue?"})
    assert r.status_code == 200
    body = r.json()
    assert body["reliability"] == "high"
    assert body["citations"][0]["doc_id"] == "acme-10k"
    assert "🟢" in body["banner"]
    assert body["surprisal"] == 0.03
    assert body["trace_id"] == "trace-x"


def test_index_page_served():
    # Serves the SvelteKit SPA when built, else the "not built" fallback — either
    # way the primary surface responds 200 (docs/decisions Q12).
    client = _client()
    r = client.get("/")
    assert r.status_code == 200


def test_api_corpus_and_config(monkeypatch):
    # /api/corpus and /api/config call these bare; /api/overview calls them
    # with the settings/corpus it already loaded once (see api_overview) — the
    # fakes accept either calling convention.
    monkeypatch.setattr(
        webapp, "corpus_stats",
        lambda *a, **k: {"totals": {"documents": 1, "chunks": 3}, "retrieval": {"mode": "hybrid"}, "index_dir": ".x"},
    )
    monkeypatch.setattr(webapp, "config_view", lambda *a, **k: {"backend_url": "http://127.0.0.1:11434/v1", "chat_model": "m"})
    monkeypatch.setattr(webapp, "health_service", lambda *a, **k: {"ok": True, "provider": "mettle", "models": ["fake"]})
    monkeypatch.setattr(webapp, "capacity_stats", lambda *a, **k: {"remaining_documents_est": 1})
    client = _client()

    corpus = client.get("/api/corpus").json()
    assert corpus["totals"]["chunks"] == 3 and corpus["retrieval"]["mode"] == "hybrid"

    cfg = client.get("/api/config").json()
    assert cfg["chat_model"] == "m"

    overview = client.get("/api/overview").json()
    assert overview["health"]["ok"] is True and overview["totals"]["documents"] == 1


def test_overview_loads_corpus_exactly_once(monkeypatch):
    # api_overview used to call corpus_stats() then capacity_stats() (which
    # itself calls corpus_stats() again) with no shared state, costing two
    # Corpus.load()s per request. It now loads the corpus once and passes it
    # down to both read models.
    monkeypatch.setattr(webapp, "health_service", lambda *a, **k: {"ok": True, "provider": "mettle"})

    calls = {"n": 0}
    orig_load = Corpus.load.__func__

    def counting_load(cls, path):
        calls["n"] += 1
        return orig_load(cls, path)

    monkeypatch.setattr(Corpus, "load", classmethod(counting_load))
    client = _client()

    r = client.get("/api/overview")
    assert r.status_code == 200
    assert calls["n"] == 1


def test_ingest_status_when_idle():
    client = _client()
    body = client.get("/api/ingest/status").json()
    assert body["running"] is False and "job" in body


def test_ingest_rejects_bad_folder():
    client = _client()
    r = client.post("/api/ingest", json={"folder": "/no/such/folder/xyz"})
    assert r.status_code == 400


def test_remove_document_endpoint(monkeypatch):
    monkeypatch.setattr(webapp, "remove_document", lambda doc_id: 4)
    client = _client()

    r = client.delete("/api/corpus/d1")
    assert r.status_code == 200 and r.json()["removed_chunks"] == 4

    monkeypatch.setattr(webapp, "remove_document", lambda doc_id: 0)
    assert client.delete("/api/corpus/nope").status_code == 404  # unknown id

    # Busy: another corpus-mutating operation (an ingest job, in practice)
    # already holds the shared lock -- B4's `_require_idle` rejects instead
    # of racing Corpus.load/save against it.
    assert webapp._pipeline.CORPUS_LOCK.acquire(blocking=False)
    try:
        assert client.delete("/api/corpus/d1").status_code == 409
    finally:
        webapp._pipeline.CORPUS_LOCK.release()


def test_remove_document_endpoint_runtime_error_is_409(monkeypatch):
    # service.remove_document raises RuntimeError when the engine is out of
    # sync with our lineage -- B4 item 8: a friendly 409, not a bare 500.
    def boom(doc_id):
        raise RuntimeError(f"engine does not hold 1 of 2 chunk(s) for document {doc_id!r}")

    monkeypatch.setattr(webapp, "remove_document", boom)
    client = _client()
    r = client.delete("/api/corpus/d1")
    assert r.status_code == 409 and "does not hold" in r.json()["detail"]


def test_samples_endpoint(monkeypatch):
    monkeypatch.setattr(webapp, "load_samples", lambda: None)
    monkeypatch.setattr(webapp, "corpus_stats", lambda: {"totals": {"documents": 2, "chunks": 5}})
    client = _client()
    r = client.post("/api/samples/load")
    assert r.status_code == 200 and r.json()["totals"]["documents"] == 2


def test_samples_endpoint_busy_while_corpus_locked(monkeypatch):
    monkeypatch.setattr(webapp, "load_samples", lambda: None)
    client = _client()
    assert webapp._pipeline.CORPUS_LOCK.acquire(blocking=False)
    try:
        r = client.post("/api/samples/load")
        assert r.status_code == 409
    finally:
        webapp._pipeline.CORPUS_LOCK.release()


def test_capacity_endpoint(monkeypatch):
    monkeypatch.setattr(
        webapp, "capacity_stats",
        lambda: {"remaining_documents_est": 123, "device": {"disk_free_bytes": 1}},
    )
    client = _client()
    r = client.get("/api/capacity")
    assert r.status_code == 200 and r.json()["remaining_documents_est"] == 123


def test_upload_endpoint(monkeypatch):
    import base64

    class FakeJob:
        def as_dict(self):
            return {"id": "job-9", "status": "running", "total": 1, "completed": 0, "docs": []}

    captured = {}

    def fake_start_uploads(files, settings=None):
        captured["files"] = files
        return FakeJob()

    monkeypatch.setattr(webapp.JOBS, "start_uploads", fake_start_uploads)
    client = _client()
    payload = {"files": [{"name": "a.pdf", "content_b64": base64.b64encode(b"%PDF-1.4 hi").decode()}]}
    r = client.post("/api/ingest/upload", json=payload)
    assert r.status_code == 200 and r.json()["id"] == "job-9"
    assert captured["files"][0][0] == "a.pdf" and captured["files"][0][1] == b"%PDF-1.4 hi"


def test_upload_rejects_bad_base64(monkeypatch):
    monkeypatch.setattr(webapp.JOBS, "start_uploads", lambda files, settings=None: None)
    client = _client()
    r = client.post("/api/ingest/upload", json={"files": [{"name": "a.pdf", "content_b64": "not!!b64"}]})
    assert r.status_code == 400


def test_upload_rejects_oversize_file(monkeypatch):
    # T11: a single PDF over the per-file limit is rejected (413) naming the file,
    # before it ever reaches the ingest worker.
    import base64

    monkeypatch.setattr(webapp, "_MAX_FILE_BYTES", 10)
    monkeypatch.setattr(webapp.JOBS, "start_uploads", lambda files, settings=None: None)
    client = _client()
    payload = {"files": [{"name": "big.pdf", "content_b64": base64.b64encode(b"x" * 11).decode()}]}
    r = client.post("/api/ingest/upload", json=payload)
    assert r.status_code == 413 and "big.pdf" in r.json()["detail"]


def test_upload_rejects_oversized_content_length_header(monkeypatch):
    # B4 item 7b: a huge DECLARED Content-Length is rejected before the body
    # is read/base64-decoded at all, using a generous bound over the raw cap.
    monkeypatch.setattr(webapp.JOBS, "start_uploads", lambda files, settings=None: None)
    client = _client()
    huge = str(webapp._MAX_UPLOAD_BYTES * 3)
    r = client.post(
        "/api/ingest/upload",
        json={"files": []},
        headers={"content-length": huge},
    )
    assert r.status_code == 413


def test_upload_small_content_length_passes_the_precheck(monkeypatch):
    class FakeJob:
        def as_dict(self):
            return {"id": "job-1", "status": "running", "total": 0, "completed": 0, "docs": []}

    monkeypatch.setattr(webapp.JOBS, "start_uploads", lambda files, settings=None: FakeJob())
    client = _client()
    r = client.post("/api/ingest/upload", json={"files": []})
    assert r.status_code == 200  # declared length is small -> precheck passes


def test_config_patch_endpoint(monkeypatch):
    called = {}
    monkeypatch.setattr(webapp, "update_config", lambda updates: called.setdefault("u", updates))
    monkeypatch.setattr(webapp, "config_view", lambda: {"chat_model": "m"})
    client = _client()
    r = client.patch("/api/config", json={"updates": {"chat_model": "m"}})
    assert r.status_code == 200 and called["u"] == {"chat_model": "m"}


def test_config_patch_rejects_invalid(monkeypatch):
    def boom(updates):
        raise webapp.ConfigWriteError("bad key")

    monkeypatch.setattr(webapp, "update_config", boom)
    client = _client()
    r = client.patch("/api/config", json={"updates": {"x": 1}})
    assert r.status_code == 422 and "bad key" in r.json()["detail"]


def test_keys_endpoint_never_echoes_secret(monkeypatch):
    store = {}
    monkeypatch.setattr(webapp, "set_secret", lambda p, v: store.update({p: v}))
    monkeypatch.setattr(webapp, "clear_secret", lambda p: store.pop(p, None))
    monkeypatch.setattr(webapp, "secret_status", lambda: {"engine": "engine" in store})
    client = _client()

    r = client.post("/api/config/keys", json={"provider": "engine", "value": "sk-secret"})
    assert r.status_code == 200
    body = r.json()
    assert body == {"api_keys": {"engine": True}}
    import json as _json

    assert "sk-secret" not in _json.dumps(body)   # value is never returned

    r2 = client.post("/api/config/keys", json={"provider": "engine", "value": None})
    assert r2.json()["api_keys"]["engine"] is False  # null value clears


def test_traces_endpoints(monkeypatch):
    class FakeStore:
        def __init__(self):
            self.ids = {"t1"}

        def list(self):
            return [{"id": tid, "question": "q"} for tid in self.ids]

        def get(self, tid):
            return {"id": "t1", "steps": []} if tid in self.ids else None

        def delete(self, tid):
            if tid not in self.ids:
                return False
            self.ids.discard(tid)
            return True

    store = FakeStore()
    monkeypatch.setattr(webapp, "trace_store", lambda: store)
    client = _client()
    assert client.get("/api/traces").json()["traces"][0]["id"] == "t1"
    assert client.get("/api/traces/t1").json()["id"] == "t1"
    assert client.get("/api/traces/nope").status_code == 404
    # delete a one-shot trace, then confirm it's gone / 404 on re-delete
    assert client.delete("/api/traces/t1").json() == {"deleted": "t1"}
    assert client.delete("/api/traces/t1").status_code == 404


def test_ask_stream_emits_steps_and_done(monkeypatch):
    from docket.agent.graph import Answer

    fake = Answer(question="q", answer="A [1].", reliability="high", surprisal=0.02)
    trace = {"id": "trace-z", "surprisal": 0.02, "elapsed_ms": 3, "steps": []}

    def fake_traced(q, on_step=None, session_id=None):
        if on_step:
            on_step("plan", {"path": "qa"})
            on_step("retrieve", {"hop": 0, "query": q, "hits": [{"text": "x"}]})
            on_step("final", {"text": "A [1].", "reliability": "high"})
        return fake, trace

    monkeypatch.setattr(webapp, "ask_traced_service", fake_traced)
    client = _client()
    with client.stream("GET", "/api/ask/stream", params={"question": "q"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    assert "event: step" in body and "event: done" in body
    assert "trace-z" in body


def test_extract_logprobs():
    choice = {"logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]}}
    assert _extract_logprobs(choice) == [-0.1, -0.2]
    assert _extract_logprobs({"logprobs": None}) is None
    assert _extract_logprobs({}) is None
    assert _extract_logprobs({"logprobs": {"content": []}}) is None


# --- T21: first-class Chat/Session (conversation context + nesting) ----------


class _FakeChats:
    """Minimal ChatStore double: known sessions by id, plus turn capture."""

    def __init__(self, chats=None):
        self.chats = {c["id"]: c for c in (chats or [])}
        self.appended = []

    def get(self, chat_id):
        return self.chats.get(chat_id)

    def list(self):
        return sorted(self.chats.values(), key=lambda c: c.get("updated_at") or "", reverse=True)

    def delete(self, chat_id):
        return self.chats.pop(chat_id, None) is not None

    def append_turn(self, chat_id, turn, title=None, now=None):
        self.appended.append((chat_id, turn))
        return self.chats.get(chat_id)


def test_ask_with_session_threads_and_returns_chat_id(monkeypatch):
    captured = {}

    fake = Answer(question="And vs last year?", answer="It grew [1].", reliability="high", surprisal=0.02)
    trace = {"id": "trace-t21", "surprisal": 0.02, "elapsed_ms": 4, "chat_id": "chat-1", "turn_index": 1}

    def fake_traced(q, on_step=None, session_id=None):
        captured["q"] = q
        captured["session_id"] = session_id
        return fake, trace

    monkeypatch.setattr(webapp, "ask_traced_service", fake_traced)
    monkeypatch.setattr(webapp, "chat_store", lambda: _FakeChats([{"id": "chat-1", "turns": []}]))
    client = _client()

    r = client.post("/api/ask", json={"question": "And vs last year?", "session_id": "chat-1"})
    assert r.status_code == 200
    body = r.json()
    assert captured["session_id"] == "chat-1"
    assert body["chat_id"] == "chat-1" and body["trace_id"] == "trace-t21"


def test_ask_unknown_session_is_404(monkeypatch):
    monkeypatch.setattr(webapp, "ask_traced_service", lambda q, on_step=None, session_id=None: None)
    monkeypatch.setattr(webapp, "chat_store", lambda: _FakeChats([]))
    client = _client()

    r = client.post("/api/ask", json={"question": "q", "session_id": "nope"})
    assert r.status_code == 404 and "nope" in r.json()["detail"]

    # The SSE stream rejects before streaming starts.
    r2 = client.get("/api/ask/stream", params={"question": "q", "session_id": "nope"})
    assert r2.status_code == 404


def test_stream_passes_session_to_service(monkeypatch):
    captured = {}
    fake = Answer(question="q", answer="A [1].", reliability="high", surprisal=0.02)
    trace = {"id": "trace-s9", "surprisal": 0.02, "elapsed_ms": 3, "steps": [], "chat_id": "c9"}

    def fake_traced(q, on_step=None, session_id=None):
        captured["session_id"] = session_id
        if on_step:
            on_step("plan", {"path": "qa"})
        return fake, trace

    monkeypatch.setattr(webapp, "ask_traced_service", fake_traced)
    monkeypatch.setattr(webapp, "chat_store", lambda: _FakeChats([{"id": "c9", "turns": []}]))
    client = _client()

    with client.stream("GET", "/api/ask/stream", params={"question": "q", "session_id": "c9"}) as r:
        body = "".join(r.iter_text())
    assert captured["session_id"] == "c9"
    assert "event: step" in body and "event: done" in body
    assert "trace-s9" in body and "c9" in body  # trace id + nested chat id


def test_chats_endpoints(monkeypatch):
    store = _FakeChats(
        [{"id": "chat-a", "title": "first question", "created_at": "x", "updated_at": "y", "turns": []}]
    )
    monkeypatch.setattr(webapp, "chat_store", lambda: store)

    # Deleting a chat cascades to its turns' traces — stub that side.
    class _FakeTraces:
        def __init__(self):
            self.removed_for = []

        def delete_by_chat(self, chat_id):
            self.removed_for.append(chat_id)
            return 2

    traces = _FakeTraces()
    monkeypatch.setattr(webapp, "trace_store", lambda: traces)
    client = _client()

    listing = client.get("/api/chats").json()
    assert listing["chats"][0]["id"] == "chat-a" and listing["chats"][0]["title"] == "first question"

    one = client.get("/api/chats/chat-a").json()
    assert one["id"] == "chat-a"

    assert client.get("/api/chats/zz").status_code == 404

    d = client.delete("/api/chats/chat-a")
    assert d.status_code == 200 and d.json() == {"deleted": "chat-a", "traces_removed": 2}
    assert traces.removed_for == ["chat-a"]  # cascade fired
    assert client.delete("/api/chats/chat-a").status_code == 404


# --- B5: Host/Origin protection (DNS rebinding + CSRF) -----------------------


def test_foreign_host_header_is_rejected():
    # A DNS-rebound hostname pointed at 127.0.0.1 must not be served.
    client = _client()
    r = client.get("/api/health", headers={"host": "evil.example.com"})
    assert r.status_code == 400


def test_loopback_host_headers_pass(monkeypatch):
    monkeypatch.setattr(webapp, "health_service", lambda: {"ok": True})
    client = _client()
    for host in ("127.0.0.1", "127.0.0.1:8760", "localhost", "localhost:8760"):
        r = client.get("/api/health", headers={"host": host})
        assert r.status_code == 200, host


def test_cross_site_origin_on_patch_config_is_rejected():
    # CSRF: the request targets this server correctly (Host is fine) but was
    # declared by a different site -- must be blocked before it ever reaches
    # update_config.
    client = _client()
    r = client.patch(
        "/api/config",
        json={"updates": {}},
        headers={"origin": "http://evil.example.com"},
    )
    assert r.status_code == 403


def test_cross_site_sec_fetch_site_without_origin_is_rejected():
    # Falls back to Sec-Fetch-Site when Origin is absent.
    client = _client()
    r = client.post("/api/engine/restart", headers={"sec-fetch-site": "cross-site"})
    assert r.status_code == 403


def test_ask_stream_cross_site_get_is_rejected():
    # GET but side-effecting (inference + a persisted trace) -> guarded like a
    # state-changing request.
    client = _client()
    r = client.get(
        "/api/ask/stream",
        params={"question": "q"},
        headers={"origin": "http://evil.example.com"},
    )
    assert r.status_code == 403


def test_same_origin_patch_config_passes(monkeypatch):
    monkeypatch.setattr(webapp, "update_config", lambda updates: None)
    monkeypatch.setattr(webapp, "config_view", lambda: {"chat_model": "m"})
    client = _client()
    r = client.patch(
        "/api/config",
        json={"updates": {"chat_model": "m"}},
        headers={"origin": "http://127.0.0.1"},
    )
    assert r.status_code == 200


def test_absent_origin_and_sec_fetch_site_passes(monkeypatch):
    # curl / the CLI / TestClient's default: no browser is enforcing
    # same-origin for these callers, so there is no signal to check.
    monkeypatch.setattr(webapp, "update_config", lambda updates: None)
    monkeypatch.setattr(webapp, "config_view", lambda: {"chat_model": "m"})
    client = _client()
    r = client.patch("/api/config", json={"updates": {"chat_model": "m"}})
    assert r.status_code == 200


def test_chat_create_endpoint(monkeypatch):
    class _Store:
        def next_id(self, now):
            return "chat-new-1"

        def create(self, chat_id, title, now=None):
            return {"id": chat_id, "title": title, "created_at": now, "updated_at": now, "turns": []}

    monkeypatch.setattr(webapp, "chat_store", lambda: _Store())
    client = _client()
    r = client.post("/api/chats", json={"title": None})
    assert r.status_code == 200 and r.json()["id"] == "chat-new-1" and r.json()["turns"] == []
    assert r.json()["title"] == ""  # untitled until the first turn names it


# --- B4: ingest job registry atomicity ---------------------------------------


def test_next_seq_is_unique_under_concurrency():
    # The tentative-seq bug: two concurrent allocations must never return the
    # same number (which used to let two uploads share -- and rmtree-race --
    # one directory).
    import threading

    from docket.web.jobs import IngestJobs

    jobs = IngestJobs()
    results: list[int] = []
    out_lock = threading.Lock()

    def worker():
        n = jobs._next_seq()
        with out_lock:
            results.append(n)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == list(range(1, 21))  # every number unique, none skipped/duplicated


def test_upload_dedupes_same_basename(tmp_path):
    # B4 item 7a: two picked files sharing a basename must not overwrite each
    # other in the shared upload dir.
    import shutil

    from docket.web.jobs import IngestJobs

    jobs = IngestJobs()
    captured = {}

    def fake_launch(folder, pdfs, *, settings=None, source_of=None, cleanup_dir=None, job_id=None):
        captured["pdfs"] = pdfs
        captured["cleanup_dir"] = cleanup_dir

        class _Job:
            def as_dict(self):
                return {"id": job_id, "status": "running", "total": 0, "completed": 0, "docs": []}

        return _Job()

    jobs._launch = fake_launch
    jobs.start_uploads([("report.pdf", b"AAAA"), ("report.pdf", b"BBBB")])

    paths = captured["pdfs"]
    names = [os.path.basename(p) for p in paths]
    assert len(names) == 2 and names[0] != names[1]
    contents = {open(p, "rb").read() for p in paths}
    assert contents == {b"AAAA", b"BBBB"}  # neither file clobbered the other
    shutil.rmtree(captured["cleanup_dir"], ignore_errors=True)


def test_sweep_stale_uploads_removes_leftover_dirs(tmp_path):
    from docket.config import Settings
    from docket.web.jobs import sweep_stale_uploads

    s = Settings(index_dir=str(tmp_path))
    stale = os.path.join(s.index_dir, ".uploads", "job-3")
    os.makedirs(stale, exist_ok=True)
    with open(os.path.join(stale, "leftover.pdf"), "wb") as f:
        f.write(b"x")

    sweep_stale_uploads(settings=s)
    assert not os.path.exists(os.path.join(s.index_dir, ".uploads"))


def test_launch_rejects_while_corpus_lock_held():
    # The _launch-time pre-check used to run BEFORE claiming anything, so two
    # concurrent launches could both see "idle". Now it's one atomic
    # acquire-or-reject on the same lock every corpus mutation shares.
    import docket.ingest.pipeline as pipeline
    from docket.web.jobs import IngestJobs

    jobs = IngestJobs()
    assert pipeline.CORPUS_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            jobs._launch("some folder", ["/x/a.pdf"])
    finally:
        pipeline.CORPUS_LOCK.release()
    assert not jobs.is_running()  # nothing was actually started


def test_launch_releases_corpus_lock_after_job_completes(monkeypatch, tmp_path):
    # The lock is acquired in _launch and must be released by _run once the
    # worker thread finishes, whether it succeeds or raises.
    import docket.ingest.pipeline as pipeline
    from docket.config import Settings
    from docket.web.jobs import IngestJobs

    from .fakes import FakeEngine

    jobs = IngestJobs()
    monkeypatch.setattr(pipeline, "get_provider", lambda s: FakeEngine())
    monkeypatch.setattr(pipeline, "pdf_to_pages", lambda path: [{"page": 1, "text": "hello world"}])

    job = jobs._launch("f", ["/x/doc.pdf"], settings=Settings(index_dir=str(tmp_path)))
    jobs._worker.join(timeout=5)
    assert not jobs._worker.is_alive()

    # Released by _run's finally -> re-acquirable. Guard the release so a
    # failed assertion here can never leave this process-wide lock stuck held
    # for every other test in the session.
    reacquired = pipeline.CORPUS_LOCK.acquire(blocking=False)
    try:
        assert reacquired
    finally:
        if reacquired:
            pipeline.CORPUS_LOCK.release()
    assert jobs.latest()["id"] == job.id
