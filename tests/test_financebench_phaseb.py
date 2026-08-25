"""Offline tests for the Phase-B decoupling helpers in ``run_full.py``.

Phase B splits full-RAG into (1) a one-time retrieval dump and (2) per-generator
generation over those cached contexts, so the generator, embedder and reranker
never need to be resident together on a 6 GB card. These tests cover the pure
plumbing of that split — no network, no backend, no model.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_RUN_FULL = Path(__file__).resolve().parents[1] / "benchmarks" / "financebench" / "run_full.py"
_spec = importlib.util.spec_from_file_location("fb_run_full", _RUN_FULL)
run_full = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_full)


def test_done_ids_reads_and_tolerates_torn_lines(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"id": "a", "hits": []}) + "\n"
        + json.dumps({"id": "b", "hits": []}) + "\n"
        + '{"id": "c", "hits": [  \n'  # torn/partial final line
    )
    assert run_full._done_ids(str(p)) == {"a", "b"}
    assert run_full._done_ids(str(tmp_path / "missing.jsonl")) == set()


def test_cached_retrieve_replays_hits_by_id(tmp_path):
    p = tmp_path / "ctx.jsonl"
    hits = [{"doc_id": "D", "page": 1, "text": "revenue was $10m"}]
    p.write_text(json.dumps({"id": "q1", "hits": hits}) + "\n")
    retrieve = run_full._cached_retrieve(str(p))
    assert retrieve({"id": "q1"}) == hits
    assert retrieve({"id": "unknown"}) == []  # honest empty, never fabricated


def test_dump_contexts_resumes_without_re_retrieving(tmp_path):
    p = tmp_path / "dump.jsonl"
    p.write_text(json.dumps({"id": "q1", "doc_id": "D", "n_hits": 0, "hits": []}) + "\n")
    items = [{"id": "q1", "doc_id": "D"}, {"id": "q2", "doc_id": "D"}]
    called = []

    def retrieve(item):
        called.append(item["id"])
        return [{"doc_id": "D", "page": 2, "text": "x"}]

    run_full._dump_contexts(items, retrieve, str(p), base_url="http://unused")
    # q1 already dumped → only q2 is retrieved on resume
    assert called == ["q2"]
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert {r["id"] for r in rows} == {"q1", "q2"}


def test_resilient_provider_delegates_and_retries():
    class _Flaky:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def chat(self, *a, **kw):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("backend went away")
            return "ok"

    inner = _Flaky()
    wrapped = run_full._ResilientProvider(inner, base_url="http://127.0.0.1:11434/v1")
    assert wrapped.name == "flaky"  # transparent delegation of non-chat attrs

    # _resilient (shared with the sweep) waits on the backend + backs off between
    # tries; patch both so the retry doesn't block on a real socket or sleep.
    import sys
    import time as _time

    sys.path.insert(0, str(_RUN_FULL.parent))
    import sweep_recall

    sweep_recall._wait_backend = lambda *a, **k: True
    orig_sleep = _time.sleep
    _time.sleep = lambda *a, **k: None
    try:
        # first call raises → _resilient waits + retries → second call succeeds
        assert wrapped.chat([{"role": "user", "content": "hi"}]) == "ok"
    finally:
        _time.sleep = orig_sleep
    assert inner.calls == 2
