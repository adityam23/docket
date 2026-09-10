"""Offline tests for the DuckDB platform corpus store: round-trip + SQL query
surface. Skips cleanly when the ``ingest`` extra (duckdb) is not installed, so
the base suite stays green either way."""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")

from docket.ingest.chunk import Chunk
from docket.ingest.index import Corpus


def _corpus() -> Corpus:
    c = Corpus()
    chunks = [
        Chunk(doc_id="a", chunk_id="a#0", page=1, text="revenue grew", source="a.pdf"),
        Chunk(doc_id="a", chunk_id="a#1", page=2, text="expenses rose", source="a.pdf"),
        Chunk(doc_id="b", chunk_id="b#0", page=1, text="cash flow", source="b.pdf"),
    ]
    vecs = [[1.0, 0.0], [0.0, 1.0], None]  # one sparse-only chunk (no vector)
    c.add(chunks, vecs)
    return c


def test_duckdb_roundtrip_preserves_chunks_and_vectors(tmp_path):
    c = _corpus()
    c.save(str(tmp_path), backend="duckdb")
    loaded = Corpus.load(str(tmp_path), backend="duckdb")
    assert len(loaded) == 3
    assert set(loaded.doc_ids()) == {"a", "b"}
    by_id = {ch.chunk_id: i for i, ch in enumerate(loaded.chunks)}
    assert loaded.vectors[by_id["a#0"]] == pytest.approx([1.0, 0.0])
    assert loaded.vectors[by_id["b#0"]] is None  # sparse slot survives
    # dense search still resolves after reload
    assert loaded.dense_search([1.0, 0.0], k=1)[0][0] == "a#0"


def test_duckdb_query_surface(tmp_path):
    from docket.ingest import duckdb_store as db

    _corpus().save(str(tmp_path), backend="duckdb")
    docs = db.list_documents(str(tmp_path))
    assert {d["doc_id"] for d in docs} == {"a", "b"}
    a = next(d for d in docs if d["doc_id"] == "a")
    assert a["n_chunks"] == 2 and a["chunk_bytes"] > 0
    assert db.count_chunks(str(tmp_path)) == 3
    rows = db.chunks_for_doc(str(tmp_path), "a")
    assert [r["chunk_id"] for r in rows] == ["a#0", "a#1"]


def test_jsonl_default_unchanged(tmp_path):
    c = _corpus()
    c.save(str(tmp_path))  # default backend stays jsonl (lite, zero-config)
    assert (tmp_path / "corpus.jsonl").exists()
    assert len(Corpus.load(str(tmp_path))) == 3
