"""DuckDB persistence for the corpus — the platform-profile store.

JSONL (``Corpus.save``/``Corpus.load``) stays the lite, zero-config path; this
module is the SQL-backed swap: chunks + optional dense vectors in a single
DuckDB file, plus document lineage (counts, bytes, ingested-at) exposed for
metadata queries and the platform SQL layer. ``duckdb`` is declared under the
``ingest`` extra (``pyproject.toml``). This module is imported lazily (only when
``backend="duckdb"``), so the lite path never requires duckdb.

Retrieval is unchanged: on load the full :class:`Corpus` is reconstructed in
memory, and ``dense_search`` stays exact brute-force cosine inside docket.
DuckDB is the *store*, not the vector index — the vector index is the engine's
turbovec surface (docs/decisions.md, 2026-08-27). Vectors are persisted so a
duckdb corpus can be re-loaded for dense retrieval, but the SQL layer never
searches them.
"""

from __future__ import annotations

import os

import duckdb

from .chunk import Chunk
from .index import Corpus

DB_NAME = "corpus.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR PRIMARY KEY,
    doc_id   VARCHAR NOT NULL,
    page     INTEGER NOT NULL,
    text     VARCHAR NOT NULL,
    source   VARCHAR NOT NULL DEFAULT '',
    form_type     VARCHAR NOT NULL DEFAULT '',
    company       VARCHAR NOT NULL DEFAULT '',
    ticker        VARCHAR NOT NULL DEFAULT '',
    fiscal_period VARCHAR NOT NULL DEFAULT '',
    vec      FLOAT[]   -- NULL when sparse-only (no embedding endpoint)
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id      VARCHAR PRIMARY KEY,
    source      VARCHAR NOT NULL DEFAULT '',
    n_chunks    BIGINT NOT NULL,
    chunk_bytes BIGINT NOT NULL,
    ingested_at TIMESTAMP NOT NULL DEFAULT current_timestamp
);
"""


def _db_path(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, DB_NAME)


def _open(path: str):
    """Connect to the corpus DB, creating the schema on first use."""
    con = duckdb.connect(_db_path(path))
    con.execute(_SCHEMA)
    return con


def save(corpus: Corpus, path: str) -> None:
    """Persist the corpus (chunks + vectors + document lineage) to DuckDB.

    Full rewrite on each save — the corpus is the authoritative in-memory state
    and small at docket scale; the platform SQL layer sits on top of this file.
    """
    con = _open(path)
    try:
        con.execute("DELETE FROM chunks")
        con.execute("DELETE FROM documents")
        con.executemany(
            "INSERT INTO chunks (chunk_id, doc_id, page, text, source, "
            "form_type, company, ticker, fiscal_period, vec) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (ch.chunk_id, ch.doc_id, ch.page, ch.text, ch.source,
                 ch.form_type, ch.company, ch.ticker, ch.fiscal_period, vec)
                for ch, vec in zip(corpus.chunks, corpus.vectors)
            ],
        )
        docs: dict[str, list] = {}  # doc_id -> [n_chunks, bytes, source]
        for ch in corpus.chunks:
            row = docs.setdefault(ch.doc_id, [0, 0, ch.source])
            row[0] += 1
            row[1] += len(ch.text.encode("utf-8"))
        con.executemany(
            "INSERT INTO documents (doc_id, source, n_chunks, chunk_bytes) "
            "VALUES (?, ?, ?, ?)",
            [(d, row[2], row[0], row[1]) for d, row in docs.items()],
        )
    finally:
        con.close()


def load(path: str) -> Corpus:
    """Reconstruct the full :class:`Corpus` from the DuckDB file.

    Chunks and vectors are paired and loaded in ``chunk_id`` order, so
    ``dense_search`` sees the same (chunk, vector) pairing as the JSONL path.
    """
    db = os.path.join(path, DB_NAME)
    if not os.path.exists(db):
        return Corpus()
    con = duckdb.connect(db)
    try:
        rows = con.execute(
            "SELECT chunk_id, doc_id, page, text, source, form_type, company, "
            "ticker, fiscal_period, vec FROM chunks "
            "ORDER BY chunk_id"
        ).fetchall()
    finally:
        con.close()
    chunks: list[Chunk] = []
    vecs: list[list[float] | None] = []
    for (chunk_id, doc_id, page, text, source, form_type, company,
         ticker, fiscal_period, vec) in rows:
        chunks.append(
            Chunk(doc_id=doc_id, chunk_id=chunk_id, page=int(page),
                  text=text, source=source, form_type=form_type,
                  company=company, ticker=ticker, fiscal_period=fiscal_period)
        )
        vecs.append(list(vec) if vec is not None else None)
    c = Corpus()
    c.add(chunks, vecs)
    return c


# --- SQL/lineage query surface (the platform value over JSONL) --------------

def list_documents(path: str) -> list[dict]:
    """Document lineage rows, newest ingest first: ``doc_id``, source,
    ``n_chunks``, ``chunk_bytes``, ``ingested_at``."""
    db = os.path.join(path, DB_NAME)
    if not os.path.exists(db):
        return []
    con = duckdb.connect(db)
    try:
        rows = con.execute(
            "SELECT doc_id, source, n_chunks, chunk_bytes, ingested_at "
            "FROM documents ORDER BY ingested_at DESC, doc_id"
        ).fetchall()
    finally:
        con.close()
    return [
        {
            "doc_id": d,
            "source": s,
            "n_chunks": int(n),
            "chunk_bytes": int(b),
            "ingested_at": str(t),
        }
        for d, s, n, b, t in rows
    ]


def count_chunks(path: str) -> int:
    """Total chunk rows — the cheap cardinality probe for capacity/stats."""
    db = os.path.join(path, DB_NAME)
    if not os.path.exists(db):
        return 0
    con = duckdb.connect(db)
    try:
        return int(con.execute("SELECT count(*) FROM chunks").fetchone()[0])
    finally:
        con.close()


def chunks_for_doc(path: str, doc_id: str) -> list[dict]:
    """Chunk rows for one document — the metadata/citation query surface
    (push a doc filter down to SQL instead of scanning every row)."""
    db = os.path.join(path, DB_NAME)
    if not os.path.exists(db):
        return []
    con = duckdb.connect(db)
    try:
        rows = con.execute(
            "SELECT chunk_id, page, text, source, vec FROM chunks "
            "WHERE doc_id = ? ORDER BY chunk_id",
            [doc_id],
        ).fetchall()
    finally:
        con.close()
    return [
        {
            "chunk_id": cid,
            "page": int(pg),
            "text": t,
            "source": src,
            "vec": list(vec) if vec is not None else None,
        }
        for cid, pg, t, src, vec in rows
    ]
