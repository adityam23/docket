"""T22: filing metadata extraction, threading through ingest, persistence, and
projection into hits / citations / the Filings read model. Offline +
deterministic."""

from __future__ import annotations

from docket.ingest.chunk import chunk_text
from docket.ingest.index import Corpus
from docket.ingest.metadata import extract_filing_metadata


def test_extract_sec_filing():
    m = extract_filing_metadata("3M_2018_10K")
    assert m["form_type"] == "10-K"
    assert m["fiscal_period"] == "2018"
    assert m["company"] == "3M"


def test_extract_ticker_from_caps():
    m = extract_filing_metadata("MMM_2022_10Q")
    assert m["form_type"] == "10-Q"
    assert m["ticker"] == "MMM"
    assert m["fiscal_period"] == "2022"


def test_extract_non_filing_is_empty():
    # An arbitrary (non-SEC) document must yield no invented identity.
    m = extract_filing_metadata("ABSLI _ LEAP", "/x/ABSLI _ LEAP.pdf")
    assert m == {"form_type": "", "company": "", "ticker": "", "fiscal_period": ""}


def test_chunk_threads_metadata():
    chunks = chunk_text(
        "revenue grew", doc_id="3M_2018_10K",
        form_type="10-K", company="3M", fiscal_period="2018",
    )
    assert chunks and all(
        ch.form_type == "10-K" and ch.company == "3M" for ch in chunks
    )


def test_metadata_roundtrip_jsonl_and_duckdb(tmp_path):
    try:
        import duckdb  # noqa: F401
    except ImportError:
        backends = ["jsonl"]
    else:
        backends = ["jsonl", "duckdb"]
    c = Corpus()
    c.add(
        chunk_text("revenue grew", doc_id="3M_2018_10K",
                   form_type="10-K", company="3M", fiscal_period="2018"),
        [[1.0]],
    )
    for backend in backends:
        c.save(str(tmp_path / backend), backend=backend)
        loaded = Corpus.load(str(tmp_path / backend), backend=backend)
        ch = loaded.chunks[0]
        assert ch.form_type == "10-K" and ch.company == "3M"
        assert ch.fiscal_period == "2018"


def test_hit_and_view_carry_metadata():
    from docket.agent.trace import hit_view
    from docket.retrieval.retriever import Retriever

    c = Corpus()
    c.add(
        chunk_text("capital expenditure increased", doc_id="3M_2018_10K",
                   form_type="10-K", company="3M"),
        None,
    )
    hits = Retriever(c).retrieve("capital expenditure", k=1)
    assert hits and hits[0]["form_type"] == "10-K"
    view = hit_view(hits[0])
    assert view["form_type"] == "10-K" and view["company"] == "3M"


def test_corpus_stats_surfaces_metadata():
    from docket.config import Settings
    from docket.web.observability import corpus_stats

    c = Corpus()
    c.add(
        chunk_text("revenue grew", doc_id="3M_2018_10K",
                   form_type="10-K", company="3M"),
        None,
    )
    stats = corpus_stats(Settings(index_dir="/tmp/unused"), corpus=c)
    doc = stats["documents"][0]
    assert doc["form_type"] == "10-K" and doc["company"] == "3M"
