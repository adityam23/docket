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


def test_form_type_not_matched_from_full_path_substring():
    # The directory "lars" contains "ars" -- matching the FULL path would read
    # this as form type "ARS". Matching is basename-only, so it must not.
    m = extract_filing_metadata("report", "/home/lars/filings/report.pdf")
    assert m["form_type"] == ""


def test_form_type_alias_requires_word_boundary():
    # "ars" must not match inside an ordinary word in the basename either.
    m = extract_filing_metadata("quarterly-notes", "cars-financial-summary.pdf")
    assert m["form_type"] == ""


def test_year_not_matched_inside_a_longer_digit_run():
    # "0000320193" (a CIK/accession-like digit run) contains the substring
    # "2019" but is not a standalone year -- it must not be extracted as one.
    m = extract_filing_metadata("0000320193-annual")
    assert m["fiscal_period"] == ""


def test_year_still_matched_when_standalone():
    m = extract_filing_metadata("3M_2018_10K")
    assert m["fiscal_period"] == "2018"


def test_chunk_threads_metadata():
    chunks = chunk_text(
        "revenue grew", doc_id="3M_2018_10K",
        form_type="10-K", company="3M", fiscal_period="2018",
    )
    assert chunks and all(
        ch.form_type == "10-K" and ch.company == "3M" for ch in chunks
    )


def test_metadata_roundtrip(tmp_path):
    c = Corpus()
    c.add(chunk_text("revenue grew", doc_id="3M_2018_10K",
                     form_type="10-K", company="3M", fiscal_period="2018"))
    c.save(str(tmp_path))
    loaded = Corpus.load(str(tmp_path))
    ch = loaded.chunks[0]
    assert ch.form_type == "10-K" and ch.company == "3M"
    assert ch.fiscal_period == "2018"


def test_hit_and_view_carry_metadata(monkeypatch):
    from docket.agent.trace import hit_view

    from .fakes import retriever_over

    chunks = chunk_text("revenue grew", doc_id="3M_2018_10K",
                        form_type="10-K", company="3M")

    hits = retriever_over(chunks, rank="all").retrieve("revenue", k=3)
    assert hits and hits[0]["form_type"] == "10-K"
    view = hit_view(hits[0])
    assert view["form_type"] == "10-K" and view["company"] == "3M"


def test_citation_from_hit_carries_filing_identity():
    from docket.report.schema import Citation

    from .fakes import retriever_over

    chunks = chunk_text("revenue grew", doc_id="3M_2018_10K",
                        form_type="10-K", company="3M", ticker="MMM", fiscal_period="2018")
    hit = retriever_over(chunks, rank="all").retrieve("revenue", k=1)[0]
    c = Citation.from_hit(hit)
    assert c.form_type == "10-K" and c.company == "3M"
    assert c.ticker == "MMM" and c.fiscal_period == "2018"


def test_citation_from_hit_leaves_unknown_identity_as_none():
    from docket.report.schema import Citation

    from .fakes import retriever_over

    chunks = chunk_text("plain text, not a filing", doc_id="misc")
    hit = retriever_over(chunks, rank="all").retrieve("plain", k=1)[0]
    c = Citation.from_hit(hit)
    assert c.form_type is None and c.company is None
    assert c.ticker is None and c.fiscal_period is None


def test_corpus_stats_surfaces_metadata(tmp_path):
    from docket.config import Settings
    from docket.web.observability import corpus_stats

    c = Corpus()
    c.add(chunk_text("revenue grew", doc_id="3M_2018_10K",
                     form_type="10-K", company="3M"))
    c.save(str(tmp_path))
    stats = corpus_stats(Settings(index_dir=str(tmp_path)))
    doc = stats["documents"][0]
    assert doc["form_type"] == "10-K" and doc["company"] == "3M"
