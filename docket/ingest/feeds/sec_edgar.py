"""SEC EDGAR live feed watcher — the DE-first ingestion spine (docs/decisions.md
Q10). Polls the daily filing index and yields new-document events (10-K/8-K) to
be OCR'd -> chunked -> embedded -> indexed. In `platform` profile these events
flow through Kafka/Redpanda; in `lite` they run inline. TODO(phase-1/2)."""

from __future__ import annotations

from collections.abc import Iterator

EDGAR_DAILY_INDEX = "https://www.sec.gov/Archives/edgar/daily-index"


def watch_new_filings(forms: tuple[str, ...] = ("10-K", "8-K")) -> Iterator[dict]:
    """Yield {cik, form, url, filed_at} for new filings. TODO(phase-1)."""
    raise NotImplementedError("EDGAR watcher not wired yet — see docs/roadmap.md")
