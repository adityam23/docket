"""Structured, cited report schema. Flagship = an SEC
filing analysis with a reliability label the end user can read at a glance."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..ingest.chunk import FILING_FIELDS


class Citation(BaseModel):
    doc_id: str
    page: int
    quote: str
    # Filing identity (T22/B3), optional and never invented: populated from the
    # hit's lineage join when known, None when the chunk's filing metadata
    # wasn't extracted. Lets the SPA label a citation ("10-K, 3M, FY2018")
    # instead of the bare doc_id.
    form_type: str | None = None
    company: str | None = None
    ticker: str | None = None
    fiscal_period: str | None = None

    @classmethod
    def from_hit(cls, hit: dict, *, max_quote: int = 280) -> "Citation":
        """Build a citation from a retrieval hit — the ONE place a hit becomes a
        cited source (used by the agent orchestrator and the cite_page tool)."""
        return cls(
            doc_id=hit["doc_id"],
            page=hit["page"],
            quote=hit["text"][:max_quote],
            **{f: (hit.get(f) or None) for f in FILING_FIELDS},
        )


class ReportSection(BaseModel):
    heading: str
    body: str
    citations: list[Citation] = Field(default_factory=list)


class Report(BaseModel):
    title: str
    reliability: str = Field(description="high | medium | low")
    sections: list[ReportSection] = Field(default_factory=list)
    # TODO(phase-1/3): metrics table, risk factors, red flags for the SEC report.
