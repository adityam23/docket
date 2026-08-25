"""Structured, cited report schema (docs/decisions.md Q15). Flagship = an SEC
filing analysis with a reliability label the end user can read at a glance."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str
    page: int
    quote: str

    @classmethod
    def from_hit(cls, hit: dict, *, max_quote: int = 280) -> "Citation":
        """Build a citation from a retrieval hit — the ONE place a hit becomes a
        cited source (used by the agent orchestrator and the cite_page tool)."""
        return cls(doc_id=hit["doc_id"], page=hit["page"], quote=hit["text"][:max_quote])


class ReportSection(BaseModel):
    heading: str
    body: str
    citations: list[Citation] = Field(default_factory=list)


class Report(BaseModel):
    title: str
    reliability: str = Field(description="high | medium | low (docs/decisions Q9/Q15)")
    sections: list[ReportSection] = Field(default_factory=list)
    # TODO(phase-1/3): metrics table, risk factors, red flags for the SEC report.
