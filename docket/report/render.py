"""Render a Report to markdown for the end user: a reliability banner they can
read at a glance (🟢/🟡/🔴, docs/decisions Q15) followed by cited sections."""

from __future__ import annotations

from .schema import Report

_BANNER = {
    "high": "🟢 High reliability — well grounded in the sources.",
    "medium": "🟡 Medium reliability — check the citations.",
    "low": "🔴 Low reliability — the model may be guessing; verify before trusting.",
    "unknown": "⚪ Reliability unknown — backend did not expose token probabilities.",
}


def reliability_banner(reliability: str) -> str:
    return _BANNER.get(reliability, _BANNER["unknown"])


def to_markdown(report: Report) -> str:
    lines = [f"# {report.title}", "", f"> {reliability_banner(report.reliability)}", ""]
    for sec in report.sections:
        lines.append(f"## {sec.heading}")
        lines.append("")
        lines.append(sec.body)
        if sec.citations:
            lines.append("")
            lines.append("**Sources**")
            for i, c in enumerate(sec.citations, start=1):
                quote = c.quote.strip().replace("\n", " ")
                if len(quote) > 200:
                    quote = quote[:197] + "…"
                lines.append(f"{i}. `{c.doc_id}` p.{c.page} — “{quote}”")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
