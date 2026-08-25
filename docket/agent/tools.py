"""Deterministic, typed agent tools (NO code-gen / NO sandbox in v1 —
docs/decisions.md Q16). These are the toolset that gets a small model on par
with larger ones (Q8). Each is a plain function so it is trivially unit-testable
and equally callable from the CLI, the web app, or a LangGraph node.

    retrieve            hybrid + rerank + iterative retrieval
    compare_across_docs diff a metric/claim across filings
    extract_table       pull a structured table from a doc  (LLM-backed; roadmap)
    compute_metric      deterministic financial metric from extracted fields
    aggregate           deterministic sum/mean/min/max/count over cited values
    cite_page           resolve an answer span to an exact source citation
    generate_report     assemble the structured, cited report (docs/decisions Q15)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..report.schema import Citation, Report, ReportSection
from ..retrieval.retriever import Retriever

if TYPE_CHECKING:  # only for the ``generate_report`` annotation — avoids a
    from .graph import Answer  # graph<->tools import cycle at runtime.

TOOL_NAMES = (
    "retrieve",
    "compare_across_docs",
    "extract_table",
    "compute_metric",
    "aggregate",
    "cite_page",
    "generate_report",
)


def retrieve(retriever: Retriever, query: str, *, k: int = 10) -> list[dict]:
    """Hybrid + rerank + (single-query) retrieval over the corpus."""
    return retriever.retrieve(query, k=k)


def cite_page(hit: dict, *, max_quote: int = 280) -> Citation:
    """Resolve a retrieved hit to an exact, quotable source citation."""
    return Citation.from_hit(hit, max_quote=max_quote)


def compare_across_docs(retriever: Retriever, query: str, *, k: int = 5) -> dict[str, list[dict]]:
    """Retrieve for ``query`` and group hits by document, so a claim/metric can be
    diffed across filings. Deterministic: it only groups retrieval output."""
    grouped: dict[str, list[dict]] = {}
    for hit in retriever.retrieve(query, k=k * 4):
        grouped.setdefault(hit["doc_id"], [])
        if len(grouped[hit["doc_id"]]) < k:
            grouped[hit["doc_id"]].append(hit)
    return grouped


def compute_metric(name: str, **fields: float) -> float:
    """Deterministic financial metrics from already-extracted numeric fields.

    Keeps arithmetic OUT of the LLM (Q16): the model extracts fields, this tool
    computes. Supported: growth, margin, ratio, delta, yoy_change.
    """
    n = name.lower()
    if n == "growth" or n == "yoy_change":  # (curr - prev) / prev
        prev = fields["previous"]
        if prev == 0:
            raise ZeroDivisionError("previous value is zero")
        return (fields["current"] - prev) / prev
    if n == "margin":  # numerator / revenue
        return fields["numerator"] / fields["denominator"]
    if n == "ratio":
        return fields["numerator"] / fields["denominator"]
    if n == "delta":
        return fields["current"] - fields["previous"]
    raise ValueError(f"unknown metric: {name}")


AGG_OPS = ("sum", "mean", "min", "max", "count")

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
# Suffix multipliers (money shorthand). Longer keys can't be mis-shadowed here
# because every candidate resolves to the same multiplier (m|mm|million → 1e6).
_MULT = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mm": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
}


def parse_number(raw: object) -> tuple[float, str]:
    """Parse a human-written quantity into ``(value, unit)`` deterministically.

    Handles ``$1,200`` · ``1.5M`` · ``10%`` · ``3.2 billion``. Unit is ``USD`` /
    ``%`` / ``""`` (dimensionless) so :func:`aggregate` can refuse to add dollars
    to percentages. Arithmetic stays out of the LLM (Q16): the model extracts the
    raw string, this turns it into a number.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw), ""
    s = str(raw).strip()
    m = _NUM.search(s)
    if not m:
        raise ValueError(f"no number in {raw!r}")
    val = float(m.group(0).replace(",", ""))
    low = s.lower()
    tail = low[m.end():].lstrip(" ")
    for suf, mult in _MULT.items():
        if tail.startswith(suf):
            val *= mult
            break
    if "%" in s:
        unit = "%"
    elif "$" in s or "usd" in low or "dollar" in low:
        unit = "USD"
    else:
        unit = ""
    return val, unit


def aggregate(op: str, items: list[dict]) -> dict:
    """Deterministic ``sum``/``mean``/``min``/``max``/``count`` over a list of
    ``{"value", "unit"?, ...}`` items, echoing each item's provenance back.

    This is the numeric core behind cross-document questions ("total per month
    across a bunch of docs"): the LLM extracts each cited value, this tool does
    the math. It is unit-aware — mixing ``USD`` and ``%`` raises rather than
    silently producing a nonsense total — and passes through opaque keys
    (``citation``, ``label``) so callers keep a per-line-item audit trail.
    """
    o = op.lower()
    if o not in AGG_OPS:
        raise ValueError(f"unknown op: {op}")

    norm: list[dict] = []
    units: set[str] = set()
    for it in items:
        v, u = parse_number(it["value"])
        u = it.get("unit", u) or u
        if u:
            units.add(u)
        norm.append({**it, "value": v, "unit": u})

    if o == "count":
        result: float = len(norm)
        unit = ""
    else:
        if not norm:
            raise ValueError("no values to aggregate")
        if len(units) > 1:
            raise ValueError(f"mixed units: {sorted(units)}")
        unit = next(iter(units)) if units else ""
        values = [n["value"] for n in norm]
        result = {
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }[o]
    return {"op": o, "result": result, "unit": unit, "count": len(norm), "items": norm}


def extract_table(retriever: Retriever, query: str, *, k: int = 5) -> list[dict]:
    """Pull a structured table from a doc. Needs LLM-backed cell extraction over
    the retrieved region — deferred (roadmap). Returns the candidate regions so a
    caller can wire extraction; raising would break the tool registry contract."""
    return retrieve(retriever, query, k=k)


def generate_report(result: Answer, *, title: str | None = None) -> Report:
    """Assemble the structured, cited report from an orchestrator Answer (Q15)."""
    return Report(
        title=title or result.question,
        reliability=result.reliability,
        sections=[
            ReportSection(
                heading="Answer",
                body=result.answer,
                citations=result.citations,
            )
        ],
    )


def build_tools(retriever: Retriever) -> dict:
    """Registry of bound tools for a LangGraph node or manual dispatch."""
    return {
        "retrieve": lambda query, k=10: retrieve(retriever, query, k=k),
        "compare_across_docs": lambda query, k=5: compare_across_docs(retriever, query, k=k),
        "extract_table": lambda query, k=5: extract_table(retriever, query, k=k),
        "compute_metric": compute_metric,
        "aggregate": aggregate,
        "cite_page": cite_page,
        "generate_report": generate_report,
    }
