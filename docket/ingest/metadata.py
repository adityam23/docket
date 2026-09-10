"""Best-effort SEC filing metadata extraction (T22).

Parses form type / company / ticker / fiscal period from a document's id or
source filename. Heuristic and deliberately conservative: anything that does
not look like an SEC filing yields empty strings, and callers fall back to the
bare ``doc_id`` (never invent identity).
"""

from __future__ import annotations

import re

# Canonical form -> substrings (lowercased) that identify it. Order matters:
# more specific forms first so "10-K" isn't swallowed by a generic match.
_FORM_ALIASES = [
    ("DEF 14A", ["def 14a", "def14a", "14a"]),
    ("10-K", ["10-k", "10k", "10_k"]),
    ("10-Q", ["10-q", "10q", "10_q"]),
    ("8-K", ["8-k", "8k", "8_k"]),
    ("ARS", ["ars"]),
]

_YEAR = re.compile(r"(?:19|20)\d{2}")


def extract_filing_metadata(doc_id: str, source: str = "") -> dict:
    """Return ``{form_type, company, ticker, fiscal_period}`` best-effort.

    Only populates company/ticker when a form type was detected (i.e. the name
    reads like a filing); otherwise everything stays empty so the citation UI
    degrades honestly to ``doc_id``.
    """
    name = f"{doc_id} {source}".lower()
    meta = {"form_type": "", "company": "", "ticker": "", "fiscal_period": ""}

    for canon, aliases in _FORM_ALIASES:
        if any(a in name for a in aliases):
            meta["form_type"] = canon
            break

    m = _YEAR.search(name)
    if m:
        meta["fiscal_period"] = m.group(0)

    if meta["form_type"]:
        lead = re.split(r"[_\-.]", doc_id)[0].strip()
        lead = re.sub(r"\s+", " ", lead)
        if lead and not lead.isdigit():
            meta["company"] = lead
        t = re.match(r"([A-Z]{1,5})(?=[_\-.]|$)", doc_id)
        if t:
            meta["ticker"] = t.group(1)
    return meta
