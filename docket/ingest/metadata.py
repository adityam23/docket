"""Best-effort SEC filing metadata extraction (T22).

Parses form type / company / ticker / fiscal period from a document's id or
source filename. Heuristic and deliberately conservative: anything that does
not look like an SEC filing yields empty strings, and callers fall back to the
bare ``doc_id`` (never invent identity).
"""

from __future__ import annotations

import os
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

# A standalone 4-digit year (19xx/20xx) — NOT embedded in a longer digit run
# (a CIK, an accession number, ...). The lookaround asserts no digit directly
# before/after the match, so "0000320193" never yields "2019".
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def _alias_pattern(alias: str) -> re.Pattern:
    """Whole-word match for a form alias: no identifier character (letter,
    digit, or space — aliases like "def 14a" are themselves multi-word)
    directly before/after it, so e.g. "ars" doesn't match inside "lars"."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")


def extract_filing_metadata(doc_id: str, source: str = "") -> dict:
    """Return ``{form_type, company, ticker, fiscal_period}`` best-effort.

    Only populates company/ticker when a form type was detected (i.e. the name
    reads like a filing); otherwise everything stays empty so the citation UI
    degrades honestly to ``doc_id``.

    Matches on ``doc_id`` and the source's BASENAME only — never the full
    path — so an incidental substring in a directory name (e.g. a username
    "lars" containing "ars") or a digit run in a parent directory can't be
    mistaken for filing identity (the module's rule: never invent it).
    """
    name = f"{doc_id} {os.path.basename(source)}".lower()
    meta = {"form_type": "", "company": "", "ticker": "", "fiscal_period": ""}

    for canon, aliases in _FORM_ALIASES:
        if any(_alias_pattern(a).search(name) for a in aliases):
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
