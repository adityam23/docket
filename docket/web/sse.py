"""The one SSE-frame formatter (CLAUDE.md: one concept, one implementation).

Two endpoints hand-built their own SSE frame text with f-strings before this
existed (``/api/ask/stream`` and the engine-events proxy) — one of them broke
on a payload containing a quote (``{{"message": "{e}"}}``), because an
f-string has no idea it's sitting inside a JSON string. ``sse()`` always goes
through ``json.dumps`` for a non-str payload, so arbitrary text is always
valid framing.
"""

from __future__ import annotations

import json
from typing import Any


def sse(data: Any, *, event: str | None = None) -> str:
    """Format one Server-Sent-Events frame. ``data`` is JSON-encoded unless
    it is already a ``str`` (sent verbatim, e.g. a pre-formatted payload) —
    either way the result is one valid frame, terminated by the blank line
    SSE requires between events."""
    payload = data if isinstance(data, str) else json.dumps(data)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"
