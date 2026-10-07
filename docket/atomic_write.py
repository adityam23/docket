"""The ONE atomic-file-replace primitive (CLAUDE.md "Reusability"), shared by
``config_write.py`` (``.env.local``) and ``agent/trace.py`` (the trace/chat
JSONL stores).

A sibling, *uniquely*-named temp file (``tempfile.mkstemp`` in the same
directory as the destination) is written in full, then ``os.replace``d over
the destination. Two properties fall out of that:

- a crash mid-write never leaves a half-written file (the destination is
  untouched until the final, atomic rename); and
- two callers racing never collide on the temp file itself — unlike a
  *fixed* ``path + ".tmp"``, which is what this module replaces. FastAPI runs
  sync endpoints concurrently in a threadpool, so two concurrent writers to
  the same store (two browser tabs, or the SSE stream racing the POST
  fallback) could otherwise both open/truncate the one shared tmp path and
  have the loser's ``os.replace`` raise ``FileNotFoundError`` after the other
  already moved it away (B6).

This only fixes the *temp-file* race. A caller still needs its own lock
around the full read-modify-write cycle to avoid a lost update (two writers
both reading the old content before either writes back) — see
``agent/trace.py``'s per-path lock.
"""

from __future__ import annotations

import os
import tempfile


def atomic_write_text(path: str, text: str, *, mode: int | None = None) -> None:
    """Atomically replace ``path`` with ``text``.

    ``mode`` (e.g. ``0o600``) is applied to the temp file before any content
    is written, so a secrets file is never world-readable even for an instant.
    Left as ``None`` (default) leaves whatever ``tempfile.mkstemp`` applies
    (already owner-only on every platform docket supports).
    """
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{os.path.basename(path)}.", suffix=".tmp")
    try:
        if mode is not None:
            os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
