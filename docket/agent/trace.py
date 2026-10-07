"""Per-chat trace capture + a small local trace/chat store.

The agent loop (``graph.answer``) builds the exact prompts, runs the model and
scores reliability; this module records all of it so Observability can show
*how* an answer was produced. It adds a single optional ``on_step`` sink
(mirroring ``ingest/pipeline.on_event``): when present the loop emits a step
for every stage (retrieve → context → prompt → response → refine → final),
when absent the loop is byte-for-byte unchanged (CLI + offline tests).

- ``TraceCollector`` is an ``on_step`` implementation that accumulates the full,
  verbatim steps and builds a JSON-serialisable trace record.
- ``compact_step`` derives the light event the live (SSE) view streams, so the
  browser gets order + progress without shipping every prompt on every step.
- ``_JsonlStore`` is the ONE JSONL ring-buffer implementation (id allocation,
  get/delete, a per-path lock around every read-modify-write) shared by
  ``TraceStore`` and ``ChatStore`` (T21) — single-user, local, no broker.

**Never store secrets.** Steps hold prompts, retrieved chunks and model output;
API keys live only in config and are redacted everywhere else — nothing here
reads them.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from typing import TYPE_CHECKING, Callable, Optional

from ..atomic_write import atomic_write_text
from ..ingest.chunk import FILING_FIELDS

if TYPE_CHECKING:  # avoid an import cycle at runtime (graph imports this module)
    from .graph import Answer

# on_step(kind, info): kind is the stage name, info a JSON-safe dict. Optional
# everywhere — passing None keeps the loop silent (CLAUDE.md: one orchestration).
StepSink = Callable[[str, dict], None]


def hit_view(h: dict) -> dict:
    """The ONE shape a retrieval hit takes in any trace/observability payload.

    Reused by the trace steps AND the ``/api/ask`` hit list (web/app.py) so a hit
    is projected to JSON in exactly one place. Full chunk text (chunks are already
    word-bounded) so the trace can show the whole quoted span — the always-present
    text fallback for verification when no original PDF is retained (T13).
    """
    return {
        "chunk_id": h.get("chunk_id"),
        "doc_id": h.get("doc_id"),
        "page": h.get("page"),
        **{f: h.get(f) or "" for f in FILING_FIELDS},
        "score": h.get("score"),
        "rank": h.get("rank"),
        "text": h.get("text") or "",
    }


# --- shared JSONL persistence (ONE implementation for both stores — CLAUDE.md)


def _read_jsonl(path: str) -> list[dict]:
    try:
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out
    except FileNotFoundError:
        return []


def _write_jsonl(path: str, records: list[dict]) -> None:
    """Atomically rewrite ``path`` from ``records`` via the shared
    ``atomic_write_text`` (unique temp file + ``os.replace`` — see that
    module). Callers still need the per-path lock below around the full
    read-modify-write cycle; this alone only stops two writers colliding on
    the temp file, not a lost update between two reads."""
    atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in records))


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.Lock] = {}


def _lock_for(path: str) -> threading.Lock:
    """One lock per store file, shared by every store instance pointed at it.

    ``TraceStore``/``ChatStore`` are constructed fresh per request (see
    ``service.trace_store``/``chat_store``), so the lock can't live on the
    instance — two sibling instances for the same path (two concurrent
    requests) must block on the SAME lock object, keyed here by path. Guards
    the read-modify-write races: FastAPI runs sync endpoints concurrently in a
    threadpool, so two concurrent asks (two tabs, or the SSE stream racing the
    POST fallback) could otherwise both read the old records before either
    writes back, silently losing one (B6)."""
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[path] = lock
        return lock


def _stamp_id(prefix: str, created_at: str) -> str:
    """Collision-free, still human-sortable: a timestamp-derived segment (ids
    from the same store roughly sort by creation order) plus a random suffix.

    Previously the suffix was a per-store sequence number derived from the
    current record count: it pinned at ``cap`` once the ring buffer filled
    (next id was always "count+1"), and two records minted within the same
    one-second timestamp resolution collided outright — ``get``/``delete``
    would then resolve to the wrong record, and a chat's turn could link the
    wrong trace (B6). A random suffix needs no shared counter to race on."""
    stamp = created_at.replace(":", "").replace("-", "").replace("+0000", "").replace("T", "-")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:12]}"


class TraceCollector:
    """Accumulates verbatim steps for one ask and builds a trace record.

    Instances are callable as an ``on_step`` sink: ``collector(kind, info)``.
    """

    def __init__(self, question: str, *, now: Optional[str] = None) -> None:
        self.question = question
        self.created_at = now if now is not None else _utc_now()
        self.steps: list[dict] = []

    def __call__(self, kind: str, info: dict) -> None:
        self.steps.append({"kind": kind, **info})

    def build(
        self,
        answer: "Answer",
        *,
        trace_id: str,
        elapsed_ms: int,
        chat_id: Optional[str] = None,
        turn_index: Optional[int] = None,
    ) -> dict:
        surprisal = None if answer.surprisal != answer.surprisal else round(answer.surprisal, 4)
        return {
            "id": trace_id,
            "question": self.question,
            "created_at": self.created_at,
            # T21 nesting: when the ask ran inside a Chat session, the trace is
            # nested under it (chat → turns → trace).
            "chat_id": chat_id,
            "turn_index": turn_index,
            "answer": answer.answer,
            "reliability": answer.reliability,
            "surprisal": surprisal,
            "citations": [c.model_dump() for c in answer.citations],
            "queries": answer.queries or [self.question],
            "hops": len(answer.queries or [self.question]),
            "elapsed_ms": elapsed_ms,
            "detail": answer.detail,
            "path": "aggregate" if answer.detail else "qa",
            "steps": self.steps,
        }


# Light per-step event for the live view — enough to render the timeline as it
# runs (kind, hop, the query, and counts) without streaming full prompts.
def compact_step(kind: str, info: dict) -> dict:
    out: dict = {"kind": kind}
    for key in ("hop", "query", "path", "op"):
        if key in info:
            out[key] = info[key]
    if "hits" in info and isinstance(info["hits"], list):
        out["hits"] = len(info["hits"])
    if "text" in info and isinstance(info["text"], str):
        out["chars"] = len(info["text"])
    if "items" in info and isinstance(info["items"], list):
        out["items"] = len(info["items"])
    return out


class _JsonlStore:
    """Shared plumbing for a capped JSONL ring-buffer store — the ONE
    implementation behind both :class:`TraceStore` and :class:`ChatStore`
    (CLAUDE.md "Reusability"): id allocation, ``get``/``delete``, and a
    per-path lock around every read-modify-write.

    Reads (``get``, and each store's own ``list``) run lock-free: writes go
    through ``atomic_write_text``'s ``os.replace``, so a concurrent reader
    always sees either the fully-old or fully-new file, never a torn one —
    there is nothing a read needs the lock for. Mutations (``delete``, and
    each subclass's append/replace) DO need it, to avoid a lost update when
    two writers both read the old records before either writes back (B6).
    """

    def __init__(self, index_dir: str, filename: str, *, cap: int, prefix: str) -> None:
        self.path = os.path.join(index_dir, filename)
        self.cap = cap
        self.prefix = prefix
        self._lock = _lock_for(self.path)

    def next_id(self, created_at: str) -> str:
        """Mint a collision-free id (see ``_stamp_id``) — pure, no disk I/O,
        so there is no shared counter for concurrent callers to race on."""
        return _stamp_id(self.prefix, created_at)

    def get(self, record_id: str) -> dict | None:
        for r in _read_jsonl(self.path):
            if r.get("id") == record_id:
                return r
        return None

    def delete(self, record_id: str) -> bool:
        """Remove one record by id. Returns False if it wasn't there."""
        with self._lock:
            records = _read_jsonl(self.path)
            kept = [r for r in records if r.get("id") != record_id]
            if len(kept) == len(records):
                return False
            _write_jsonl(self.path, kept)
            return True

    def _append(self, record: dict) -> None:
        """Append one record, enforcing the ring-buffer cap, under the lock."""
        with self._lock:
            records = _read_jsonl(self.path)
            records.append(record)
            _write_jsonl(self.path, records[-self.cap :])

    def _replace(self, record_id: str, build) -> dict:
        """Create-or-update one record by id, in a single locked
        read-modify-write: ``build(existing)`` (``existing`` is ``None`` for a
        create) returns the record to store. Used for ``ChatStore`` so a
        create-or-append is one atomic cycle rather than a separate ``get``
        then a separate write (which would itself race)."""
        with self._lock:
            records = _read_jsonl(self.path)
            existing = next((r for r in records if r.get("id") == record_id), None)
            new = build(existing)
            kept = [r for r in records if r.get("id") != record_id]
            kept.append(new)
            _write_jsonl(self.path, kept[-self.cap :])
            return new


class TraceStore(_JsonlStore):
    """JSONL ring buffer of trace records under ``index_dir`` (cap oldest-pruned).

    Single-user/local, so a plain rewrite-on-append (guarded by the shared
    per-path lock) is fine — the cap bounds the file, and the whole point is a
    short recent history for Observability.
    """

    def __init__(self, index_dir: str, *, cap: int = 200) -> None:
        super().__init__(index_dir, "traces.jsonl", cap=cap, prefix="trace")

    def append(self, trace: dict) -> None:
        self._append(trace)

    def list(self) -> list[dict]:
        """Summaries (no steps), newest first — the Observability chat list."""
        out = []
        for r in _read_jsonl(self.path):
            out.append(
                {
                    "id": r.get("id"),
                    "question": r.get("question"),
                    "created_at": r.get("created_at"),
                    "reliability": r.get("reliability"),
                    "surprisal": r.get("surprisal"),
                    "hops": r.get("hops"),
                    "elapsed_ms": r.get("elapsed_ms"),
                    "path": r.get("path"),
                    "chat_id": r.get("chat_id"),
                    "turn_index": r.get("turn_index"),
                }
            )
        out.reverse()
        return out

    def delete_by_chat(self, chat_id: str) -> int:
        """Remove every trace belonging to a chat (cascade when a Chat is deleted).
        Returns how many were removed."""
        with self._lock:
            records = _read_jsonl(self.path)
            kept = [r for r in records if r.get("chat_id") != chat_id]
            removed = len(records) - len(kept)
            if removed:
                _write_jsonl(self.path, kept)
            return removed


class ChatStore(_JsonlStore):
    """JSONL ring buffer of first-class Chat/Session records (T21).

    A **Chat** groups an ordered list of **turns**; each turn carries its own
    ``trace_id`` — the full record stays nested in the :class:`TraceStore`
    (chat → turns → trace). The answer text lives ONLY in the trace (one source
    of truth); this store keeps light turn summaries so Observability can render
    the nesting and the ask path can re-thread prior turns as context.
    """

    def __init__(self, index_dir: str, *, cap: int = 100) -> None:
        super().__init__(index_dir, "chats.jsonl", cap=cap, prefix="chat")

    def new(self, title: str = "") -> dict:
        """Mint a brand-new, empty chat: id + both timestamps from ONE
        ``_utc_now()`` call (previously ``create`` took an optional ``now`` but
        the ``/api/chats`` route minted its own via a separate ``_utc_now()``
        for the id and relied on ``create``'s default for the timestamps —
        two calls a request apart could disagree)."""
        now = _utc_now()
        return self.create(self.next_id(now), title, now=now)

    def create(self, chat_id: str, title: str, *, now: Optional[str] = None) -> dict:
        ts = now or _utc_now()
        chat = {"id": chat_id, "title": title, "created_at": ts, "updated_at": ts, "turns": []}
        return self._replace(chat_id, lambda _existing: chat)

    def append_turn(
        self,
        chat_id: str,
        turn: dict,
        *,
        title: Optional[str] = None,
        now: Optional[str] = None,
    ) -> dict:
        """Create-or-update: append one turn to the chat (first turn names it),
        in one locked read-modify-write cycle."""

        def build(existing: dict | None) -> dict:
            ts = now or _utc_now()
            chat = dict(existing) if existing is not None else {
                "id": chat_id,
                "title": title or turn.get("question", ""),
                "created_at": ts,
                "updated_at": ts,
                "turns": [],
            }
            chat["turns"] = [*chat.get("turns", []), turn]
            chat["updated_at"] = ts
            if title and not chat.get("title"):
                chat["title"] = title
            return chat

        return self._replace(chat_id, build)

    def list(self) -> list[dict]:
        """Chat summaries with their turn lists, newest-first."""
        out = [c for c in _read_jsonl(self.path)]
        out.sort(key=lambda c: c.get("updated_at") or c.get("created_at") or "", reverse=True)
        return out
