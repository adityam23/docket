"""Per-chat trace capture + a small local trace store (docs/decisions Q12/Q13).

The agent loop (``graph.answer``) already builds the exact prompts, runs the model
and scores reliability — but until now none of it was recorded, so Observability
could not show *how* an answer was produced. This module adds a single optional
``on_step`` sink (mirroring ``ingest/pipeline.on_event``): when present the loop
emits a step for every stage (retrieve → context → prompt → response → refine →
final), when absent the loop is byte-for-byte unchanged (CLI + offline tests).

- ``TraceCollector`` is an ``on_step`` implementation that accumulates the full,
  verbatim steps and builds a JSON-serialisable trace record.
- ``compact_step`` derives the light event the live (SSE) view streams, so the
  browser gets order + progress without shipping every prompt on every step.
- ``TraceStore`` persists records as a JSONL **ring buffer** (newest kept, oldest
  pruned past ``cap``) under ``index_dir`` — single-user, local, no broker.

**Never store secrets.** Steps hold prompts, retrieved chunks and model output;
API keys live only in config and are redacted everywhere else — nothing here
reads them.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Callable, Optional

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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp_id(prefix: str, created_at: str, seq: int) -> str:
    """Stable, sortable id from timestamp + a per-store sequence (no RNG needed)."""
    stamp = created_at.replace(":", "").replace("-", "").replace("+0000", "").replace("T", "-")
    return f"{prefix}-{stamp}-{seq:04d}"


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


def _trace_id(created_at: str, seq: int) -> str:
    return _stamp_id("trace", created_at, seq)


class TraceStore:
    """JSONL ring buffer of trace records under ``index_dir`` (cap oldest-pruned).

    Single-user/local, so a plain rewrite-on-append is fine — the cap bounds the
    file, and the whole point is a short recent history for Observability.
    """

    def __init__(self, index_dir: str, *, cap: int = 200) -> None:
        self.path = os.path.join(index_dir, "traces.jsonl")
        self.cap = cap

    def next_id(self, created_at: str) -> str:
        """Allocate an id; sequence is the current record count (monotone enough
        for a single-process, single-user store)."""
        return _trace_id(created_at, len(_read_jsonl(self.path)) + 1)

    def append(self, trace: dict) -> None:
        records = _read_jsonl(self.path)
        records.append(trace)
        records = records[-self.cap :]  # ring buffer: keep newest ``cap``
        _write_jsonl(self.path, records)

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

    def get(self, trace_id: str) -> dict | None:
        for r in _read_jsonl(self.path):
            if r.get("id") == trace_id:
                return r
        return None

    def delete(self, trace_id: str) -> bool:
        """Remove one trace by id. Returns False if it wasn't there."""
        records = _read_jsonl(self.path)
        kept = [r for r in records if r.get("id") != trace_id]
        if len(kept) == len(records):
            return False
        _write_jsonl(self.path, kept)
        return True

    def delete_by_chat(self, chat_id: str) -> int:
        """Remove every trace belonging to a chat (cascade when a Chat is deleted).
        Returns how many were removed."""
        records = _read_jsonl(self.path)
        kept = [r for r in records if r.get("chat_id") != chat_id]
        removed = len(records) - len(kept)
        if removed:
            _write_jsonl(self.path, kept)
        return removed


def _chat_id(created_at: str, seq: int) -> str:
    return _stamp_id("chat", created_at, seq)


class ChatStore:
    """JSONL ring buffer of first-class Chat/Session records (TODO T21).

    A **Chat** groups an ordered list of **turns**; each turn carries its own
    ``trace_id`` — the full record stays nested in the :class:`TraceStore`
    (chat → turns → trace). The answer text lives ONLY in the trace (one source
    of truth); this store keeps light turn summaries so Observability can render
    the nesting and the ask path can re-thread prior turns as context.
    """

    def __init__(self, index_dir: str, *, cap: int = 100) -> None:
        self.path = os.path.join(index_dir, "chats.jsonl")
        self.cap = cap

    def next_id(self, created_at: str) -> str:
        return _chat_id(created_at, len(_read_jsonl(self.path)) + 1)

    def create(self, chat_id: str, title: str, *, now: Optional[str] = None) -> dict:
        chat = {
            "id": chat_id,
            "title": title,
            "created_at": now or _utc_now(),
            "updated_at": now or _utc_now(),
            "turns": [],
        }
        self._put(chat)
        return chat

    def append_turn(
        self,
        chat_id: str,
        turn: dict,
        *,
        title: Optional[str] = None,
        now: Optional[str] = None,
    ) -> dict | None:
        """Create-or-update: append one turn to the chat (first turn names it)."""
        chat = self.get(chat_id)
        if chat is None:
            chat = self.create(chat_id, title or turn.get("question", ""), now=now)
        chat["turns"].append(turn)
        chat["updated_at"] = now or _utc_now()
        if title and not chat.get("title"):
            chat["title"] = title
        self._put(chat)
        return chat

    def _put(self, chat: dict) -> None:
        records = [c for c in _read_jsonl(self.path) if c.get("id") != chat["id"]]
        records.append(chat)
        records = records[-self.cap :]
        _write_jsonl(self.path, records)

    def delete(self, chat_id: str) -> bool:
        """Remove a chat record. Traces keep their (now historical) ``chat_id`` —
        they remain individually addressable via ``TraceStore``."""
        records = _read_jsonl(self.path)
        kept = [c for c in records if c.get("id") != chat_id]
        if len(kept) == len(records):
            return False
        _write_jsonl(self.path, kept)
        return True

    def list(self) -> list[dict]:
        """Chat summaries with their turn lists, newest-first."""
        out = [c for c in _read_jsonl(self.path)]
        out.sort(key=lambda c: c.get("updated_at") or c.get("created_at") or "", reverse=True)
        return out

    def get(self, chat_id: str) -> dict | None:
        for c in _read_jsonl(self.path):
            if c.get("id") == chat_id:
                return c
        return None
