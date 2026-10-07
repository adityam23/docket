"""Application service layer — the ONE place that assembles a retriever + provider
and answers a question. The CLI, the web ``/ask`` endpoint, and (later) the eval
harness all go through here so the wiring exists exactly once (CLAUDE.md).
"""

from __future__ import annotations

import time

from .agent.graph import Answer, answer
from .agent.trace import ChatStore, StepSink, TraceCollector, TraceStore
from .config import Settings, load_settings
from .ingest.index import Corpus
from .providers.base import SlotUnavailable
from .providers.router import get_provider
from .retrieval.rerank_client import make_reranker
from .retrieval.retriever import Retriever


def health(settings: Settings | None = None) -> dict:
    """Probe the configured backend. The ONE health implementation, shared by the
    CLI and the web ``/health`` endpoint. Never raises — degradation is data."""
    s = settings or load_settings()
    try:
        info = get_provider(s).health()
        return {"ok": True, "provider": "mettle", **info}
    except Exception as e:  # noqa: BLE001 - report failure as data, don't crash the probe
        return {"ok": False, "provider": "mettle", "error": str(e)}


def load_retriever(
    settings: Settings | None = None,
    *,
    corpus: Corpus | None = None,
    scope_doc: str | None = None,
) -> Retriever:
    """Build a Retriever over the persisted lineage, pointed at the engine's
    collection. Search mode and rerank are engine capabilities — mettle serves
    both; an engine without them fails loudly (cutover contract §1–2).
    ``scope_doc`` restricts engine search to one filing's chunks (the search
    allowlist), for per-document harnesses like full-RAG FinanceBench."""
    s = settings or load_settings()
    corpus = corpus if corpus is not None else Corpus.load(s.index_dir)
    provider = get_provider(s)
    return Retriever(
        provider,
        corpus,
        collection=s.engine_collection,
        mode=s.retrieval_mode,
        reranker=make_reranker(provider),
        scope_doc=scope_doc,
        candidates=s.retrieval_k,
    )


def remove_document(doc_id: str, *, settings: Settings | None = None) -> int:
    """Delete a document from the engine's collection and the local lineage.

    The single write-side counterpart to ingest — CLI and web both call here so
    add/remove share one implementation (CLAUDE.md). Each docket chunk lives in
    the engine as its own document (id = chunk_id), so removal is one engine
    DELETE per chunk plus the lineage rewrite. Engine unreachable → loud
    failure (no silent local-only removal). Returns chunks removed (0 if the id
    was not present in the lineage).

    Every chunk must be confirmed deleted engine-side before the lineage is
    touched. ``provider.delete_document`` returning False means the engine does
    not hold that chunk (404) — with a correctly-encoded chunk id (see
    ``openai_compat._path``) that is a genuine out-of-sync state, not a routine
    "already gone", so it is NOT treated as success. If any chunk comes back
    missing we raise, naming the offending chunk ids, and leave the document's
    *entire* lineage untouched rather than removing it: ``Corpus`` only removes
    whole documents, not individual chunks, so a half-removed doc can't be
    represented — keeping the full lineage means a retry (once the engine is
    back in sync) deletes every chunk again rather than silently reporting
    success while some chunks were never actually removed.
    """
    s = settings or load_settings()
    corpus = Corpus.load(s.index_dir)
    chunk_ids = [ch.chunk_id for ch in corpus.chunks if ch.doc_id == doc_id]
    if not chunk_ids:
        return 0
    provider = get_provider(s)
    missing = [cid for cid in chunk_ids if not provider.delete_document(s.engine_collection, cid)]
    if missing:
        raise RuntimeError(
            f"engine does not hold {len(missing)} of {len(chunk_ids)} chunk(s) "
            f"for document {doc_id!r} — lineage kept for retry: {', '.join(missing)}"
        )
    removed = corpus.remove(doc_id)
    if removed:
        corpus.save(s.index_dir)
    return removed


def ask(
    question: str,
    *,
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    on_step: StepSink | None = None,
    history: list[dict] | None = None,
) -> Answer:
    """Answer a question over the ingested corpus (grounded, cited, scored).

    ``on_step`` (optional) forwards trace events to a sink — used by the traced
    web path and the live SSE stream; the CLI leaves it None and is unaffected.
    ``history`` (optional) threads prior conversation turns into the prompt
    (T21); the graph bounds it before use."""
    s = settings or load_settings()
    retriever = retriever if retriever is not None else load_retriever(s)
    provider = get_provider(s)
    try:
        return answer(
            question, retriever=retriever, provider=provider, settings=s, on_step=on_step, history=history
        )
    except SlotUnavailable as e:
        # The engine dropped/shrunk the requested model slot (mettle profile
        # switch / VRAM pressure): surface the reason and offer the fix instead of
        # a bare 500. The trace's earlier steps were already recorded by on_step.
        detail = {
            "reason": e.reason,
            "missing": e.missing,
            "profile_degraded": e.profile_degraded,
        }
        msg = (
            "The requested model isn't resident on the engine — the active profile "
            f"dropped it ({e.reason}"
            + (f": {', '.join(e.missing)}" if e.missing else "")
            + "). Switch the engine to a profile that includes it, then retry."
        )
        return Answer(question=question, answer=msg, reliability="unknown",
                      surprisal=float("nan"), detail=detail)


def trace_store(settings: Settings | None = None) -> TraceStore:
    """The per-chat trace store under ``index_dir`` (one place — CLAUDE.md)."""
    s = settings or load_settings()
    return TraceStore(s.index_dir)


def chat_store(settings: Settings | None = None) -> ChatStore:
    """The Chat/Session store under ``index_dir`` (T21) — chats group turns,
    each turn keeping its own trace in :func:`trace_store`."""
    s = settings or load_settings()
    return ChatStore(s.index_dir)


def no_such_chat(session_id: str) -> str:
    """The one "unknown chat session" message (B6) — used by this module's own
    ``ValueError`` below and reused by the web layer's 404s so the wording
    never drifts between the two call sites."""
    return f"no such chat session: {session_id}"


def session_history(
    session_id: str,
    *,
    settings: Settings | None = None,
    chats: ChatStore | None = None,
    traces: TraceStore | None = None,
) -> list[dict]:
    """Derive prior conversation turns for a session as ``[{role, content}]``.

    The chat record holds light turn summaries; the ANSWER text lives only in
    its trace (one source of truth), so it is joined back from the trace store.
    Turns whose trace was pruned from the ring buffer are skipped — history
    simply forgets them rather than inventing content. The graph applies the
    context-window trim; this returns the full available thread.
    """
    chats = chats if chats is not None else chat_store(settings)
    traces = traces if traces is not None else trace_store(settings)
    chat = chats.get(session_id)
    if chat is None:
        raise ValueError(no_such_chat(session_id))
    out: list[dict] = []
    for turn in chat.get("turns", []):
        t = traces.get(turn.get("trace_id") or "")
        if not t:
            continue
        out.append({"role": "user", "content": t.get("question") or turn.get("question") or ""})
        if t.get("answer"):
            out.append({"role": "assistant", "content": t["answer"]})
    return out


def ask_traced(
    question: str,
    *,
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    on_step: StepSink | None = None,
    session_id: str | None = None,
) -> tuple[Answer, dict]:
    """Answer AND persist a full trace, returning ``(answer, stored_trace)``.

    A :class:`TraceCollector` records every stage verbatim; an optional external
    ``on_step`` (e.g. the SSE live sink) is fanned out alongside. The record is
    appended to the capped trace store so Observability / the per-chat page can
    replay it. Timing wraps the whole answer so ``elapsed_ms`` matches ``/api/ask``.

    With ``session_id`` (T21) the ask runs inside a first-class **Chat**: prior
    turns are threaded into the model's context (bounded by the graph's window),
    the stored trace is nested under the chat (``chat_id`` + ``turn_index``),
    and the turn joins the chat record. Unknown sessions raise ``ValueError``
    (the web layer maps that to 404).
    """
    s = settings or load_settings()
    store = trace_store(s)
    chats = chat_store(s)  # one instance for the whole call (B6) — not re-built per use
    collector = TraceCollector(question)

    history: list[dict] = []
    if session_id:
        history = session_history(session_id, settings=s, chats=chats, traces=store)

    def sink(kind: str, info: dict) -> None:
        collector(kind, info)
        if on_step is not None:
            on_step(kind, info)

    t0 = time.monotonic()
    result = ask(question, settings=s, retriever=retriever, on_step=sink, history=history or None)
    elapsed_ms = round((time.monotonic() - t0) * 1000)

    chat_id = session_id
    turn_index = None
    if session_id:
        chat = chats.get(session_id)
        turn_index = len(chat["turns"]) if chat else 0

    trace = collector.build(
        result,
        trace_id=store.next_id(collector.created_at),
        elapsed_ms=elapsed_ms,
        chat_id=chat_id,
        turn_index=turn_index,
    )
    store.append(trace)

    if session_id:
        chats.append_turn(
            session_id,
            {
                "trace_id": trace["id"],
                "question": question,
                "reliability": result.reliability,
                "created_at": collector.created_at,
            },
            title=question,
        )
    return result, trace
