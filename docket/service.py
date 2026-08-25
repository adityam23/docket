"""Application service layer — the ONE place that assembles a retriever + provider
and answers a question. The CLI, the web ``/ask`` endpoint, and (later) the eval
harness all go through here so the wiring exists exactly once (CLAUDE.md).
"""

from __future__ import annotations

import time

from .agent.graph import Answer, answer
from .agent.trace import ChatStore, StepSink, TraceCollector, TraceStore
from .config import Settings, load_settings
from .ingest.embed import embed_texts, embeddings_available
from .ingest.index import Corpus
from .providers.router import get_provider
from .retrieval.rerank_client import make_reranker, rerank_available
from .retrieval.retriever import Retriever


def health(settings: Settings | None = None) -> dict:
    """Probe the configured backend. The ONE health implementation, shared by the
    CLI and the web ``/health`` endpoint. Never raises — degradation is data."""
    s = settings or load_settings()
    try:
        info = get_provider(s).health()
        return {"ok": True, "provider": s.provider.value, **info}
    except Exception as e:  # noqa: BLE001 - report failure as data, don't crash the probe
        return {"ok": False, "provider": s.provider.value, "error": str(e)}


def load_retriever(settings: Settings | None = None, *, corpus: Corpus | None = None) -> Retriever:
    """Build a Retriever over the persisted corpus, wiring dense search only when
    a dedicated embedding endpoint is configured (otherwise sparse-only) and a
    cross-encoder reranker only when ``DK_RERANK_URL`` is set (otherwise the fused
    RRF order stands). Both are graceful, offline-safe degradations (T26)."""
    s = settings or load_settings()
    corpus = corpus if corpus is not None else Corpus.load(s.index_dir)
    embed_query = None
    if embeddings_available(s):
        embed_query = lambda q: embed_texts([q], settings=s)[0]  # noqa: E731
    reranker = make_reranker(s) if rerank_available(s) else None
    return Retriever(corpus, embed_query=embed_query, reranker=reranker)


def remove_document(doc_id: str, *, settings: Settings | None = None) -> int:
    """Delete a document (all its chunks + vectors) from the persisted corpus.

    The single write-side counterpart to ingest — CLI and web both call here so
    add/remove share one implementation (CLAUDE.md). Returns chunks removed
    (0 if the id was not present); only rewrites the index when something changed.
    """
    s = settings or load_settings()
    corpus = Corpus.load(s.index_dir)
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
    return answer(
        question, retriever=retriever, provider=provider, settings=s, on_step=on_step, history=history
    )


def trace_store(settings: Settings | None = None) -> TraceStore:
    """The per-chat trace store under ``index_dir`` (one place — CLAUDE.md)."""
    s = settings or load_settings()
    return TraceStore(s.index_dir)


def chat_store(settings: Settings | None = None) -> ChatStore:
    """The Chat/Session store under ``index_dir`` (T21) — chats group turns,
    each turn keeping its own trace in :func:`trace_store`."""
    s = settings or load_settings()
    return ChatStore(s.index_dir)


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
        raise ValueError(f"no such chat session: {session_id}")
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
    collector = TraceCollector(question)

    history: list[dict] = []
    if session_id:
        history = session_history(session_id, settings=s, chats=chat_store(s), traces=store)

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
        chat = chat_store(s).get(session_id)
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
        chat_store(s).append_turn(
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
