"""Offline tests that the phase coordinator is actually wired into the engine
call path (ticket D5 point 3/5): ``providers.router.PhasedProvider`` enters
CHAT for ``chat`` and RETRIEVAL for every retrieval/ingest call, ``health``/
``ocr`` enter neither, and the agent's ``answer()`` flow produces the minimal
phase-switch sequence for a plain QA ask, the aggregate path, and an ask that
triggers one re-query hop.

No real engine or network: ``PhaseCoordinator`` is given injected
``switch_fn``/``status_fn`` fakes (same pattern as ``test_phase.py``), and the
provider surface is ``tests/fakes.py``'s ``FakeEngine`` -- CLAUDE.md: tests
stay offline & deterministic.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from docket.agent.graph import answer
from docket.config import Settings
from docket.ingest.index import Corpus
from docket.phase import Phase, PhaseCoordinator
from docket.providers.router import PhasedProvider
from docket.retrieval.retriever import Retriever

from .fakes import FakeEngine, acme_chunks


class _RecordingCoordinator:
    """A minimal coordinator double: records which ``Phase`` each wrapped
    call entered, without any of the real switching/priority machinery --
    isolates ``PhasedProvider``'s own wiring from ``PhaseCoordinator``'s
    concurrency logic (covered separately in ``test_phase.py``)."""

    def __init__(self) -> None:
        self.entries: list[Phase] = []

    @contextmanager
    def enter(self, phase: Phase):
        self.entries.append(phase)
        yield


def _wrapped(chunks, **engine_kw):
    engine = FakeEngine(chunks, **engine_kw)
    coord = _RecordingCoordinator()
    return PhasedProvider(engine, coord), coord, engine


def test_chat_enters_chat_phase():
    provider, coord, _ = _wrapped(acme_chunks())
    provider.chat([{"role": "user", "content": "hi"}])
    assert coord.entries == [Phase.CHAT]


def test_retrieval_calls_enter_retrieval_phase():
    chunks = acme_chunks()
    provider, coord, engine = _wrapped(chunks, rank="all", rerank_fn=lambda q, d: [1.0] * len(d))
    # FakeEngine (tests/fakes.py) doesn't model embed() -- it's a thin fake of
    # the collection/chat surface only; stub it here rather than widening the
    # shared fixture for one test.
    engine.embed = lambda texts: [[0.0]] * len(texts)
    engine.delete_collection = lambda collection: True

    provider.embed(["x"])
    provider.rerank("q", ["a", "b"])
    provider.ensure_collection("docket")
    provider.collection_search("docket", "revenue", mode="hybrid", k=5)
    provider.post_document("docket", "doc#1", "text", {})
    provider.delete_document("docket", "doc#1")
    provider.delete_collection("docket")

    assert coord.entries == [Phase.RETRIEVAL] * 7


def test_health_and_ocr_enter_no_phase():
    provider, coord, engine = _wrapped(acme_chunks())
    engine.ocr = lambda image_b64, instruction=None: "text"  # not modeled by FakeEngine
    provider.health()
    provider.ocr("base64==", instruction=None)
    assert coord.entries == []


def test_name_attribute_delegates_through_the_wrapper():
    # test_smoke.py's test_router_builds_local_provider relies on exactly
    # this transparent delegation (mirrors _ResilientProvider in
    # benchmarks/financebench/run_full.py).
    engine = FakeEngine(acme_chunks())
    engine.name = "mettle"
    provider = PhasedProvider(engine, _RecordingCoordinator())
    assert provider.name == "mettle"


# -- end-to-end: answer() produces the minimal phase-switch sequence --------


def _real_coordinator(initial_active: str = "retrieval"):
    calls: list[str] = []
    state = {"active": initial_active}

    def switch_fn(settings, target):
        calls.append(target)
        state["active"] = target

    def status_fn(settings):
        return state["active"], False

    settings = Settings(engine_retrieval_profile="retrieval", engine_chat_profile="chat")
    return PhaseCoordinator(settings, switch_fn=switch_fn, status_fn=status_fn), calls


def _retriever_and_provider(engine_kw: dict, coord: PhaseCoordinator):
    chunks = acme_chunks()
    corpus = Corpus()
    corpus.add(chunks)
    engine = FakeEngine(chunks, **engine_kw)
    provider = PhasedProvider(engine, coord)
    retriever = Retriever(provider, corpus, collection="docket", mode="hybrid")
    return retriever, provider


def test_plain_qa_ask_is_one_switch_retrieval_to_chat():
    coord, calls = _real_coordinator()
    retriever, provider = _retriever_and_provider(
        {"rank": "all", "text": "Revenue was $10M [1]."}, coord
    )
    settings = Settings(engine_retrieval_profile="retrieval", engine_chat_profile="chat")

    result = answer("What was revenue?", retriever=retriever, provider=provider, settings=settings)

    assert calls == ["chat"]
    assert result.answer == "Revenue was $10M [1]."


def test_aggregate_path_is_one_switch_retrieval_to_chat():
    coord, calls = _real_coordinator()
    # The extraction call's response doesn't need to parse into real items for
    # this test -- only the phase SEQUENCE is under test.
    retriever, provider = _retriever_and_provider({"rank": "all", "text": "[]"}, coord)
    settings = Settings(engine_retrieval_profile="retrieval", engine_chat_profile="chat")

    answer(
        "What is the total revenue across all documents?",
        retriever=retriever, provider=provider, settings=settings,
    )

    assert calls == ["chat"]


def test_requery_hop_switches_retrieval_chat_retrieval_chat():
    coord, calls = _real_coordinator()
    # Deliberately low-confidence logprobs (large mean surprisal -> "low"
    # reliability) so hop 0 triggers exactly one re-query/re-retrieve before
    # hop 1 (the last hop, max_hops=2) settles regardless of its own label.
    retriever, provider = _retriever_and_provider(
        {"rank": "all", "text": "Revenue was $10M [1].", "logprobs": [-2.0, -2.0, -2.0]}, coord
    )
    settings = Settings(
        engine_retrieval_profile="retrieval", engine_chat_profile="chat", max_hops=2
    )

    result = answer("What was revenue?", retriever=retriever, provider=provider, settings=settings)

    assert calls == ["chat", "retrieval", "chat"]
    assert len(result.queries) == 2  # the original question + one refined re-query


def test_contention_ingest_thread_does_not_cause_extra_switches():
    """The review's headline fix: per-call phase entry alone lets in-flight
    hit zero BETWEEN two calls that are really one logical operation (the
    hop's answer+refine chat pair; retrieve's search+rerank pair), so a
    contending thread can switch the engine away and back for nothing. With
    ``Retriever.retrieve``/``iterative_retrieve`` and the agent's per-hop
    chat block each holding their phase open (docket/phase.py same-thread
    reentrancy), a concurrently running "ingest" loop must not add any extra
    switches around a QA ask that refines once: retrieval -> chat (answer +
    refine, ONE held block, no flip between them) -> retrieval -> chat."""
    coord, calls = _real_coordinator()

    ingest_chunks = acme_chunks()
    ingest_corpus = Corpus()
    ingest_corpus.add(ingest_chunks)
    ingest_provider = PhasedProvider(FakeEngine(ingest_chunks, rank="all"), coord)
    ingest_retriever = Retriever(
        ingest_provider, ingest_corpus, collection="docket", mode="hybrid"
    )

    stop = threading.Event()

    def ingest_loop():
        while not stop.is_set():
            ingest_retriever.retrieve("ingest query", k=5)
            time.sleep(0.005)

    t = threading.Thread(target=ingest_loop, daemon=True)
    t.start()
    time.sleep(0.05)  # let ingest run a few iterations before the ask lands

    retriever, provider = _retriever_and_provider(
        {"rank": "all", "text": "Revenue was $10M [1].", "logprobs": [-2.0, -2.0, -2.0]}, coord
    )
    settings = Settings(
        engine_retrieval_profile="retrieval", engine_chat_profile="chat", max_hops=2
    )

    try:
        result = answer(
            "What was revenue?", retriever=retriever, provider=provider, settings=settings
        )
    finally:
        stop.set()
        t.join(timeout=5.0)

    # Deterministic prefix regardless of ingest's timing: one switch into
    # chat for the WHOLE answer+refine block (no "retrieval" sneaks in
    # between those two chat calls -- the bug this fix closes), one switch
    # back to retrieval for iterative_retrieve, one switch back to chat for
    # hop 1. Once current == a phase, any number of concurrent same-phase
    # entrants (ingest included) never add another switch_fn call -- that
    # invariant is what bounds this, not luck. Ingest resuming in retrieval
    # after the ask fully releases (if it wins the race before this
    # assertion runs) is correct "ask preempts, then ingest resumes"
    # behaviour, not thrash, so at most one trailing entry is allowed.
    assert calls[:3] == ["chat", "retrieval", "chat"]
    assert len(calls) <= 4
    assert len(result.queries) == 2
