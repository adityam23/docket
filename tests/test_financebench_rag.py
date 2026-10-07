"""Offline tests for FinanceBench full-RAG mode (T26). No network: an in-memory
corpus + real Retriever provide context, and a recording fake provider lets us
assert the generator sees RETRIEVED chunks, not the gold evidence."""

from __future__ import annotations

from docket.config import Settings
from docket.eval.financebench import (
    PrefetchedRetriever,
    make_rag_retrieve,
    run_financebench,
    run_rag_via_answer,
    score_financebench,
)
from docket.ingest.chunk import chunk_pages
from docket.ingest.index import Corpus
from docket.phase import PhaseCoordinator
from docket.providers.base import ChatResult
from docket.providers.router import PhasedProvider
from docket.retrieval.retriever import Retriever

from .fakes import FakeEngine, ScriptedProvider, retriever_over


def _retriever():
    pages = [
        {"page": 1, "text": "Net revenue for fiscal 2020 was reported as forty two dollars in the filing."},
        {"page": 2, "text": "Unrelated boilerplate about corporate governance and board committees."},
    ]
    chunks = chunk_pages(pages, doc_id="ACME_2020_10K", source="acme.pdf", words=12, overlap=0)
    return retriever_over(chunks, rank="all")


def _item():
    return {
        "id": "fb-1",
        "kind": "financebench",
        "doc_id": "ACME_2020_10K",
        "question": "What was net revenue for fiscal 2020?",
        # Oracle evidence that must NOT reach the model in RAG mode:
        "evidence": ["ORACLE-ONLY-SENTINEL evidence snippet"],
        "gold": {"answer": "$42.00"},
        "meta": {"question_type": "metrics-generated", "company": "Acme"},
    }


def test_rag_feeds_retrieved_context_not_oracle():
    retriever = _retriever()
    retrieve = make_rag_retrieve(lambda doc_id: retriever, k=3)
    provider = ScriptedProvider(reply="42.0 [1]")

    card = run_financebench(provider, [_item()], retrieve=retrieve)

    assert card.n == 1
    # The model saw the RETRIEVED chunk, never the oracle sentinel.
    assert "net revenue" in provider.last_user.lower()
    assert "ORACLE-ONLY-SENTINEL" not in provider.last_user
    assert card.correct == 1  # "42.0" matches gold $42.00 numerically


def test_oracle_mode_still_feeds_gold_evidence():
    provider = ScriptedProvider(reply="42.0 [1]")
    run_financebench(provider, [_item()])  # no retrieve => oracle
    assert "ORACLE-ONLY-SENTINEL" in provider.last_user


def test_rag_missing_filing_yields_empty_context():
    # retriever_for returns None (no PDF) → no hits → empty context, no crash.
    retrieve = make_rag_retrieve(lambda doc_id: None, k=3)
    provider = ScriptedProvider(reply="42.0 [1]")
    card = run_financebench(provider, [_item()], retrieve=retrieve)
    assert card.n == 1
    assert "ORACLE-ONLY-SENTINEL" not in provider.last_user


def test_score_financebench_numeric_unit_agnostic():
    # Sanity that the shared scorer is unit-shift tolerant (regression guard).
    r = score_financebench(_item(), "The answer is $42.0 million [1].")
    # 42.0 million vs gold 42 -> scaled match within the ×1000 family.
    assert r.ok is True and r.cited is True


def test_gold_number_only_for_bare_figures():
    # Pure-figure golds (what metrics questions produce) are auto-scorable...
    from docket.eval.financebench import _gold_number

    assert _gold_number("$1616.00") == 1616.0
    assert _gold_number("-0.02") == -0.02
    assert abs(_gold_number("24.26%") - 24.26) < 1e-9
    # ...but a prose gold that merely leads with a fiscal year must NOT be reduced
    # to that year (the auto-scorer would then pass any response citing 2022).
    assert _gold_number("Yes, gross margin improved as of FY2022. 4.8% -> 5.3%.") is None
    assert _gold_number("In 2022, AMD reported higher EPYC server processor sales.") is None
    # A gold of exactly 0 ("none") is too loose to auto-match -> defer to judge.
    assert _gold_number("0") is None


def test_year_first_prose_gold_is_not_auto_passed():
    # Regression: a refusal response against a year-leading prose gold must not be
    # scored correct just because the response mentions the fiscal year.
    item = {
        "id": "fb-yr", "kind": "financebench", "doc_id": "BOEING_2022_10K",
        "question": "Does Boeing have an improving gross margin profile as of FY2022?",
        "evidence": ["x"], "gold": {"answer": "Yes. Improving as of FY2022, 4.8% -> 5.3%."},
        "meta": {"question_type": "domain-relevant", "company": "Boeing"},
    }
    r = score_financebench(item, "The documents do not contain this for 2022. [1]")
    assert r.ok is False


# --- B7: full-RAG routed through the REAL agent.graph.answer() path ----------


def test_run_rag_via_answer_routes_aggregate_questions_through_the_router():
    # An aggregate-shaped question must hit agent.graph.plan_operation's
    # aggregate/extraction path, not the plain grounded prompt the direct
    # shortcut (generate_financebench) would have sent -- proof the harness
    # actually calls agent.graph.answer(), not _grounded_messages directly.
    retriever = _retriever()
    pr = PrefetchedRetriever(retriever)
    provider = ScriptedProvider(reply='[{"i": 1, "value": "$42"}]')
    item = {
        "id": "fb-agg", "kind": "financebench", "doc_id": "ACME_2020_10K",
        "question": "What is the total revenue across all documents?",
        "evidence": ["ORACLE-ONLY-SENTINEL"], "gold": {"answer": "$42.00"},
        "meta": {"question_type": "metrics-generated", "company": "Acme"},
    }
    settings = Settings(context_chunks=3)

    card = run_rag_via_answer(
        [item], retriever_of=lambda _it: pr, provider=provider, settings=settings,
    )

    assert card.n == 1
    sent = provider.calls[0][-1]["content"]
    assert "Extract this quantity" in sent      # the aggregate path's extraction prompt
    assert "ORACLE-ONLY-SENTINEL" not in sent    # never the oracle evidence


def test_run_rag_via_answer_batches_retrieval_before_chat_except_for_a_requery_hop():
    """B7(1b): across a multi-question run, ALL retrieval happens before ANY
    chat -- except a genuine re-query hop, which legitimately switches back
    into retrieval mid-generation (it needs the model's refined query first,
    so it can never be prefetched)."""
    calls: list[str] = []
    state = {"active": "chat"}

    def switch_fn(settings, target):
        calls.append(target)
        state["active"] = target

    def status_fn(settings):
        return state["active"], False

    settings = Settings(
        engine_retrieval_profile="retrieval", engine_chat_profile="chat",
        context_chunks=3, max_hops=2,
    )
    coord = PhaseCoordinator(settings, switch_fn=switch_fn, status_fn=status_fn)

    pages = [{"page": 1, "text": "Net revenue for fiscal 2020 was forty two dollars."}]
    chunks = chunk_pages(pages, doc_id="D", source="d.pdf", words=12, overlap=0)
    corpus = Corpus()
    corpus.add(chunks)

    class _PerQuestionConfidenceEngine(FakeEngine):
        """FakeEngine only has ONE fixed logprobs config for every chat call;
        this test needs q2 (and only q2) to come back low-confidence so it
        triggers exactly one re-query hop while q1 settles immediately."""

        def chat(self, messages, **kw):
            confident = "LOWCONF" not in messages[-1]["content"]
            lp = [-0.02, -0.05, -0.03] if confident else [-2.0, -2.0, -2.0]
            return ChatResult(text=self._text, logprobs=lp, raw={})

    engine = _PerQuestionConfidenceEngine(chunks, rank="all", text="Revenue was $42 [1].")
    provider = PhasedProvider(engine, coord)
    base_retriever = Retriever(provider, corpus, collection="docket", mode="hybrid")
    prefetched = PrefetchedRetriever(base_retriever)

    items = [
        {"id": "q1", "kind": "financebench", "doc_id": "D",
         "question": "What was net revenue?", "evidence": [], "gold": {"answer": "42"},
         "meta": {}},
        {"id": "q2", "kind": "financebench", "doc_id": "D",
         "question": "What was LOWCONF revenue?", "evidence": [], "gold": {"answer": "42"},
         "meta": {}},
    ]

    run_rag_via_answer(
        items, retriever_of=lambda _it: prefetched, provider=provider, settings=settings,
    )

    # Pass 1 prefetches BOTH questions' retrieval (one switch into
    # "retrieval" -- the second prefetch is already on the right profile).
    # Pass 2 answers q1 fully with zero extra retrieval, THEN q2's hop0
    # triggers a genuine re-query (low confidence) -- one extra
    # retrieval<->chat round trip, the single documented exception to "all
    # retrieval before any chat".
    assert calls == ["retrieval", "chat", "retrieval", "chat"]
