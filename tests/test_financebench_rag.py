"""Offline tests for FinanceBench full-RAG mode (T26). No network: an in-memory
corpus + real Retriever provide context, and a recording fake provider lets us
assert the generator sees RETRIEVED chunks, not the gold evidence."""

from __future__ import annotations

from docket.eval.financebench import (
    make_rag_retrieve,
    run_financebench,
    score_financebench,
)
from docket.ingest.chunk import chunk_pages
from docket.ingest.index import Corpus
from docket.providers.base import Capability, ChatResult
from docket.retrieval.retriever import Retriever


class _RecordingProvider:
    """Echoes the user prompt it received so tests can inspect the context."""

    name = "rec"
    capabilities = Capability.CHAT

    def __init__(self):
        self.last_user = None

    def chat(self, messages, **kw) -> ChatResult:
        self.last_user = messages[-1]["content"]
        return ChatResult(text="42.0 [1]", logprobs=None, raw={})


def _retriever():
    pages = [
        {"page": 1, "text": "Net revenue for fiscal 2020 was reported as forty two dollars in the filing."},
        {"page": 2, "text": "Unrelated boilerplate about corporate governance and board committees."},
    ]
    c = Corpus()
    c.add(chunk_pages(pages, doc_id="ACME_2020_10K", source="acme.pdf", words=12, overlap=0))
    return Retriever(c)


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
    provider = _RecordingProvider()

    card = run_financebench(provider, [_item()], retrieve=retrieve)

    assert card.n == 1
    # The model saw the RETRIEVED chunk, never the oracle sentinel.
    assert "net revenue" in provider.last_user.lower()
    assert "ORACLE-ONLY-SENTINEL" not in provider.last_user
    assert card.correct == 1  # "42.0" matches gold $42.00 numerically


def test_oracle_mode_still_feeds_gold_evidence():
    provider = _RecordingProvider()
    run_financebench(provider, [_item()])  # no retrieve => oracle
    assert "ORACLE-ONLY-SENTINEL" in provider.last_user


def test_rag_missing_filing_yields_empty_context():
    # retriever_for returns None (no PDF) → no hits → empty context, no crash.
    retrieve = make_rag_retrieve(lambda doc_id: None, k=3)
    provider = _RecordingProvider()
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
