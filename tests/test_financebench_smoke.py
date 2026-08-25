"""Offline tests for the FinanceBench generator smoke harness — pure Python, no
network, no real model (a scripted fake provider stands in)."""

from __future__ import annotations

import json
from pathlib import Path

from docket.eval.financebench import (
    _match_value,
    _numbers,
    fb_to_item,
    run_smoke,
    score_financebench,
    score_item,
    to_sft_example,
)
from docket.providers.base import Capability, ChatResult

SMOKE_SET = Path(__file__).resolve().parents[1] / "benchmarks" / "financebench" / "smoke_set.jsonl"


def test_number_parsing_normalises_magnitudes():
    assert _numbers("$4,820 million")[0] == 4820 * 1e6
    assert _numbers("revenue of $4.82 billion")[0] == 4.82 * 1e9
    assert _numbers("grew 17.3%")[0] == 17.3
    assert _numbers("no digits here") == []


def test_match_value_tolerance():
    assert _match_value("about $4.82 billion", 4820000000, tol=0.005)
    assert _match_value("17.3%", 17.27, tol=0.06)
    assert not _match_value("it was 4110 million", 4820000000, tol=0.005)


def test_score_item_value_and_refuse():
    value_item = {"id": "v", "kind": "value", "gold": {"value": 612000000, "tol": 0.005}}
    good = score_item(value_item, "Net income was $612 million [1].", 10)
    assert good.ok and good.cited and not good.hallucinated

    wrong = score_item(value_item, "Net income was $588 million [1].", 10)
    assert not wrong.ok and wrong.hallucinated  # confident but wrong

    refuse_item = {"id": "r", "kind": "refuse", "gold": {"answer": "The documents do not contain this."}}
    refused = score_item(refuse_item, "The documents do not contain this.", 10)
    assert refused.ok and not refused.hallucinated
    leaked = score_item(refuse_item, "Headcount was 5,000 employees.", 10)
    assert not leaked.ok and leaked.hallucinated  # answered when it should refuse


def _oracle_answer(item: dict) -> str:
    """Derive the canonical correct response from an item's gold — set-agnostic,
    so the test doesn't need updating when the smoke set grows."""
    g, kind = item["gold"], item["kind"]
    if kind == "refuse":
        return "The documents do not contain this."
    if kind == "value":
        return f"The answer is {g['value']} [1]."
    return f"The answer is {g['answer']} [1]."


def test_run_smoke_scores_full_set_with_scripted_provider():
    items = [json.loads(l) for l in SMOKE_SET.read_text().splitlines() if l.strip()]

    class Oracle:
        name = "oracle"
        capabilities = Capability.CHAT

        def health(self):
            return {}

        def chat(self, messages, **kw):
            # The grounded prompt ends with "Question: <q>" — match exactly (endswith)
            # so questions that are substrings of others don't collide.
            user = messages[-1]["content"]
            for it in items:
                if user.endswith("Question: " + it["question"]):
                    return ChatResult(text=_oracle_answer(it))
            return ChatResult(text="?")

        def embed(self, texts):
            return []

    card = run_smoke(Oracle(), items, model_label="oracle")
    assert card.n == len(items)
    assert card.accuracy == 1.0            # oracle nails every item
    assert card.hallucinated == 0
    assert card.refuse_ok == card.refuse_total
    assert card.cited == card.citable      # cited every answerable item


# --- real FinanceBench mapping + scoring (offline; no dataset download) -------

_FB_RECORD = {
    "financebench_id": "financebench_id_03029",
    "company": "3M",
    "doc_name": "3M_2018_10K",
    "question_type": "metrics-generated",
    "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
    "answer": "$1577.00",
    "evidence": [{"evidence_text": "Purchases of property, plant and equipment (PP&E) (1,577)"}],
}


def test_fb_to_item_maps_real_schema():
    it = fb_to_item(_FB_RECORD)
    assert it["id"] == "financebench_id_03029"
    assert it["kind"] == "financebench"
    assert it["gold"]["answer"] == "$1577.00"
    assert "1,577" in it["evidence"][0]
    assert it["meta"]["question_type"] == "metrics-generated"
    # missing evidence degrades gracefully, never crashes
    assert fb_to_item({"question": "q", "answer": "a"})["evidence"]


def test_score_financebench_numeric_first():
    it = fb_to_item(_FB_RECORD)
    good = score_financebench(it, "FY2018 capex was $1,577 million [1].")
    assert good.ok and good.reason == "numeric match"
    bad = score_financebench(it, "It was $2,000 million [1].")
    assert not bad.ok and bad.hallucinated


def test_score_financebench_judge_fallback_for_qualitative():
    it = fb_to_item({"financebench_id": "x", "question": "Is liquidity healthy?",
                     "answer": "Yes, liquidity is strong.", "evidence": []})

    class YesJudge:
        name = "judge"
        capabilities = Capability.CHAT

        def health(self):
            return {}

        def chat(self, messages, **kw):
            return ChatResult(text="CORRECT")

        def embed(self, texts):
            return []

    r = score_financebench(it, "The company has ample liquidity.", judge=YesJudge())
    assert r.ok and r.reason == "judge: correct"


def test_to_sft_example_matches_grounded_contract():
    items = [json.loads(l) for l in SMOKE_SET.read_text().splitlines() if l.strip()]
    by_kind = {it["kind"]: it for it in items}

    ex = to_sft_example(by_kind["value"])
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "Context:" in ex["messages"][1]["content"]
    assert "Question:" in ex["messages"][1]["content"]
    assert "[1]" in ex["messages"][2]["content"]              # cited target

    refuse_ex = to_sft_example(by_kind["refuse"])
    assert refuse_ex["messages"][2]["content"] == "The documents do not contain this."

    fb_ex = to_sft_example(fb_to_item(_FB_RECORD))
    assert "$1577.00" in fb_ex["messages"][2]["content"] and "[1]" in fb_ex["messages"][2]["content"]
