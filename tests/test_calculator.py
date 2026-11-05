"""The calculator: ``tools.calculate`` (safe arithmetic) and the two-pass
``graph.grounded_chat`` protocol that lets the model request it."""

from __future__ import annotations

import pytest

from docket.agent.graph import answer, grounded_chat
from docket.agent.tools import calculate
from docket.agent.trace import TraceCollector
from docket.config import Settings
from docket.eval.financebench import generate_financebench

from .fakes import ScriptedProvider, acme_chunks, retriever_over


@pytest.mark.parametrize(
    ("expr", "value"),
    [
        ("(23601-16434)/16434*100", (23601 - 16434) / 16434 * 100),
        ("23,601 - 16,434", 7167.0),
        ("$1,000.5 * 2", 2001.0),
        ("-2**3", -8.0),
        ("7 % 4", 3.0),
        ("10 − 4 × 2 ÷ 4", 8.0),  # typographic minus/times/divide
    ],
)
def test_calculate_evaluates_arithmetic(expr, value):
    assert calculate(expr) == pytest.approx(value)


@pytest.mark.parametrize(
    "expr",
    [
        '__import__("os").system("x")',  # names/calls never evaluate
        "revenue - cost",
        "(1).__class__",
        "1, 5",
        "2 ** 999",          # exponent bound
        "9 ** 9 ** 9",
        "(-8) ** 0.5",        # complex result
        "1e308 * 10",         # overflow to inf
        "1" * 400,            # length bound
        "",
    ],
)
def test_calculate_rejects_anything_but_bounded_arithmetic(expr):
    with pytest.raises(ValueError):
        calculate(expr)


def test_calculate_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        calculate("1/0")


def _two_pass(first: str, final: str) -> ScriptedProvider:
    """Answers ``first`` to the initial grounded prompt and ``final`` once the
    calculator results have been fed back."""
    return ScriptedProvider(
        lambda msgs: final if "Calculator results" in msgs[-1]["content"] else first
    )


def test_grounded_chat_without_calc_is_a_single_call():
    p = ScriptedProvider("Revenue was $10M [1].")
    res, calcs = grounded_chat(p, [{"role": "user", "content": "q"}], max_tokens=50)
    assert res.text == "Revenue was $10M [1]." and calcs == []
    assert len(p.calls) == 1


def test_grounded_chat_evaluates_calcs_and_feeds_exact_results_back():
    p = _two_pass(
        "Growth was <calc>(23601-16434)/16434*100</calc>% [1]; bad <calc>1/0</calc>",
        "Growth was 43.6% [1].",
    )
    steps = []
    res, calcs = grounded_chat(
        p, [{"role": "user", "content": "q"}], max_tokens=50,
        on_step=lambda kind, info: steps.append((kind, info)),
    )
    assert res.text == "Growth was 43.6% [1]."
    assert calcs == [
        {"expression": "(23601-16434)/16434*100", "result": "43.6108068638"},
        {"expression": "1/0", "result": "error: float division by zero"},
    ]
    # Second pass sees the draft plus the exact results.
    followup = p.calls[1]
    assert followup[-2]["role"] == "assistant" and "<calc>" in followup[-2]["content"]
    assert "(23601-16434)/16434*100 = 43.6108068638" in followup[-1]["content"]
    assert [k for k, _ in steps] == ["calc"]


def test_grounded_chat_resolves_tags_left_in_the_final_pass():
    p = _two_pass("<calc>2+2</calc>", "Total <calc>2+2</calc>, double <calc>4*2</calc>.")
    res, _ = grounded_chat(p, [{"role": "user", "content": "q"}], max_tokens=50)
    assert res.text == "Total 4, double 8."


def test_answer_records_calculations_on_answer_and_trace():
    r = retriever_over(acme_chunks())
    p = _two_pass("Revenue <calc>10-4</calc> [1]", "Revenue rose by 6 [1].")
    collector = TraceCollector("What was revenue growth?")
    res = answer("What was revenue growth?", retriever=r, provider=p,
                 settings=Settings(), on_step=collector)
    assert res.answer == "Revenue rose by 6 [1]."
    assert res.calculations == [{"expression": "10-4", "result": "6"}]
    trace = collector.build(res, trace_id="t1", elapsed_ms=1)
    assert trace["calculations"] == res.calculations
    assert "calc" in [s["kind"] for s in trace["steps"]]


def test_system_prompt_offers_the_calculator():
    p = ScriptedProvider("ok")
    r = retriever_over(acme_chunks())
    answer("What was revenue?", retriever=r, provider=p, settings=Settings())
    assert "<calc>" in p.calls[-1][0]["content"]


def test_financebench_direct_path_evaluates_calcs():
    p = _two_pass("Margin <calc>50/200*100</calc>%", "Margin 25%.")
    item = {"question": "Margin?", "evidence": ["Net income 50, revenue 200."], "doc_id": "X"}
    [(_, text, _, _)] = list(generate_financebench(p, [item], max_tokens=50))
    assert text == "Margin 25%."
