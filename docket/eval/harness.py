"""Eval harness — RAGAS (reference-free RAG
metrics) for batch golden-set scoring + DeepEval (pytest) as the CI deploy gate
+ Promptfoo for the small-vs-frontier comparison; traced in Langfuse. Also the
benchmark that proves the thesis (FinanceBench + RULER). TODO(phase-3)."""

from __future__ import annotations

METRICS = ("faithfulness", "answer_relevancy", "context_precision",
           "context_recall", "tool_call_correctness", "hallucination_flag_precision")


def run_golden_set(*args, **kwargs):  # noqa: ANN002, ANN003
    """TODO(phase-3): score the golden set; emit Langfuse traces + CI gate."""
    raise NotImplementedError("eval harness not wired yet")
