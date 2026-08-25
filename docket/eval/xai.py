"""Explainable auto-evaluator: an XGBoost/LightGBM + SHAP
classifier over TRACE FEATURES (retrieval/rerank scores, logprob + semantic
entropy, tool-call success, hop count) predicting hallucination/correctness —
cheaper and less biased than pure LLM-judge, and it shows *why* it flagged.
Human-in-the-loop corrections retrain it. TODO(phase-3)."""

from __future__ import annotations

TRACE_FEATURES = (
    "retrieval_top_score",
    "rerank_top_score",
    "mean_token_surprisal",
    "semantic_entropy",
    "tool_call_success_rate",
    "num_hops",
    "answer_len_tokens",
)


def predict_hallucination(features: dict) -> float:
    """Return P(hallucination). TODO(phase-3): train on HITL-labelled traces."""
    raise NotImplementedError("xai auto-eval not wired yet")
