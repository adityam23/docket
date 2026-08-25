"""Tier-1 trust layer: turn token logprobs into an uncertainty score and a
traffic-light label a non-technical user can read. Tier-2 (white-box probe) and
Tier-3 (Gemma Scope SAE lab mode) are roadmap.

Dependency-free by design so it runs everywhere and degrades gracefully.
"""

from __future__ import annotations

import math


def mean_token_surprisal(logprobs: list[float]) -> float:
    """Mean per-token surprisal in nats from top-token logprobs.
    Higher = the model was less sure. NaN if no logprobs available."""
    if not logprobs:
        return float("nan")
    return -sum(logprobs) / len(logprobs)


def label_for_surprisal(h: float) -> str:
    """Map a mean-surprisal value to a traffic-light label. The ONE place the
    thresholds live.

    THRESHOLDS ARE PLACEHOLDERS — they must be calibrated against the golden set
    with the human-in-the-loop judge. Semantic
    entropy (sample-and-cluster) will augment this in phase-2.
    """
    if math.isnan(h):
        return "unknown"
    if h < 0.30:
        return "high"
    if h < 0.80:
        return "medium"
    return "low"


def reliability_label(logprobs: list[float]) -> str:
    """🟢 high / 🟡 medium / 🔴 low from mean surprisal."""
    return label_for_surprisal(mean_token_surprisal(logprobs))


def reliability_score(logprobs: list[float]) -> tuple[str, float]:
    """Label + mean surprisal in one pass (surprisal computed once). Returned as
    a pair so callers surface both the badge and the underlying number."""
    h = mean_token_surprisal(logprobs)
    return label_for_surprisal(h), h
