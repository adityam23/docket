"""Offline tests for Settings' own domain validation (B5 #1) and the frozen
sidecar's config isolation (B5 #6). ``tests/conftest.py``'s autouse fixture
already keeps every ``Settings()`` call here off the real ``.env``/``.env.local``
(env_file forced empty), so these construct ``Settings`` directly with init
kwargs, which validate immediately regardless of any env default."""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

import docket.config as config_module
from docket.config import Settings


def test_retrieval_mode_rejects_unknown_value():
    with pytest.raises(ValidationError):
        Settings(retrieval_mode="foo")  # previously accepted -> every ask 500s
    assert Settings(retrieval_mode="keyword").retrieval_mode == "keyword"


@pytest.mark.parametrize(
    "field,value",
    [
        ("chunk_words", 0),
        ("context_chunks", 0),
        ("retrieval_k", 0),
        ("max_hops", 0),
        ("request_timeout_s", 0.0),
        ("engine_startup_window_s", 0.0),
        ("chunk_overlap", -1),
    ],
)
def test_non_positive_knobs_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_chunk_overlap_must_be_less_than_chunk_words():
    with pytest.raises(ValidationError):
        Settings(chunk_words=100, chunk_overlap=100)  # equal -> rejected
    with pytest.raises(ValidationError):
        Settings(chunk_words=100, chunk_overlap=150)  # greater -> rejected
    s = Settings(chunk_words=100, chunk_overlap=99)
    assert s.chunk_overlap == 99 and s.chunk_words == 100


def test_engine_collection_matches_mettle_regex():
    for bad in (".leading-dot", "has space", "has/slash", "a" * 129, ""):
        with pytest.raises(ValidationError):
            Settings(engine_collection=bad)
    for good in ("docket", "my.collection", "my-collection_1", "a" * 128):
        assert Settings(engine_collection=good).engine_collection == good


def test_retrieval_k_default_is_40():
    # Matches the retriever's hard-coded `candidates=40` default (retrieval.py)
    # — the knob is not yet wired through (separate ticket), but its default
    # should not lie about what happens today.
    assert Settings().retrieval_k == 40


def test_frozen_build_never_adopts_the_checkout_root(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert config_module._project_root() is None
