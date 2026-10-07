"""Offline tests for the .env.local writer (T01 settings persistence + T02 BYOK).
All writes are redirected to a tmp file so the real project .env.local is never
touched; asserts cover round-trip, allowlist/type rejection, 0600 perms, secret
non-echo, and injection guards."""

from __future__ import annotations

import os
import stat

import pytest

import docket.config_write as cw
from docket.config import Settings


def _redirect(monkeypatch, tmp_path):
    envfile = tmp_path / ".env.local"
    monkeypatch.setattr(cw, "_ENV_LOCAL_FILE", str(envfile))
    return envfile


def test_update_config_round_trips_and_is_readable(tmp_path, monkeypatch):
    envfile = _redirect(monkeypatch, tmp_path)
    cw.update_config({"chat_model": "my-model", "retrieval_k": 9, "retrieval_mode": "hybrid"})
    text = envfile.read_text()
    # Values are single-quoted on disk (see _quote_env_value) — still readable
    # by a human, and what makes '#', '"', '$', spaces, '=' round-trip exactly.
    assert "DK_CHAT_MODEL='my-model'" in text
    assert "DK_RETRIEVAL_K='9'" in text
    assert "DK_RETRIEVAL_MODE='hybrid'" in text

    # A Settings pointed at this overlay picks the values up (usable next request).
    s = Settings(_env_file=str(envfile))
    assert s.chat_model == "my-model" and s.retrieval_k == 9 and s.retrieval_mode == "hybrid"


def test_update_config_perms_0600(tmp_path, monkeypatch):
    envfile = _redirect(monkeypatch, tmp_path)
    cw.update_config({"chat_model": "m"})
    assert stat.S_IMODE(os.stat(envfile).st_mode) == 0o600


def test_update_config_clears_empty_engine_collection(tmp_path, monkeypatch):
    envfile = _redirect(monkeypatch, tmp_path)
    cw.update_config({"engine_collection": "filings"})
    assert "DK_ENGINE_COLLECTION='filings'" in envfile.read_text()
    cw.update_config({"engine_collection": ""})  # empty → remove the override
    assert "DK_ENGINE_COLLECTION" not in envfile.read_text()


def test_update_config_rejects_unknown_and_bad_type(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    with pytest.raises(cw.ConfigWriteError):
        cw.update_config({"nonsense_key": 1})           # not in the allowlist
    with pytest.raises(cw.ConfigWriteError):
        cw.update_config({"provider_api_key": "x"})     # secrets are not knob-writable
    with pytest.raises(cw.ConfigWriteError):
        cw.update_config({"retrieval_k": "not-an-int"})  # type violation


def test_set_and_clear_secret(tmp_path, monkeypatch):
    envfile = _redirect(monkeypatch, tmp_path)
    cw.set_secret("engine", "sk-abc123")
    assert "DK_ENGINE_KEY='sk-abc123'" in envfile.read_text()
    assert stat.S_IMODE(os.stat(envfile).st_mode) == 0o600

    cw.clear_secret("engine")
    assert "DK_ENGINE_KEY" not in envfile.read_text()


def test_secret_rejects_unknown_empty_and_injection(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    with pytest.raises(cw.ConfigWriteError):
        cw.set_secret("unknown_provider", "x")
    with pytest.raises(cw.ConfigWriteError):
        cw.set_secret("engine", "")                     # empty
    with pytest.raises(cw.ConfigWriteError):
        cw.set_secret("engine", "line1\nAA_EVIL=1")     # newline injection guard


@pytest.mark.parametrize(
    "value",
    [
        "abc #def",                       # '#' -> would start a comment unquoted
        'has "double quotes" inside',      # '"' -> a bare value starting with it
        #                                     is a (different) quote style
        "has 'single quotes' inside",      # "'" -> the quote char this format uses
        "weird=chars and spaces here",     # '=' / spaces -> second '=' / re-split
        "a lone $ and a $(not-brace) too",  # '$' without '{' -> never interpolated
        "trailing backslash \\",
    ],
    ids=["hash", "double-quote", "single-quote", "equals-and-spaces", "dollar", "backslash"],
)
def test_update_config_round_trips_special_chars_through_real_settings(tmp_path, monkeypatch, value):
    envfile = _redirect(monkeypatch, tmp_path)
    cw.update_config({"chat_model": value})
    s = Settings(_env_file=str(envfile))
    assert s.chat_model == value


def test_update_config_rejects_dollar_brace_interpolation(tmp_path, monkeypatch):
    # python-dotenv expands '${NAME}' unconditionally -- even single-quoted --
    # and it cannot be escaped (verified against the installed parser), so a
    # value containing it would silently read back as something else entirely
    # on the next load. Reject it outright instead.
    _redirect(monkeypatch, tmp_path)
    with pytest.raises(cw.ConfigWriteError):
        cw.update_config({"chat_model": "prefix-${HOME}-suffix"})


def test_config_view_redacts_secrets(monkeypatch):
    from docket.web.observability import config_view
    import json

    monkeypatch.setenv("DK_ENGINE_KEY", "super-secret-xyz")
    cfg = config_view()
    assert cfg["api_keys"]["engine"] is True
    assert "super-secret-xyz" not in json.dumps(cfg)     # never surfaced, only a bool


def test_config_view_reuses_secret_status(monkeypatch):
    """One dict builder for 'which keys are set' (CLAUDE.md reusability) --
    observability.config_view must call config_write.secret_status rather than
    re-deriving the same {"engine": bool(...)} shape itself."""
    import docket.web.observability as obs

    sentinel = {"engine": "sentinel"}
    monkeypatch.setattr(obs, "secret_status", lambda: sentinel)
    assert obs.config_view()["api_keys"] is sentinel
