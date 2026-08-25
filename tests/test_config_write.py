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
    cw.update_config({"chat_model": "my-model", "retrieval_k": 9, "provider": "groq"})
    text = envfile.read_text()
    assert "DK_CHAT_MODEL=my-model" in text
    assert "DK_RETRIEVAL_K=9" in text
    assert "DK_PROVIDER=groq" in text

    # A Settings pointed at this overlay picks the values up (usable next request).
    s = Settings(_env_file=str(envfile))
    assert s.chat_model == "my-model" and s.retrieval_k == 9 and s.provider.value == "groq"


def test_update_config_perms_0600(tmp_path, monkeypatch):
    envfile = _redirect(monkeypatch, tmp_path)
    cw.update_config({"chat_model": "m"})
    assert stat.S_IMODE(os.stat(envfile).st_mode) == 0o600


def test_update_config_clears_empty_embed_url(tmp_path, monkeypatch):
    envfile = _redirect(monkeypatch, tmp_path)
    cw.update_config({"embed_url": "http://x/v1"})
    assert "DK_EMBED_URL=http://x/v1" in envfile.read_text()
    cw.update_config({"embed_url": ""})  # empty → remove the override
    assert "DK_EMBED_URL" not in envfile.read_text()


def test_update_config_rejects_unknown_and_bad_type(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    with pytest.raises(cw.ConfigWriteError):
        cw.update_config({"nonsense_key": 1})           # not in the allowlist
    with pytest.raises(cw.ConfigWriteError):
        cw.update_config({"cerebras_api_key": "x"})     # secrets are not knob-writable
    with pytest.raises(cw.ConfigWriteError):
        cw.update_config({"retrieval_k": "not-an-int"})  # type violation


def test_set_and_clear_secret(tmp_path, monkeypatch):
    envfile = _redirect(monkeypatch, tmp_path)
    cw.set_secret("groq", "sk-abc123")
    assert "DK_GROQ_API_KEY=sk-abc123" in envfile.read_text()
    assert stat.S_IMODE(os.stat(envfile).st_mode) == 0o600

    cw.clear_secret("groq")
    assert "DK_GROQ_API_KEY" not in envfile.read_text()


def test_secret_rejects_unknown_empty_and_injection(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path)
    with pytest.raises(cw.ConfigWriteError):
        cw.set_secret("unknown_provider", "x")
    with pytest.raises(cw.ConfigWriteError):
        cw.set_secret("groq", "")                       # empty
    with pytest.raises(cw.ConfigWriteError):
        cw.set_secret("groq", "line1\nAA_EVIL=1")       # newline injection guard


def test_config_view_redacts_secrets(monkeypatch):
    from docket.web.observability import config_view
    import json

    monkeypatch.setenv("DK_GROQ_API_KEY", "super-secret-xyz")
    cfg = config_view()
    assert cfg["api_keys"]["groq"] is True
    assert "super-secret-xyz" not in json.dumps(cfg)     # never surfaced, only a bool
