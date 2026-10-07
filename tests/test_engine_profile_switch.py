"""Offline, deterministic tests for the D5 switch primitive in
``docket/engine.py``: ``read_status_meta`` (shared with EngineSupervisor) and
``switch_profile`` (PUT + poll ready + verify active/!switching, bounded by a
timeout). No real engine or network -- ``engine._get``/``engine._put`` are
monkeypatched fakes, same pattern as ``test_engine_supervisor.py``.
"""

from __future__ import annotations

import pytest

import docket.engine as engine_mod
from docket.config import Settings
from docket.engine import EngineSwitchError, read_status_meta, switch_profile


class _Resp:
    def __init__(self, code: int, body: dict):
        self.status_code = code
        self._body = body

    def json(self):
        return self._body


# -- read_status_meta ---------------------------------------------------


def test_read_status_meta_happy_path(monkeypatch):
    monkeypatch.setattr(
        engine_mod, "_get",
        lambda *a, **k: _Resp(200, {"active": "retrieval", "switching": False}),
    )
    assert read_status_meta(Settings()) == ("retrieval", False)


def test_read_status_meta_unreachable_or_non_200_is_unknown(monkeypatch):
    monkeypatch.setattr(engine_mod, "_get", lambda *a, **k: None)
    assert read_status_meta(Settings()) == (None, None)

    monkeypatch.setattr(engine_mod, "_get", lambda *a, **k: _Resp(503, {}))
    assert read_status_meta(Settings()) == (None, None)


# -- switch_profile -------------------------------------------------------


def test_switch_profile_succeeds_when_ready_confirms_the_target(monkeypatch):
    calls = {"put": 0, "get": []}

    def fake_put(settings, path, json_body, timeout=10.0):
        calls["put"] += 1
        assert path == "/engine/profiles/active"
        assert json_body == {"name": "chat"}
        return _Resp(202, {"status": "activating", "profile": "chat"})

    def fake_get(settings, path, timeout=5.0):
        calls["get"].append(path)
        if path == "/engine/ready":
            return _Resp(200, {"phase": {"phase": "ready"}})
        if path == "/engine/status":
            return _Resp(200, {"active": "chat", "switching": False})
        raise AssertionError(path)

    monkeypatch.setattr(engine_mod, "_put", fake_put)
    monkeypatch.setattr(engine_mod, "_get", fake_get)

    switch_profile(Settings(), "chat")  # must not raise
    assert calls["put"] == 1
    assert calls["get"] == ["/engine/ready", "/engine/status"]


def test_switch_profile_rejects_outright_without_polling(monkeypatch):
    def fake_put(settings, path, json_body, timeout=10.0):
        return _Resp(404, {"error": "unknown profile"})

    def fake_get_must_not_be_called(*a, **k):
        raise AssertionError("must not poll after an outright rejection")

    monkeypatch.setattr(engine_mod, "_put", fake_put)
    monkeypatch.setattr(engine_mod, "_get", fake_get_must_not_be_called)

    with pytest.raises(EngineSwitchError):
        switch_profile(Settings(), "chat")


def test_switch_profile_accepts_409_already_switching(monkeypatch):
    """409 (a switch to the same target already in flight, from elsewhere)
    means "proceed to confirmation", not rejection."""

    def fake_put(settings, path, json_body, timeout=10.0):
        return _Resp(409, {"error": "switch already in progress"})

    def fake_get(settings, path, timeout=5.0):
        if path == "/engine/ready":
            return _Resp(200, {})
        if path == "/engine/status":
            return _Resp(200, {"active": "chat", "switching": False})
        raise AssertionError(path)

    monkeypatch.setattr(engine_mod, "_put", fake_put)
    monkeypatch.setattr(engine_mod, "_get", fake_get)
    switch_profile(Settings(), "chat")  # must not raise


def test_switch_profile_never_settles_on_a_200_still_carrying_the_old_profile(monkeypatch):
    """The ready-200-but-wrong-active race the mettle team fixed on their
    side (2026-10-06): `/engine/ready` answering 200 while `/engine/status`
    still names the OUTGOING profile must not be treated as success."""
    state = {"n": 0}

    def fake_put(settings, path, json_body, timeout=10.0):
        return _Resp(202, {})

    def fake_get(settings, path, timeout=5.0):
        if path == "/engine/ready":
            return _Resp(200, {})
        if path == "/engine/status":
            state["n"] += 1
            # First poll still reports the outgoing profile; second poll has
            # caught up.
            active = "retrieval" if state["n"] == 1 else "chat"
            return _Resp(200, {"active": active, "switching": False})
        raise AssertionError(path)

    monkeypatch.setattr(engine_mod, "_put", fake_put)
    monkeypatch.setattr(engine_mod, "_get", fake_get)
    monkeypatch.setattr(engine_mod.time, "sleep", lambda s: None)

    switch_profile(Settings(), "chat")  # must not raise -- just polls again
    assert state["n"] == 2


def test_switch_profile_raises_on_timeout(monkeypatch):
    def fake_put(settings, path, json_body, timeout=10.0):
        return _Resp(202, {})

    def fake_get(settings, path, timeout=5.0):
        if path == "/engine/ready":
            return _Resp(503, {})  # never ready
        raise AssertionError(path)

    clock = {"t": 0.0}

    def fake_monotonic():
        clock["t"] += 1.0
        return clock["t"]

    monkeypatch.setattr(engine_mod, "_put", fake_put)
    monkeypatch.setattr(engine_mod, "_get", fake_get)
    monkeypatch.setattr(engine_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(engine_mod.time, "sleep", lambda s: None)

    with pytest.raises(EngineSwitchError):
        switch_profile(Settings(), "chat", timeout_s=3.0)


def test_switch_profile_default_timeout_comes_from_settings(monkeypatch):
    seen_deadline_checks = []

    def fake_put(settings, path, json_body, timeout=10.0):
        return _Resp(202, {})

    def fake_get(settings, path, timeout=5.0):
        if path == "/engine/ready":
            return _Resp(503, {})
        raise AssertionError(path)

    clock = {"t": 0.0}

    def fake_monotonic():
        clock["t"] += 1.0
        seen_deadline_checks.append(clock["t"])
        return clock["t"]

    monkeypatch.setattr(engine_mod, "_put", fake_put)
    monkeypatch.setattr(engine_mod, "_get", fake_get)
    monkeypatch.setattr(engine_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(engine_mod.time, "sleep", lambda s: None)

    settings = Settings(engine_switch_timeout_s=2.0)
    with pytest.raises(EngineSwitchError):
        switch_profile(settings, "chat")  # no explicit timeout_s -> Settings'
    # deadline = first monotonic() call (1.0) + 2.0 = 3.0; raises once a later
    # monotonic() call reaches/exceeds it.
    assert seen_deadline_checks[-1] >= 3.0
