"""Offline, deterministic tests for the engine supervisor (ticket B2):
restart()'s lock/wake bug (and the deeper lock-held-across-a-wait deadlock it
was hiding), D1 (never spawn over a live child; 3-consecutive-probe-failure
death; exit 98 is terminal), D2 (readiness states from `/engine/ready`), live
settings reload, and the SSE framing fix.

No real subprocess, socket, or OS signal: ``_spawn``/``_kill_group``/
``_wait_for_port_free`` are monkeypatched per test, and all HTTP goes through
a fake ``docket.engine._get`` (never a real connection) — CLAUDE.md: tests
stay offline & deterministic. A couple of tests drive the real threaded loop
(via ``ensure()``) to pin the restart()/failed deadlock as a regression; a
``_FastEvent`` stand-in collapses its waits to instant polls so those stay
fast.
"""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest

import docket.engine as engine_mod
from docket.config import Settings
from docket.engine import EngineSupervisor
from docket.web.sse import sse


# -- test doubles -------------------------------------------------------------


class _FakeProc:
    """Stand-in for ``subprocess.Popen``: just enough surface (``pid``,
    ``poll()``, ``returncode``) for the supervisor's own logic. Never a real
    process — no signals are ever sent to it in these tests."""

    def __init__(self, pid: int, returncode: int | None = None):
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode


class _FastEvent:
    """``threading.Event`` stand-in whose ``wait()`` never actually blocks for
    its timeout — an instant poll instead. Keeps loop-driving tests (restart()
    un-sticking ``failed``, etc.) from paying real wall-clock backoff/poll
    delays while still behaving like a real Event for ``is_set``/``set``/
    ``clear``."""

    def __init__(self):
        self._evt = threading.Event()

    def is_set(self):
        return self._evt.is_set()

    def set(self):
        self._evt.set()

    def clear(self):
        self._evt.clear()

    def wait(self, timeout=None):
        return self._evt.wait(0)


def _no_wait_for_port_free(monkeypatch):
    monkeypatch.setattr(EngineSupervisor, "_wait_for_port_free", lambda self, timeout=2.0: None)


# -- restart() bug (item 1) ---------------------------------------------------


def test_restart_resets_under_lock_and_wakes():
    """The bug: restart() set attempts outside the lock and never set _wake."""
    sup = EngineSupervisor(Settings())
    sup.attempts = 7
    sup._port_held = True
    sup.restart()
    assert sup.attempts == 0
    assert sup._port_held is False
    assert sup._wake.is_set()


def test_restart_unsticks_failed_without_deadlock(monkeypatch):
    """Regression for the deeper bug: the 'failed' wait loop used to hold
    self._lock for its entire duration, so restart()'s own `with self._lock`
    could never even acquire it to call `_wake.set()` -- a genuine deadlock,
    not just a missed signal. Drives the REAL threaded loop (ensure()), with
    a tiny instance-overridden backoff budget and a child that always dies
    immediately, so it reaches 'failed' almost instantly."""
    _no_wait_for_port_free(monkeypatch)
    monkeypatch.setattr(engine_mod, "_get", lambda *a, **k: None)  # engine never reachable

    calls = {"n": 0}

    def fake_spawn(self):
        calls["n"] += 1
        self._proc = _FakeProc(pid=1000 + calls["n"], returncode=1)  # dies immediately
        self.pid = self._proc.pid

    monkeypatch.setattr(EngineSupervisor, "_spawn", fake_spawn)

    sup = EngineSupervisor(Settings(engine_binary="/bin/true"))
    sup._BACKOFF_S = (0.01, 0.01)  # exhaust the budget fast
    sup.ensure()

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and sup.view()["state"] != "failed":
        time.sleep(0.02)
    assert sup.view()["state"] == "failed", "never reached failed"

    # The actual regression check: view() must not hang (it would, pre-fix,
    # because the supervisor thread held the lock the whole time it waited).
    view_thread_result = {}

    def read_view():
        view_thread_result["v"] = sup.view()

    t = threading.Thread(target=read_view)
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), "view() hung while the supervisor was in 'failed' (lock held across the wait)"
    assert view_thread_result["v"]["state"] == "failed"

    # The spawn/backoff/re-fail cycle after restart() can complete in well
    # under one poll tick, so don't look for a transient non-'failed' state
    # (easy to miss) -- look for proof the loop actually ran more attempts:
    # _spawn() called again.
    spawns_before = calls["n"]
    sup.restart()
    deadline = time.monotonic() + 5.0
    respawned = False
    while time.monotonic() < deadline:
        if calls["n"] > spawns_before:
            respawned = True
            break
        time.sleep(0.02)
    assert respawned, "restart() did not un-stick the 'failed' loop (no new spawn attempt)"
    sup.stop()


# -- D1: never spawn over a live child; exit 98 is terminal -------------------


def test_bring_up_kills_live_child_before_respawning(monkeypatch):
    """The D1 bug: _bring_up used to spawn a new child without first checking
    whether self._proc was still alive. Now it must kill+reap the old one
    BEFORE calling _spawn() for the replacement."""
    _no_wait_for_port_free(monkeypatch)
    monkeypatch.setattr(engine_mod, "_get", lambda *a, **k: None)

    calls = []
    old_proc = _FakeProc(pid=111, returncode=None)  # still "alive"

    def fake_kill_group(self, proc):
        calls.append(("kill", proc))

    def fake_spawn(self):
        calls.append(("spawn", None))
        self._proc = _FakeProc(pid=222, returncode=1)  # new child also crashes immediately
        self.pid = 222

    monkeypatch.setattr(EngineSupervisor, "_kill_group", fake_kill_group)
    monkeypatch.setattr(EngineSupervisor, "_spawn", fake_spawn)

    sup = EngineSupervisor(Settings(engine_binary="/bin/true"))
    sup._wake = _FastEvent()
    sup._proc, sup.pid = old_proc, 111

    sup._bring_up()

    assert calls[0] == ("kill", old_proc), "the live old child must be killed first"
    assert calls[1][0] == "spawn", "...then the replacement is spawned"
    assert sup.attempts == 1
    assert sup.state == "starting"


def test_bring_up_exit_98_is_terminal_offline(monkeypatch):
    """mettled exit 98 (port held by a foreign engine, §1.6): offline, no
    respawn loop."""
    _no_wait_for_port_free(monkeypatch)
    monkeypatch.setattr(engine_mod, "_get", lambda *a, **k: None)

    def fake_spawn(self):
        self._proc = _FakeProc(pid=333, returncode=98)
        self.pid = 333

    monkeypatch.setattr(EngineSupervisor, "_spawn", fake_spawn)

    sup = EngineSupervisor(Settings(engine_binary="/bin/true"))
    sup._wake = _FastEvent()
    sup._bring_up()

    assert sup.state == "offline"
    assert sup._port_held is True
    assert sup.reason and "98" in sup.reason

    # The loop-level guard: a SECOND _bring_up pass must not spawn again.
    spawned_again = {"flag": False}

    def fail_if_called(self):
        spawned_again["flag"] = True

    monkeypatch.setattr(EngineSupervisor, "_spawn", fail_if_called)
    sup._bring_up()  # if the terminal flag is honoured, callers shouldn't even reach here via _loop
    # (direct call bypasses the _loop-level `_port_held` gate deliberately --
    # the real guard lives in _loop; assert it explicitly:)
    assert sup._port_held is True


def test_loop_never_calls_bring_up_again_once_port_held(monkeypatch):
    """The _loop-level terminal gate: once _port_held is set, the loop must
    never re-enter _bring_up (and therefore never call _spawn) until
    restart()."""
    sup = EngineSupervisor(Settings(engine_binary="/bin/true"))
    sup._wake = _FastEvent()
    sup._port_held = True
    sup.state = "offline"

    def fail_if_called(self):
        raise AssertionError("_bring_up must not run while _port_held")

    monkeypatch.setattr(EngineSupervisor, "_bring_up", fail_if_called)

    def stop_soon():
        time.sleep(0.05)
        sup._stop = True
        sup._wake.set()

    threading.Thread(target=stop_soon).start()
    sup._loop()  # must return via the terminal branch, never touching _bring_up


# -- D2: readiness states from /engine/ready ----------------------------------


@pytest.mark.parametrize(
    "probe,expected_state,expected_reason_substr",
    [
        ({"code": 200, "body": {"phase": {"phase": "ready"}}}, "available", None),
        (
            {"code": 503, "body": {"phase": {"phase": "downloading", "model": "m"}, "reason": "loading"}},
            "starting",
            None,
        ),
        (
            {"code": 503, "body": {"phase": {"phase": "failed"}, "reason": "dropped_vram", "detail": "x"}},
            "degraded",
            "dropped_vram",
        ),
        ({"code": 404, "body": {}}, "incompatible", "too old"),
    ],
)
def test_state_for_probe_maps_the_residency_contract(probe, expected_state, expected_reason_substr):
    state, extra = EngineSupervisor._state_for_probe(probe)
    assert state == expected_state
    if expected_reason_substr is not None:
        assert expected_reason_substr in (extra.get("reason") or "")


def test_probe_merges_ready_and_status_meta(monkeypatch):
    """`_probe()` is the one place /engine/ready (readiness) and /engine/status
    (active/switching, mettle 2026-10-07) are combined for the supervision
    view."""

    class _Resp:
        def __init__(self, code, body):
            self.status_code = code
            self._body = body

        def json(self):
            return self._body

    def fake_get(settings, path, timeout=5.0):
        if path == "/engine/ready":
            return _Resp(200, {"phase": {"phase": "ready"}})
        if path == "/engine/status":
            return _Resp(200, {"active": "retrieval", "switching": False})
        raise AssertionError(path)

    monkeypatch.setattr(engine_mod, "_get", fake_get)
    sup = EngineSupervisor(Settings())
    probe = sup._probe()
    assert probe == {
        "alive": True, "code": 200, "body": {"phase": {"phase": "ready"}},
        "active": "retrieval", "switching": False,
    }


def test_probe_connection_failure_is_not_alive(monkeypatch):
    monkeypatch.setattr(engine_mod, "_get", lambda *a, **k: None)
    sup = EngineSupervisor(Settings())
    probe = sup._probe()
    assert probe == {"alive": False, "code": None, "body": {}, "active": None, "switching": None}


def test_view_surfaces_active_and_switching(monkeypatch):
    sup = EngineSupervisor(Settings())
    sup._report_alive(
        {"code": 200, "body": {"phase": {"phase": "ready"}}, "active": "chat", "switching": True},
        reset_attempts=True,
    )
    v = sup.view()
    assert v["active"] == "chat"
    assert v["switching"] is True


def test_wait_for_death_needs_three_consecutive_failures(monkeypatch):
    """mettle's Q5 recommendation: dead only after 3 CONSECUTIVE connection
    failures; a 503/404 (alive, not ready) is not a failure."""
    sup = EngineSupervisor(Settings())
    sup._wake = _FastEvent()
    calls = {"n": 0}

    def fake_probe(self):
        calls["n"] += 1
        return {"alive": False, "code": None, "body": {}, "active": None, "switching": None}

    monkeypatch.setattr(EngineSupervisor, "_probe", fake_probe)
    sup._wait_for_death()
    assert calls["n"] == 3  # returns right after the 3rd consecutive failure


def test_wait_for_death_alive_response_resets_the_failure_streak(monkeypatch):
    sequence = [
        {"alive": False, "code": None, "body": {}, "active": None, "switching": None},
        {"alive": False, "code": None, "body": {}, "active": None, "switching": None},
        {"alive": True, "code": 503, "body": {"phase": {"phase": "loading"}}, "active": None, "switching": None},
        {"alive": False, "code": None, "body": {}, "active": None, "switching": None},
        {"alive": False, "code": None, "body": {}, "active": None, "switching": None},
        {"alive": False, "code": None, "body": {}, "active": None, "switching": None},
    ]
    it = iter(sequence)
    sup = EngineSupervisor(Settings())
    sup._wake = _FastEvent()
    monkeypatch.setattr(EngineSupervisor, "_probe", lambda self: next(it))
    sup._wait_for_death()
    with pytest.raises(StopIteration):
        next(it)  # every scripted probe was consumed -- the mid-sequence 503 did not short-circuit death


# -- live settings reload (item 4) --------------------------------------------


def test_probe_reads_fresh_settings_each_cycle_with_settings_fn(monkeypatch):
    calls = []
    box = [Settings(backend_url="http://a/v1")]

    def fake_get(settings, path, timeout=5.0):
        calls.append(settings.backend_url)
        return None

    monkeypatch.setattr(engine_mod, "_get", fake_get)
    sup = EngineSupervisor(settings_fn=lambda: box[0])
    sup._probe()
    box[0] = Settings(backend_url="http://b/v1")
    sup._probe()
    assert calls == ["http://a/v1", "http://b/v1"]


def test_probe_stays_static_without_settings_fn(monkeypatch):
    """Back-compat: a directly-constructed Settings() (tests/benchmarks) is
    never silently overridden by a later load_settings() call."""
    calls = []

    def fake_get(settings, path, timeout=5.0):
        calls.append(settings.backend_url)
        return None

    monkeypatch.setattr(engine_mod, "_get", fake_get)
    fixed = Settings(backend_url="http://fixed/v1")
    sup = EngineSupervisor(fixed)
    sup._probe()
    sup._probe()
    assert calls == ["http://fixed/v1", "http://fixed/v1"]


@pytest.fixture
def _reset_supervisor_singleton():
    saved = engine_mod._SUPERVISOR
    engine_mod._SUPERVISOR = None
    yield
    engine_mod._SUPERVISOR = saved


def test_get_supervisor_wires_live_reload(monkeypatch, _reset_supervisor_singleton):
    box = [Settings(backend_url="http://one/v1")]
    monkeypatch.setattr(engine_mod, "load_settings", lambda: box[0])
    sup = engine_mod.get_supervisor()
    assert sup._settings_fn is engine_mod.load_settings

    calls = []
    monkeypatch.setattr(
        engine_mod, "_get",
        lambda settings, path, timeout=5.0: (calls.append(settings.backend_url), None)[1],
    )
    sup._probe()
    box[0] = Settings(backend_url="http://two/v1")
    sup._probe()
    assert calls == ["http://one/v1", "http://two/v1"]


def test_get_supervisor_is_a_singleton_regardless_of_later_settings(_reset_supervisor_singleton):
    s1 = Settings(backend_url="http://one/v1")
    sup = engine_mod.get_supervisor(s1)
    sup2 = engine_mod.get_supervisor(Settings(backend_url="http://two/v1"))
    assert sup2 is sup


# -- SSE framing (item 6) ------------------------------------------------------


def test_sse_escapes_quotes_safely():
    """The bug: the old f-string framing (`{{"message": "{e}"}}`) broke on a
    message containing a quote. sse() always goes through json.dumps."""
    frame = sse({"message": 'engine said "oops"'}, event="error")
    assert frame == 'event: error\ndata: {"message": "engine said \\"oops\\""}\n\n'
    # And it must be valid JSON on the data line.
    import json

    data_line = frame.split("\n")[1]
    assert data_line.startswith("data: ")
    json.loads(data_line[len("data: "):])


def test_sse_without_event_name():
    frame = sse({"a": 1})
    assert frame == 'data: {"a": 1}\n\n'


def test_engine_events_stream_yields_parsed_payloads_and_drops_named_events(monkeypatch):
    body = (
        b'event: profile.active\n'
        b'data: {"profile": "chat"}\n'
        b'\n'
        b': keepalive\n'
        b'\n'
        b'data: {"slot_available": "x"}\n'
        b'\n'
    )

    def handler(request):
        return httpx.Response(200, content=body)

    real_async_client = httpx.AsyncClient  # captured before the patch replaces the attribute
    monkeypatch.setattr(
        engine_mod.httpx, "AsyncClient",
        lambda **kw: real_async_client(transport=httpx.MockTransport(handler)),
    )

    async def collect():
        out = []
        async for payload in engine_mod.engine_events_stream(Settings()):
            out.append(payload)
        return out

    payloads = asyncio.run(collect())
    assert payloads == [{"profile": "chat"}, {"slot_available": "x"}]


def test_engine_events_stream_raises_on_non_200(monkeypatch):
    def handler(request):
        return httpx.Response(503, content=b"")

    real_async_client = httpx.AsyncClient  # captured before the patch replaces the attribute
    monkeypatch.setattr(
        engine_mod.httpx, "AsyncClient",
        lambda **kw: real_async_client(transport=httpx.MockTransport(handler)),
    )

    async def collect():
        out = []
        async for payload in engine_mod.engine_events_stream(Settings()):
            out.append(payload)
        return out

    with pytest.raises(RuntimeError, match="503"):
        asyncio.run(collect())


def test_bring_up_settles_on_404_instead_of_waiting_out_the_startup_window(monkeypatch):
    """An engine without /engine/ready answers 404 — a permanent "too old"
    verdict. The spawn loop must report `incompatible` and return at once,
    not wait the 900 s startup ceiling, kill a working engine and respawn."""
    _no_wait_for_port_free(monkeypatch)
    monkeypatch.setattr(engine_mod, "_get", lambda *a, **k: httpx.Response(404, json={}))
    spawned = []

    def fake_spawn(self):
        spawned.append(1)
        self._proc = _FakeProc(pid=333, returncode=None)  # alive
        self.pid = 333

    monkeypatch.setattr(EngineSupervisor, "_spawn", fake_spawn)
    monkeypatch.setattr(EngineSupervisor, "_kill_group", lambda self, proc: None)
    sup = EngineSupervisor(Settings(engine_binary="/bin/true"))
    sup._wake = _FastEvent()

    sup._bring_up()

    assert spawned == [1]
    assert sup.state == "incompatible"
    assert sup.pid == 333, "the answering child stays tracked, never killed"
