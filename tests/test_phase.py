"""Offline, deterministic tests for the engine phase coordinator (ticket D5):
no switch while a request is in flight, CHAT preemption priority, switch
timeout/rejection propagation, the empty-profile-names no-op, the lazy
startup "already correct" check, same-phase nested reentrancy not
deadlocking, cross-phase nested entry raising instead of deadlocking, the
stale-cache-on-failed-switch fix, and that a switch/status probe never holds
the lock across its (simulated) network I/O.

No real engine, no network: ``switch_fn``/``status_fn`` are injected fakes
(CLAUDE.md: tests stay offline & deterministic), mirroring how
``test_engine_supervisor.py`` fakes ``docket.engine._get``.
"""

from __future__ import annotations

import threading
import time

import pytest

from docket.config import Settings
from docket.phase import Phase, PhaseCoordinator, PhaseDeadlockError


def _settings(**kw) -> Settings:
    return Settings(engine_retrieval_profile="retrieval", engine_chat_profile="chat", **kw)


def _coordinator(initial_active: str | None = "retrieval", *, switch_fn=None, status_fn=None, settings=None):
    """A coordinator wired to an in-memory fake engine state: ``state["active"]``
    tracks what the "engine" would report, updated by the fake switch."""
    calls: list[str] = []
    state = {"active": initial_active, "switching": False}

    def _switch(s, target):
        if switch_fn is not None:
            switch_fn(s, target)  # may raise, simulating rejection/timeout
        calls.append(target)
        state["active"] = target

    def _status(s):
        if status_fn is not None:
            return status_fn(s)
        return state["active"], state["switching"]

    coord = PhaseCoordinator(settings or _settings(), switch_fn=_switch, status_fn=_status)
    return coord, calls


# -- no switch while in flight -----------------------------------------------


def test_no_switch_while_request_in_flight():
    coord, calls = _coordinator(initial_active="retrieval")

    entered_retrieval = threading.Event()
    release_retrieval = threading.Event()

    def hold_retrieval():
        with coord.enter(Phase.RETRIEVAL):
            entered_retrieval.set()
            release_retrieval.wait(timeout=5.0)

    t = threading.Thread(target=hold_retrieval, daemon=True)
    t.start()
    assert entered_retrieval.wait(timeout=2.0)

    chat_done = threading.Event()

    def do_chat():
        with coord.enter(Phase.CHAT):
            pass
        chat_done.set()

    t2 = threading.Thread(target=do_chat, daemon=True)
    t2.start()
    # The retrieval hold is still open -- no switch must happen yet.
    time.sleep(0.3)
    assert calls == []
    assert not chat_done.is_set()

    release_retrieval.set()
    t.join(timeout=5.0)
    assert chat_done.wait(timeout=5.0)
    t2.join(timeout=5.0)
    assert calls == ["chat"]


# -- chat priority / preemption ----------------------------------------------


def test_chat_waiter_blocks_new_retrieval_entrants_until_served():
    coord, calls = _coordinator(initial_active="retrieval")

    entered_retrieval = threading.Event()
    release_retrieval = threading.Event()

    def hold_retrieval():
        with coord.enter(Phase.RETRIEVAL):
            entered_retrieval.set()
            release_retrieval.wait(timeout=5.0)

    t_r1 = threading.Thread(target=hold_retrieval, daemon=True)
    t_r1.start()
    assert entered_retrieval.wait(timeout=2.0)

    chat_entered = threading.Event()
    chat_released = threading.Event()

    def do_chat():
        with coord.enter(Phase.CHAT):
            chat_entered.set()
            chat_released.wait(timeout=5.0)

    t_chat = threading.Thread(target=do_chat, daemon=True)
    t_chat.start()
    time.sleep(0.2)  # let the chat waiter register itself
    assert not chat_entered.is_set()

    # A brand-new retrieval entrant must now block too (chat has priority),
    # even though the engine is still nominally on "retrieval".
    new_retrieval_entered = threading.Event()

    def new_retrieval():
        with coord.enter(Phase.RETRIEVAL):
            new_retrieval_entered.set()

    t_r2 = threading.Thread(target=new_retrieval, daemon=True)
    t_r2.start()
    time.sleep(0.3)
    assert not new_retrieval_entered.is_set(), "new retrieval entrant jumped ahead of a waiting chat request"

    # Release the held retrieval request: chat should now switch in and run.
    release_retrieval.set()
    t_r1.join(timeout=5.0)
    assert chat_entered.wait(timeout=5.0)
    assert calls == ["chat"]

    # The new retrieval entrant must still be waiting -- chat hasn't finished.
    time.sleep(0.2)
    assert not new_retrieval_entered.is_set()

    chat_released.set()
    t_chat.join(timeout=5.0)
    assert new_retrieval_entered.wait(timeout=5.0)
    t_r2.join(timeout=5.0)
    assert calls == ["chat", "retrieval"]


# -- switch timeout / rejection propagates loudly ----------------------------


def test_switch_failure_propagates_as_is():
    class _Boom(RuntimeError):
        pass

    def failing_switch(settings, target):
        raise _Boom(f"switch to {target} timed out")

    coord, calls = _coordinator(initial_active="retrieval", switch_fn=failing_switch)
    with pytest.raises(_Boom):
        with coord.enter(Phase.CHAT):
            pass  # pragma: no cover - never reached


# -- empty profile names = no-op ---------------------------------------------


def test_empty_profile_names_disable_switching():
    def never(*a, **kw):
        raise AssertionError("must not be called when switching is disabled")

    settings = Settings(engine_retrieval_profile="", engine_chat_profile="")
    coord = PhaseCoordinator(settings, switch_fn=never, status_fn=never)

    with coord.enter(Phase.RETRIEVAL):
        pass
    with coord.enter(Phase.CHAT):
        pass  # neither call touched switch_fn/status_fn -- no AssertionError


# -- startup: already correct profile -> no PUT, status read once -----------


def test_startup_already_on_target_profile_skips_switch():
    status_calls = {"n": 0}

    def status_fn(settings):
        status_calls["n"] += 1
        return "retrieval", False

    def never_switch(settings, target):
        raise AssertionError("must not switch when already on the target profile")

    coord = PhaseCoordinator(_settings(), switch_fn=never_switch, status_fn=status_fn)
    with coord.enter(Phase.RETRIEVAL):
        pass
    with coord.enter(Phase.RETRIEVAL):
        pass
    assert status_calls["n"] == 1  # lazy, cached after the first check


# -- same-phase nested reentrancy never deadlocks ----------------------------


def test_nested_same_phase_entry_does_not_deadlock_under_chat_pressure():
    """The exact scenario the module docstring calls out: a thread holding
    RETRIEVAL open, with a chat request already waiting (chat has priority
    over NEW retrieval entrants) must still be able to re-enter RETRIEVAL
    itself (nested, same thread) without blocking on its own chat-priority
    rule -- otherwise the outer hold could never complete and release, and
    the waiting chat request would starve forever."""
    coord, calls = _coordinator(initial_active="retrieval")

    nested_entered = threading.Event()
    outer_may_finish = threading.Event()
    result: dict = {}

    def outer_with_nested():
        with coord.enter(Phase.RETRIEVAL):
            # A chat request is (about to be) waiting on the other phase.
            chat_waiting = threading.Event()

            def do_chat():
                chat_waiting.set()
                with coord.enter(Phase.CHAT):
                    pass

            t_chat = threading.Thread(target=do_chat, daemon=True)
            t_chat.start()
            assert chat_waiting.wait(timeout=2.0)
            time.sleep(0.2)  # give the chat thread time to register as a waiter

            # Nested re-entry of the SAME phase, same thread: must return
            # promptly, not hang behind the chat waiter.
            with coord.enter(Phase.RETRIEVAL):
                nested_entered.set()

            outer_may_finish.wait(timeout=2.0)
            result["chat_thread"] = t_chat

    t_outer = threading.Thread(target=outer_with_nested, daemon=True)
    t_outer.start()
    assert nested_entered.wait(timeout=5.0), "nested same-phase entry deadlocked"
    outer_may_finish.set()
    t_outer.join(timeout=5.0)
    assert not t_outer.is_alive()
    chat_thread = result.get("chat_thread")
    if chat_thread is not None:
        chat_thread.join(timeout=5.0)
        assert not chat_thread.is_alive()
    assert calls == ["chat"]


# -- cross-phase nested entry raises instead of deadlocking -----------------


def test_entering_the_other_phase_while_holding_one_raises():
    """The scenario fix #1 (the review's main point) explicitly calls out:
    nesting a DIFFERENT phase on the same thread can never resolve (only
    this same, now-blocked thread could ever drop its own in-flight count),
    so it must fail loud instead of hanging."""
    coord, _ = _coordinator(initial_active="retrieval")
    with coord.enter(Phase.RETRIEVAL):
        with pytest.raises(PhaseDeadlockError):
            with coord.enter(Phase.CHAT):
                pass  # pragma: no cover - never reached

    # And the symmetric direction.
    with coord.enter(Phase.CHAT):
        with pytest.raises(PhaseDeadlockError):
            with coord.enter(Phase.RETRIEVAL):
                pass  # pragma: no cover - never reached


# -- failed switch does not leave a stale cache ------------------------------


def test_failed_switch_clears_cached_current():
    status_calls = {"n": 0}

    def status_fn(settings):
        status_calls["n"] += 1
        return "retrieval", False

    attempts = {"n": 0}

    def switch_fn(settings, target):
        attempts["n"] += 1
        raise RuntimeError("switch failed")

    coord = PhaseCoordinator(_settings(), switch_fn=switch_fn, status_fn=status_fn)

    # First entry resolves via the lazy status probe (current -> "retrieval").
    with coord.enter(Phase.RETRIEVAL):
        pass
    assert status_calls["n"] == 1

    # Entering CHAT must attempt a switch (current="retrieval" != "chat"),
    # which fails.
    with pytest.raises(RuntimeError):
        with coord.enter(Phase.CHAT):
            pass
    assert attempts["n"] == 1
    assert coord._current is None  # not left stale as "retrieval"

    # The next entrant must re-probe rather than trust a cached value that
    # might not reflect a half-finished switch.
    with coord.enter(Phase.RETRIEVAL):
        pass
    assert status_calls["n"] == 2


def test_failed_switch_notifies_waiters_so_they_dont_hang():
    """A DIFFERENT thread blocked waiting for its own turn must wake up (and
    be able to make progress) after someone else's switch attempt fails --
    notify_all() has to fire on the failure path too, not just on success."""
    status_calls = {"n": 0}

    def status_fn(settings):
        status_calls["n"] += 1
        return "retrieval", False

    def switch_fn(settings, target):
        raise RuntimeError(f"switch to {target} failed")

    coord = PhaseCoordinator(_settings(), switch_fn=switch_fn, status_fn=status_fn)

    entered_retrieval = threading.Event()
    release_retrieval = threading.Event()

    def hold_retrieval():
        with coord.enter(Phase.RETRIEVAL):
            entered_retrieval.set()
            release_retrieval.wait(timeout=5.0)

    t_hold = threading.Thread(target=hold_retrieval, daemon=True)
    t_hold.start()
    assert entered_retrieval.wait(timeout=2.0)

    chat_failed = threading.Event()
    chat_error: dict = {}

    def do_chat():
        try:
            with coord.enter(Phase.CHAT):
                pass  # pragma: no cover - switch_fn always raises first
        except RuntimeError as e:
            chat_error["e"] = e
        finally:
            chat_failed.set()

    t_chat = threading.Thread(target=do_chat, daemon=True)
    t_chat.start()
    time.sleep(0.2)  # chat registers as a waiter behind the held retrieval

    retrieval_entered = threading.Event()

    def new_retrieval():
        with coord.enter(Phase.RETRIEVAL):
            retrieval_entered.set()

    t_r2 = threading.Thread(target=new_retrieval, daemon=True)
    t_r2.start()
    time.sleep(0.2)
    assert not retrieval_entered.is_set()  # still behind chat's priority

    release_retrieval.set()
    t_hold.join(timeout=5.0)

    assert chat_failed.wait(timeout=5.0)
    assert isinstance(chat_error.get("e"), RuntimeError)
    t_chat.join(timeout=5.0)

    # The second retrieval entrant must NOT hang behind the failed switch --
    # it wakes (notify fired on the failure path too), re-probes, and
    # proceeds once the probe confirms "retrieval" is already current.
    assert retrieval_entered.wait(timeout=5.0), "waiter hung after a failed switch (missing notify)"
    t_r2.join(timeout=5.0)
    assert status_calls["n"] >= 1


# -- switch I/O never runs with the lock held --------------------------------


def test_switch_io_runs_without_holding_the_lock():
    """Fix #3: only a cooperative ``_switching`` flag (checked under brief
    lock acquisitions) gates other entrants during a switch -- the real
    ``threading.Lock`` must be free the whole time the (simulated) HTTP call
    is stalled, so every other ``enter()``/``release()`` stays cheap instead
    of queuing behind ~2 s of network I/O."""
    io_started = threading.Event()
    release_io = threading.Event()

    def slow_switch(settings, target):
        io_started.set()
        release_io.wait(timeout=5.0)

    coord = PhaseCoordinator(
        _settings(), switch_fn=slow_switch, status_fn=lambda s: ("retrieval", False)
    )
    # Seed `_current` via the lazy probe first so the next enter() goes
    # straight into a real switch attempt (not the startup-probe branch).
    with coord.enter(Phase.RETRIEVAL):
        pass

    def do_chat():
        with coord.enter(Phase.CHAT):
            pass

    t = threading.Thread(target=do_chat, daemon=True)
    t.start()
    assert io_started.wait(timeout=2.0)

    # The switch's "network call" is stalled right now (holding io_started,
    # not yet released). The coordinator's own lock must be acquirable.
    got_lock = coord._lock.acquire(blocking=False)
    if got_lock:
        coord._lock.release()

    release_io.set()
    t.join(timeout=5.0)

    assert got_lock, "the lock was held across the switch's I/O call"


def test_unknown_active_profile_switches_instead_of_reprobing_forever():
    """A probe that can't tell the active profile must lead to one switch
    attempt (which fails loudly if the engine is down), never a re-probe loop."""
    from docket.config import Settings
    from docket.phase import Phase, PhaseCoordinator

    probes, switches = [], []

    def status_fn(settings):
        probes.append(1)
        return None, None

    def switch_fn(settings, target):
        switches.append(target)
        raise RuntimeError("engine down")

    coord = PhaseCoordinator(Settings(), switch_fn=switch_fn, status_fn=status_fn)
    import pytest
    with pytest.raises(RuntimeError, match="engine down"):
        with coord.enter(Phase.CHAT):
            pass
    assert probes == [1]
    assert switches == ["chat"]
