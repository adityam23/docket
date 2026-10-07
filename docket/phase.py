"""ONE phase coordinator for mettle's two-profile engine (residency contract
D5/D6): the reference 6 GB config can't keep embed+rerank and the chat
generator resident together, so it ships two profiles -- ``retrieval``
(embed + rerank) and ``chat`` (the generator) -- and docket switches between
them per phase. Every engine call enters the phase it needs through this
single process-wide coordinator (see ``providers/router.py``'s per-call
wrapping) -- never a second, ad hoc switching path (CLAUDE.md reusability).

Semantics (operator-decided, 2026-10-06/07):
- Requests of the CURRENT phase run concurrently (an in-flight counter; no
  switch is ever attempted while any request is in flight -- D5: "never
  switch with a request in flight").
- A request for the OTHER phase waits for in-flight to drop to zero, then
  switches (:func:`docket.engine.switch_profile`: PUT + poll ready + verify
  active/!switching, bounded by ``Settings.engine_switch_timeout_s`` ->
  :class:`docket.engine.EngineSwitchError` on timeout -- fail loud, never a
  silent indefinite wait).
- CHAT waiters have priority: while any chat request is waiting to switch
  in, new RETRIEVAL entrants block. This is what lets an ``ask`` preempt a
  running ingest at the next safe point (the next per-chunk engine call)
  instead of queuing behind the whole batch.
- Empty profile names (``Settings.engine_retrieval_profile`` /
  ``engine_chat_profile``) disable switching entirely (a single-profile
  engine): every :meth:`PhaseCoordinator.enter` is then a no-op.
- "Already in the right profile at startup" is read once, lazily, from
  ``/engine/status`` -- never an unconditional switch on the very first call.
- Nested entry of the SAME phase by the SAME thread (an outer caller already
  holding e.g. RETRIEVAL open, calling into code that itself enters
  RETRIEVAL again) never deadlocks: a thread's own nested re-entry is
  accounted for but bypasses the blocking/chat-priority checks below, which
  only make sense for a thread's OUTERMOST entry -- otherwise a chat waiter
  could permanently block a retrieval call that only the outer, still-open
  retrieval context can ever complete (self-deadlock). Nesting a DIFFERENT
  phase inside an already-open one on the SAME thread is not supportable --
  this thread's own hold is the only thing that could ever release the
  in-flight count the new entry would wait on, so it raises
  :class:`PhaseDeadlockError` immediately instead of hanging (see
  :meth:`PhaseCoordinator._acquire`).
- A switch (or the lazy startup status read) runs its HTTP call OUTSIDE the
  condition lock -- only a ``_switching`` flag is held across it. Every other
  ``enter()``/``release()`` is then a cheap, brief, lock-only operation; they
  never queue up behind a ~2 s network round trip. Other entrants simply wait
  while ``_switching`` is set, same as they'd wait on in-flight count.
- If a switch (or the status read) raises, the cached ``_current`` profile is
  cleared (not left stale) before the exception propagates, so the next
  entrant re-reads ``/engine/status`` instead of trusting a profile the
  engine may have only half-adopted.

Holding ONE phase open across a logical group of calls (e.g.
``Retriever.retrieve``'s search+rerank pair, or the agent's per-hop
chat-answer+refine pair) uses the same same-thread re-entrancy: the outer
caller enters the phase once via :func:`provider_phase`, and every individual
engine call underneath -- wrapped per-call by
``providers/router.py``'s ``PhasedProvider`` -- nests into that same open
phase for free, instead of releasing and re-acquiring (and risking a
competing thread's phase winning the gap) between each call.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from enum import Enum
from typing import Callable

from .config import Settings, load_settings
from .engine import read_status_meta, switch_profile


class Phase(str, Enum):
    RETRIEVAL = "retrieval"
    CHAT = "chat"


class PhaseDeadlockError(RuntimeError):
    """Raised instead of deadlocking: a thread already holding one phase open
    tried to enter the OTHER phase on the same thread. Only that thread could
    ever drop its own in-flight count to let the switch happen, so the wait
    could never resolve -- this is a caller bug (nest the SAME phase, or
    release the held phase first), not a contention case to wait out."""


# Injectable so tests exercise the coordinator's own concurrency/priority
# logic without a real engine or network (CLAUDE.md: tests stay offline &
# deterministic) -- mirrors EngineSupervisor's settings_fn injection.
SwitchFn = Callable[[Settings, str], None]
StatusFn = Callable[[Settings], "tuple[str | None, bool | None]"]


class PhaseCoordinator:
    """Process-wide (per ``Settings``) phase gate. See module docstring for
    the full semantics."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        settings_fn: Callable[[], Settings] | None = None,
        switch_fn: SwitchFn = switch_profile,
        status_fn: StatusFn = read_status_meta,
    ) -> None:
        # ``settings_fn``, when given, is re-invoked on every ``enter()`` so a
        # Settings-page change to the profile names/timeout takes effect
        # without a restart (mirrors EngineSupervisor's live-reload).
        self._settings_fn = settings_fn
        self._settings = settings if settings is not None else (
            settings_fn() if settings_fn else load_settings()
        )
        self._switch_fn = switch_fn
        self._status_fn = status_fn
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        # The engine's active profile NAME, as last observed/switched-to.
        # None = not yet checked (triggers the one lazy /engine/status read).
        self._current: str | None = None
        self._in_flight = 0
        self._chat_waiters = 0
        # True while a status probe or a real switch's HTTP call is running
        # OUTSIDE the lock (fix: never hold the lock across network I/O).
        # Every other enter() waits on this exactly like it waits on
        # in-flight count -- it's a short-lived mutual-exclusion flag, not a
        # second source of truth about the engine.
        self._switching = False
        # Per-thread nesting depth, keyed by Phase -- lets a thread already
        # holding a phase open re-enter the SAME phase without re-running the
        # blocking/priority checks (which would wait on a condition only that
        # same, now-blocked thread could ever resolve).
        self._tls = threading.local()

    def _refresh_settings(self) -> None:
        if self._settings_fn is not None:
            self._settings = self._settings_fn()

    def _enabled(self) -> bool:
        return bool(self._settings.engine_retrieval_profile) and bool(
            self._settings.engine_chat_profile
        )

    def _target(self, phase: Phase) -> str:
        return (
            self._settings.engine_retrieval_profile
            if phase is Phase.RETRIEVAL
            else self._settings.engine_chat_profile
        )

    def _depth(self) -> dict:
        d = getattr(self._tls, "depth", None)
        if d is None:
            d = {}
            self._tls.depth = d
        return d

    @contextmanager
    def enter(self, phase: Phase):
        """Block until the engine is (or becomes) ``phase``, run the wrapped
        call, then release. A no-op when phase switching is disabled (either
        profile name is empty)."""
        self._refresh_settings()
        if not self._enabled():
            yield
            return
        self._acquire(phase)
        try:
            yield
        finally:
            self._release(phase)

    # -- internals ------------------------------------------------------

    def _acquire(self, phase: Phase) -> None:
        depth = self._depth()
        if depth.get(phase, 0) > 0:
            # Reentrant: this thread already holds `phase` open (see the
            # module docstring's nesting note). Just account for it.
            with self._cond:
                self._in_flight += 1
            depth[phase] = depth[phase] + 1
            return

        held_other = next((p for p, d in depth.items() if d > 0 and p != phase), None)
        if held_other is not None:
            raise PhaseDeadlockError(
                f"this thread already holds phase {held_other.value!r} open -- "
                f"entering {phase.value!r} on the same thread would wait on "
                "its own in-flight count and deadlock forever. Release the "
                "held phase before entering the other one (or nest the SAME "
                "phase, which is fine)."
            )

        target = self._target(phase)
        is_chat = phase is Phase.CHAT
        with self._cond:
            if is_chat:
                self._chat_waiters += 1
        try:
            while True:
                probe = False
                with self._cond:
                    if self._switching:
                        # Someone else's status probe or real switch is
                        # mid-flight (I/O happening OUTSIDE this lock right
                        # now) -- wait for it like any other state change.
                        self._cond.wait()
                        continue
                    if self._current is None:
                        # Lazy startup read: don't switch needlessly if the
                        # engine already happens to be on the right profile.
                        # Flagged + done below, outside the lock (fix: a
                        # ~2s-ish HTTP call must never block every other
                        # enter()/release() in the process).
                        self._switching = True
                        probe = True
                    elif self._current == target:
                        if not is_chat and self._chat_waiters > 0:
                            # A chat request is waiting for its zero-in-flight
                            # window to switch in -- a fresh retrieval
                            # entrant must not keep reopening that window.
                            self._cond.wait()
                            continue
                        self._in_flight += 1
                        depth[phase] = 1
                        return
                    elif self._in_flight == 0:
                        # Nobody is mid-request on the outgoing phase: safe
                        # to switch (D5: never switch with a request in
                        # flight).
                        self._switching = True
                    else:
                        self._cond.wait()
                        continue
                # Lock released here: the actual HTTP call (status probe or
                # the real profile switch) runs WITHOUT holding it.
                if probe:
                    try:
                        active = self._status_fn(self._settings)[0]
                    except Exception:  # noqa: BLE001 - a probe failure is "unknown", not fatal
                        active = None
                    with self._cond:
                        # Unknown (probe failed / no `active`) must NOT go back
                        # to None: that would re-probe in a hot loop against a
                        # down engine. "" matches no profile, so the next pass
                        # switches -- which fails loudly if the engine is down.
                        self._current = active or ""
                        self._switching = False
                        self._cond.notify_all()
                    continue
                try:
                    self._switch_fn(self._settings, target)
                except Exception:
                    # Fail loud (propagates to the caller below) AND drop the
                    # cached profile -- the engine may be half-switched, so
                    # the next entrant must re-read /engine/status rather
                    # than trust a profile we're no longer sure is active.
                    with self._cond:
                        self._current = None
                        self._switching = False
                        self._cond.notify_all()
                    raise
                with self._cond:
                    self._current = target
                    self._switching = False
                    self._cond.notify_all()
        finally:
            if is_chat:
                with self._cond:
                    self._chat_waiters -= 1
                    self._cond.notify_all()

    def _release(self, phase: Phase) -> None:
        depth = self._depth()
        depth[phase] = depth.get(phase, 1) - 1
        with self._cond:
            self._in_flight -= 1
            self._cond.notify_all()


_COORDINATOR: PhaseCoordinator | None = None
_COORDINATOR_LOCK = threading.Lock()


def get_phase_coordinator(settings: Settings | None = None) -> PhaseCoordinator:
    """The process-wide coordinator (one engine, one coordinator -- mirrors
    ``engine.get_supervisor``). ``settings`` seeds the FIRST construction
    only; later calls live-reload via ``load_settings`` like the supervisor
    already does."""
    global _COORDINATOR
    with _COORDINATOR_LOCK:
        if _COORDINATOR is None:
            _COORDINATOR = PhaseCoordinator(settings or load_settings(), settings_fn=load_settings)
        return _COORDINATOR


def provider_phase(provider: object, phase: Phase):
    """Enter ``phase`` via ``provider.phase(phase)`` -- the context manager
    ``providers/router.py``'s ``PhasedProvider`` exposes for holding ONE
    phase open across several individual engine calls (CLAUDE.md
    reusability: the ONE call site every such caller uses, instead of each
    duplicating the ``getattr`` fallback below).

    Falls back to a no-op (:func:`contextlib.nullcontext`) when ``provider``
    doesn't implement ``.phase`` -- every test double in ``tests/fakes.py``
    models the engine's chat/collection surface, not its phase surface, so
    this keeps ``Retriever``/``agent.graph.answer`` callable with a bare fake
    and zero coordinator/network involvement, with no change needed in
    ``tests/conftest.py`` or any existing test (CLAUDE.md: tests stay
    offline & deterministic by construction, not by remembering to disable
    a global switch)."""
    fn = getattr(provider, "phase", None)
    if fn is None:
        return nullcontext()
    return fn(phase)
