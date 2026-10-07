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
  phase inside an already-open one is not a pattern any caller needs and is
  not guaranteed deadlock-free.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from enum import Enum
from typing import Callable

from .config import Settings, load_settings
from .engine import read_status_meta, switch_profile


class Phase(str, Enum):
    RETRIEVAL = "retrieval"
    CHAT = "chat"


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

        target = self._target(phase)
        is_chat = phase is Phase.CHAT
        with self._cond:
            if is_chat:
                self._chat_waiters += 1
            try:
                while True:
                    if self._current is None:
                        # Lazy startup read: don't switch needlessly if the
                        # engine already happens to be on the right profile.
                        self._current = self._status_fn(self._settings)[0]
                    if self._current == target:
                        if not is_chat and self._chat_waiters > 0:
                            # A chat request is waiting for its zero-in-flight
                            # window to switch in -- a fresh retrieval
                            # entrant must not keep reopening that window.
                            self._cond.wait()
                            continue
                        break
                    if self._in_flight == 0:
                        # Nobody is mid-request on the outgoing phase: safe
                        # to switch (D5: never switch with a request in
                        # flight). Raises loudly (EngineSwitchError) on
                        # rejection/timeout -- propagates out of enter().
                        self._switch_fn(self._settings, target)
                        self._current = target
                        self._cond.notify_all()
                        continue
                    self._cond.wait()
            finally:
                if is_chat:
                    self._chat_waiters -= 1
                    self._cond.notify_all()
            self._in_flight += 1
        depth[phase] = 1

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


@contextmanager
def engine_phase(phase: Phase):
    """Enter ``phase`` on the process-wide coordinator. The convenience call
    for any explicit caller; ``providers/router.py``'s per-call wrapping is
    the one place that calls it for every engine request."""
    with get_phase_coordinator().enter(phase):
        yield
