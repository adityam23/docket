"""Consumer side of the engine's management surface (``/engine/*``).

docket only ever *consumes* the engine's HTTP API — profile definitions via
``GET /engine/profiles``, activation via ``PUT /engine/profiles/active`` (202,
completion arrives on the SSE bus at ``/engine/events``), and per-slot/device
state via ``GET /engine/status`` + ``GET /engine/device``. We never import or
edit engine code (the boundary): feature requests for the engine are written as
contract docs into ``../mettle/agent-docs/``, never code changes here.

Capability-detect once: a backend that answers ``GET /engine/profiles`` is a
mettle-class engine with a profile surface; anything else 404s it and the whole
feature is hidden — the same pattern as ``Capability`` in ``providers/base.py``.

**Supervision (§3, residency contract D1/D2):** docket's backend owns mettled's
lifecycle. The :class:`EngineSupervisor` probes ``GET /engine/ready`` — the
single readiness gate (mettle contract, 2026-10-06): 200 = ready, 503 =
alive-but-not-ready carrying a structured ``phase``/``reason``/``detail`` body,
404 = "engine too old" (never treated as ready). Liveness (is the process up at
all) is "the connection succeeded", independent of status code; the engine is
declared dead only after 3 CONSECUTIVE connection errors/timeouts (a 503/404 is
alive, not a failure — mettle's Q5 answer). Binary path unset ⇒ probe-only,
never spawn. mettled exit code 98 (§1.6) means a foreign engine holds the
port: state ``offline``, no respawn loop. Engine code is never imported.

States: ``offline`` (not answering, not supervised — or exit-98 final),
``starting`` (spawned/alive, not yet ready, within the startup ceiling),
``available`` (``/engine/ready`` == 200), ``degraded`` (alive, not ready, and
the engine itself reports a failed phase or an unrecognised body), and
``incompatible`` (alive, but 404s ``/engine/ready`` — too old to speak the
readiness contract), or ``failed`` (restart budget exhausted — manual restart
via :meth:`EngineSupervisor.restart`).

The ``engine`` surface is mounted at the server ROOT (``/engine/*``), NOT under
the OpenAI ``/v1`` base that ``backend_url`` points at — so the base is derived by
stripping the trailing ``/v1`` (see :func:`_engine_base`).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
from typing import AsyncIterator, Callable
from urllib.parse import urlparse

import httpx

from .config import Settings, load_settings
from .auth import bearer_headers


def _headers(settings: Settings) -> dict[str, str]:
    return bearer_headers(settings.engine_key)


def _engine_base(settings: Settings) -> str:
    """The engine-management base. ``backend_url`` is the OpenAI-/v1 base
    (e.g. ``http://host/v1``), but mettle mounts the ``/engine/*`` surface at the
    server ROOT of the same host — so strip the trailing ``/v1``. A URL without
    a ``/v1`` suffix is used as-is."""
    url = settings.backend_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def _get(settings: Settings, path: str, timeout: float = 5.0) -> httpx.Response | None:
    """The one synchronous GET helper for every ``/engine/*`` read — profile
    reads, status/device, and the supervisor's readiness probe all go through
    this (CLAUDE.md: one concept, one implementation)."""
    url = _engine_base(settings) + path
    try:
        return httpx.get(url, headers=_headers(settings), timeout=timeout)
    except httpx.HTTPError:
        return None


def _put(settings: Settings, path: str, json_body: dict, timeout: float = 10.0) -> httpx.Response | None:
    """The PUT counterpart to :func:`_get` (engine activation today)."""
    url = _engine_base(settings) + path
    try:
        return httpx.put(url, json=json_body, headers=_headers(settings), timeout=timeout)
    except httpx.HTTPError:
        return None


def _json(resp: httpx.Response | None, default: dict) -> dict:
    return default if resp is None else (resp.json() if resp.status_code == 200 else default)


def engine_view(settings: Settings) -> dict:
    """The SPA's window onto the engine: profile definitions + active + status +
    device, or ``{"available": False}`` when the backend has no engine surface."""
    profiles = _get(settings, "/engine/profiles")
    if profiles is None or profiles.status_code != 200:
        return {"available": False}
    out = profiles.json()
    out["available"] = True
    out["status"] = _json(_get(settings, "/engine/status"), {})
    out["device"] = _json(_get(settings, "/engine/device"), {})
    return out


def engine_activate(settings: Settings, name: str) -> tuple[int, dict]:
    """Ask the engine to switch to ``name``. Returns ``(status, body)``:
    202 accepted (completion streams on ``engine_events``), 404 unknown profile,
    409 switch already in flight, 503 busy, 0 if the engine is unreachable."""
    resp = _put(settings, "/engine/profiles/active", {"name": name})
    if resp is None:
        return 0, {"error": "engine unreachable"}
    body: dict = {}
    try:
        body = resp.json()
    except ValueError:
        pass
    return resp.status_code, body


class EngineSwitchError(RuntimeError):
    """A profile switch (:func:`switch_profile`) was rejected outright by the
    engine, or did not confirm within the bounded wait. Raised instead of
    treating the activation's bare 202 as "done" or retrying forever —
    residency contract D5 is explicit that 202 is accepted, NOT finished, and
    CLAUDE.md's backend boundary means an unreachable/misbehaving engine
    fails loudly rather than degrading silently."""


def read_status_meta(settings: Settings) -> tuple[str | None, bool | None]:
    """``(active, switching)`` from ``/engine/status`` -- the always-200
    liveness read that additionally carries which profile is active and
    whether a switch is in flight (mettle, 2026-10-07). ``(None, None)`` when
    the engine is unreachable or the body is malformed; callers treat that as
    "unknown", never as a specific profile. The ONE place this pair is parsed
    (CLAUDE.md reusability) -- :class:`EngineSupervisor` and
    :mod:`docket.phase`'s coordinator both read through here instead of each
    re-parsing ``/engine/status`` themselves."""
    resp = _get(settings, "/engine/status", timeout=5.0)
    if resp is None or resp.status_code != 200:
        return None, None
    try:
        body = resp.json()
    except ValueError:
        return None, None
    return body.get("active"), body.get("switching")


def switch_profile(settings: Settings, target: str, *, timeout_s: float | None = None) -> None:
    """Activate ``target`` and block until the engine actually serves it
    (residency contract D5), reusing :func:`_put`/:func:`_get` like every
    other engine call: ``PUT /engine/profiles/active`` (202 accepted, or 409
    if a switch to the same target is already in flight from elsewhere --
    both mean "proceed to confirmation", NOT "done"), then poll
    ``GET /engine/ready`` until 200 AND ``GET /engine/status`` confirms
    ``active == target`` and ``switching`` is false. The ``/engine/status``
    cross-check is load-bearing, not redundant: mettle's readiness gates on
    ``switching`` (2026-10-06 fix) precisely because an engine built before
    that fix could answer ``/engine/ready`` 200 for the OUTGOING profile
    mid-switch.

    Raises :class:`EngineSwitchError` if the engine rejects the switch
    outright, or if confirmation doesn't land within ``timeout_s`` (default
    ``settings.engine_switch_timeout_s``) -- fail loud, never silently retry
    forever or declare success on a timed-out switch. Callers (the phase
    coordinator) must only call this with no request of the OUTGOING phase
    in flight (D5: "never switch with a request in flight").
    """
    timeout = settings.engine_switch_timeout_s if timeout_s is None else timeout_s
    code, body = engine_activate(settings, target)
    if code not in (200, 202, 409):
        raise EngineSwitchError(
            f"engine rejected switch to profile {target!r}: HTTP {code} {body}"
        )
    deadline = time.monotonic() + timeout
    while True:
        ready = _get(settings, "/engine/ready", timeout=5.0)
        if ready is not None and ready.status_code == 200:
            active, switching = read_status_meta(settings)
            if active == target and not switching:
                return
        if time.monotonic() >= deadline:
            raise EngineSwitchError(
                f"profile switch to {target!r} did not confirm within {timeout}s "
                "(engine_switch_timeout_s)"
            )
        time.sleep(0.2)


async def engine_events_stream(settings: Settings) -> AsyncIterator[dict]:
    """Replay the engine's SSE event stream (last 100 replayed on connect) as
    parsed JSON payloads. The web layer frames each payload as one SSE event via
    ``docket.web.sse.sse()`` — hand-building the frame text here was the bug
    (f-string JSON breaks on a message containing a quote); this function raises
    instead, and the web layer frames the error the same way as every other
    event.

    Async + a bounded read timeout: mettle's ``/engine/events`` keepalive is a
    contractual 15 s (Q6), so a 30 s read timeout is a >1 keepalive-interval
    margin. If the engine dies or the client walks away, the underlying
    ``httpx`` read raises within that bound and this generator unwinds —
    releasing the proxying task instead of blocking it indefinitely.
    """
    url = _engine_base(settings) + "/engine/events"
    timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url, headers=_headers(settings)) as r:
            if r.status_code != 200:
                raise RuntimeError(f"engine returned {r.status_code} for /engine/events")
            async for line in r.aiter_lines():
                # Drop `event:` lines so every payload is treated as a default
                # message — the SPA's EventSource.onmessage then catches ALL
                # events (named events would need addEventListener per name).
                # The JSON `data` payload already carries the event type.
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload:
                    continue
                try:
                    yield json.loads(payload)
                except ValueError:
                    continue


def resolve_model(settings: Settings, model_id: str) -> str:
    """Resolve a ``role:<kind>`` alias to the active profile's slot id so a
    profile switch automatically re-points chat/embed/rerank. Anything that is
    NOT a ``role:`` alias passes through unchanged, and when the engine surface
    is absent (or unreachable) the alias is passed through too (the backend
    errors, surfaced properly) — no silent substitution."""
    if not model_id.startswith("role:"):
        return model_id
    role = model_id[len("role:"):]
    data = _get(settings, "/engine/profiles")
    if data is None or data.status_code != 200:
        return model_id
    body = data.json()
    active = body.get("active")
    slots: list[dict] = []
    for p in body.get("profiles", []):
        if p.get("name") == active:
            slots = p.get("slots", [])
            break
    role_slots = [s for s in slots if s.get("role") == role]
    if not role_slots:
        return model_id
    # Highest priority first (mettle orders by priority, then declaration order);
    # slots without a priority rank below all priority-bearing ones.
    role_slots.sort(key=lambda s: (s.get("priority") is None, -(s.get("priority") or 0)))
    return role_slots[0].get("id") or model_id


def _engine_host_port(settings: Settings) -> tuple[str, int | None]:
    u = urlparse(_engine_base(settings))
    return u.hostname or "127.0.0.1", u.port


class EngineSupervisor:
    """Spawn + health-probe + crash-restart mettled (cutover contract §3,
    residency contract D1/D2).

    States: see module docstring. Binary path unset ⇒ probe-only: states move
    between ``offline``/``starting``/``available``/``degraded``/``incompatible``,
    never spawn.
    """

    _BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 16.0)  # restart budget: 5 attempts

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        settings_fn: Callable[[], Settings] | None = None,
    ) -> None:
        # ``settings_fn``, when given, is re-invoked once per supervision cycle
        # (every probe) so a Settings-page change to engine_binary/backend_url/
        # mettle_config/engine_key takes effect without a restart — the operator
        # decision behind live-reload. Tests/benchmarks that construct a fixed
        # ``Settings()`` directly (no ``settings_fn``) keep static behaviour, so
        # a custom backend_url/engine_binary they pass is never silently
        # overridden by the real environment.
        self._settings_fn = settings_fn
        self._settings = settings if settings is not None else (
            settings_fn() if settings_fn else load_settings()
        )
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()   # nudges the loop: restart() or stop
        self._stop = False
        self.state = "offline"
        self.attempts = 0
        self.pid: int | None = None
        self._proc: subprocess.Popen | None = None
        # Readiness detail surfaced via view() for the Settings/engine panel.
        self.phase: dict = {}
        self.reason: str | None = None
        self.detail = None
        # `/engine/status` additionally carries `active` (profile name) and
        # `switching` (bool) (mettle, 2026-10-07) -- read-only here. Profile
        # switching itself is a follow-up ticket; this supervisor only
        # surfaces which phase the engine is in.
        self.active: str | None = None
        self.switching: bool | None = None
        # §1.6/D1: a foreign engine holds the port (mettled exit 98). Terminal
        # like ``failed``, but reported as ``offline`` — :meth:`restart` is the
        # only exit, same as ``failed``.
        self._port_held = False

    # -- settings -------------------------------------------------------------

    def _refresh_settings(self) -> None:
        if self._settings_fn is not None:
            self._settings = self._settings_fn()

    # -- probe (readiness gate, D2) ------------------------------------------

    def _probe(self) -> dict:
        """``GET /engine/ready``: 2 s timeout (mettle's Q5 recommendation).
        Refreshes settings first, so this is the one place a changed
        backend_url/engine_key takes effect. Returns ``{"alive", "code",
        "body", "active", "switching"}`` — ``alive`` is True for ANY HTTP
        response (connection succeeded), independent of status code; only a
        connection error or timeout is a liveness failure.

        ``active``/``switching`` come from a best-effort follow-up read of
        ``/engine/status`` (mettle, 2026-10-07: that endpoint additionally
        carries the active profile name and whether a switch is in flight).
        docket does not implement profile switching here (a follow-up
        ticket); this only surfaces what mettle already reports."""
        self._refresh_settings()
        resp = _get(self._settings, "/engine/ready", timeout=2.0)
        if resp is None:
            return {"alive": False, "code": None, "body": {}, "active": None, "switching": None}
        try:
            body = resp.json()
        except ValueError:
            body = {}
        active, switching = self._read_status_meta()
        return {
            "alive": True, "code": resp.status_code, "body": body,
            "active": active, "switching": switching,
        }

    def _read_status_meta(self) -> tuple[str | None, bool | None]:
        """Best-effort ``active``/``switching`` from ``/engine/status``
        (liveness-only, always 200 while the process is up). A failure here
        never affects liveness/readiness — ``/engine/ready`` already decided
        that — it just leaves ``active``/``switching`` unknown for this tick.
        Delegates to the module-level :func:`read_status_meta` (CLAUDE.md:
        one implementation, shared with the phase coordinator)."""
        return read_status_meta(self._settings)

    @staticmethod
    def _state_for_probe(probe: dict) -> tuple[str, dict]:
        """Map one ``/engine/ready`` probe to ``(state, {"phase","reason",
        "detail"})`` per the residency contract: 200 ready, 503 alive-but-not-
        ready (structured body), 404 "too old" (never ready)."""
        code, body = probe["code"], probe["body"]
        if code == 200:
            return "available", {"phase": body.get("phase", {})}
        if code == 404:
            return "incompatible", {"reason": "engine too old: no /engine/ready"}
        if code == 503:
            phase = body.get("phase") or {}
            state = "degraded" if phase.get("phase") == "failed" else "starting"
            return state, {
                "phase": phase,
                "reason": body.get("reason"),
                "detail": body.get("detail"),
            }
        return "degraded", {"detail": body}

    def _report_alive(self, probe: dict, *, reset_attempts: bool) -> None:
        state, extra = self._state_for_probe(probe)
        with self._lock:
            self.state = state
            if reset_attempts:
                self.attempts = 0
            self.phase = extra.get("phase", {})
            self.reason = extra.get("reason")
            self.detail = extra.get("detail")
            self.active = probe.get("active")
            self.switching = probe.get("switching")

    # -- spawn --------------------------------------------------------------

    def _spawn(self) -> None:
        """Start mettled. The port/config come from mettle's own config file
        (docket never pushes engine config, §4); METTLE_CONFIG is honoured when
        the operator set one. Never imports engine code."""
        binary = self._settings.engine_binary
        env = dict(os.environ)
        if self._settings.mettle_config:
            env["METTLE_CONFIG"] = self._settings.mettle_config
        proc = subprocess.Popen(
            [binary], env=env, cwd=os.path.dirname(binary) or None,
            start_new_session=True,  # own process group: stop() can kill the whole tree
        )
        with self._lock:
            self.pid = proc.pid
            self._proc = proc

    def _wait_for_port_free(self, timeout: float = 2.0) -> None:
        """Bounded wait for the OS to release the port after we just killed our
        own child (D1), before spawning the replacement — avoids a pointless
        self-inflicted EADDRINUSE race. Not fatal if it never frees inside the
        window: the spawn is attempted anyway, and a still-held port surfaces
        as mettled's own exit 98 (§1.6) or a startup-ceiling failure, both
        already handled."""
        host, port = _engine_host_port(self._settings)
        if port is None:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.2):
                    pass
            except OSError:
                return  # nothing listening -- port is free
            time.sleep(0.1)

    # -- lifecycle ----------------------------------------------------------

    def ensure(self) -> None:
        """Idempotent: start the supervision loop (web-app startup, CLI entry)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = False
            self._thread = threading.Thread(
                target=self._loop, name="engine-supervisor", daemon=True
            )
            self._thread.start()

    def restart(self) -> None:
        """Manual restart — the only exit from the supervisor's terminal
        ``failed`` state AND from the ``_port_held`` (exit-98) terminal state.
        Resets the budget under the lock and wakes the loop; the next
        probe/spawn cycle decides the state. (Bug fix: previously only reset
        ``attempts`` outside the lock and never set ``_wake``, so the ``failed``
        loop — which waits only on ``_wake``/``_stop`` — never saw this call and
        POST /api/engine/restart could not leave ``failed``.)"""
        with self._lock:
            self.attempts = 0
            self._port_held = False
        self._wake.set()

    def _kill_group(self, proc: subprocess.Popen) -> None:
        """SIGTERM the spawned engine's process group, then SIGKILL whatever
        survived and REAP it. Kept to one implementation because both the
        shutdown path and the startup-ceiling path need identical semantics.
        The TERM wait (3 s) is sized so TERM+KILL completes inside uvicorn's
        graceful-shutdown window — a KILL that never fires is the exact
        orphan we are preventing (mettled holds VRAM)."""
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=3.0)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5.0)  # reap: no zombie holding the port
        except subprocess.TimeoutExpired:
            pass

    def _reap_current(self) -> subprocess.Popen | None:
        """Take ``self._proc``, clear our tracking, and :meth:`_kill_group` it
        if still alive. Returns the reaped ``Popen`` (its ``.returncode`` is
        set either by our own kill or by its own prior exit) so callers can
        inspect the exit code — e.g. mettled's contractual 98 (§1.6, port held
        by a foreign engine) — or ``None`` if there was nothing to reap. One
        implementation for every "take + clear + kill" site: ``stop()``, the
        stop()-race guard in ``_bring_up``, the startup-ceiling kill, and the
        D1 never-spawn-over-a-live-child guard."""
        proc = self._proc
        self._proc = None
        self.pid = None
        if proc is not None:
            self._kill_group(proc)
        return proc

    def stop(self) -> None:
        """Stop supervision AND the spawned engine (see :meth:`_kill_group`)."""
        self._stop = True
        self._wake.set()
        self._reap_current()

    def view(self) -> dict:
        """Supervision state for the SPA: honest ``offline/starting/available/
        degraded/incompatible/failed``, plus the readiness ``phase``/``reason``/
        ``detail`` the engine reported (residency contract D2) so the UI can
        render WHY, not just the state label. ``active``/``switching`` are
        read-only passthroughs of mettle's ``/engine/status`` fields (2026-10-07)
        — which profile is active and whether a switch is in flight; this
        supervisor does not initiate switches itself (a follow-up ticket)."""
        with self._lock:
            return {
                "state": self.state,
                "pid": self.pid,
                "binary": self._settings.engine_binary,
                "attempts": self.attempts,
                "phase": self.phase,
                "reason": self.reason,
                "detail": self.detail,
                "active": self.active,
                "switching": self.switching,
            }

    # -- the loop -----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop:
            if self._port_held:
                # §1.6/D1 terminal: a foreign engine holds the port. No probe,
                # no spawn, no respawn loop -- another attempt would just
                # collide again. :meth:`restart` is the only exit.
                while not self._stop and not self._wake.is_set():
                    self._wake.wait(timeout=5.0)
                self._wake.clear()
                continue
            probe = self._probe()
            if probe["alive"]:
                self._report_alive(probe, reset_attempts=True)
                self._wait_for_death()
            elif not self._settings.engine_binary:
                # Probe-only: nothing to spawn or fix. An engine that stops
                # answering goes (and stays) ``offline`` — the loud, honest
                # state — retried on the cheap probe interval.
                with self._lock:
                    self.state = "offline"
                    self.attempts = 0
                    self.phase, self.reason, self.detail = {}, None, None
                    self.active, self.switching = None, None
                self._wake.wait(timeout=2.0)
                self._wake.clear()
            else:
                self._bring_up()

    def _wait_for_death(self) -> None:
        """Stay alive-reporting until the probe fails 3 CONSECUTIVE times
        (mettle's Q5 recommendation: 2 s timeout, 2 s interval, dead after 3
        consecutive connection errors/timeouts — a 503/404 is alive, just not
        ready, and is NOT a failure), or the operator nudges us (restart/stop).
        """
        fails = 0
        while not self._stop and not self._wake.is_set():
            probe = self._probe()
            if probe["alive"]:
                fails = 0
                self._report_alive(probe, reset_attempts=True)
            else:
                fails += 1
                if fails >= 3:
                    return
            self._wake.wait(timeout=2.0)
        self._wake.clear()

    def _bring_up(self) -> None:
        """Get the engine answering again: spawn + probe until ready, with a
        bounded restart budget on genuine crashes. ``failed`` is terminal for
        the loop until :meth:`restart` resets it.

        D1 fix: the 3-consecutive-probe-failure death declaration does NOT
        prove the previous child died — it may be alive but hung/unresponsive.
        Before ever spawning, we kill+reap whatever ``self._proc`` still is
        (a no-op if it already exited on its own) and give the port a bounded
        window to free, so a respawn can never overlap with a live, now-
        untracked original (the exact orphan this supervisor exists to
        prevent).

        While the spawned child is ALIVE and merely not ready yet, we keep
        waiting without consuming the budget — a first run legitimately spends
        minutes downloading GB-scale weights, and respawning over it would
        double-book VRAM. An attempt is counted only when the child DIES: it
        is reaped, we back off, and the next loop pass spawns a fresh one.
        mettled exit code 98 (§1.6, port held by a foreign engine) is final:
        ``offline``, no respawn loop — another attempt would just collide
        again.

        Deadlock fix: the ``failed`` wait loop must NOT hold ``self._lock``
        while it blocks on ``self._wake`` — :meth:`restart` (and :meth:`view`)
        need that same lock for a brief, ordinary read/write, and a lock held
        across an indefinite wait starves both forever. This was the deeper
        half of the restart()-can't-leave-failed bug: even with ``restart()``
        fixed to set ``_wake``, a lock held here would make ``restart()``'s own
        ``with self._lock:`` block before it ever reaches ``_wake.set()``."""
        self._refresh_settings()
        with self._lock:
            exhausted = self.attempts >= len(self._BACKOFF_S)
            self.state = "failed" if exhausted else "starting"
            if exhausted:
                # Cleared under the same lock section as the state flip (not
                # held across the wait below) so a restart() landing right
                # around this transition is never silently swallowed.
                self._wake.clear()
        if exhausted:
            # Sleep until nudged; the loop must not spin in failed state, and
            # must not hold the lock while it waits (see docstring).
            while not self._stop and not self._wake.is_set():
                self._wake.wait(timeout=5.0)
            return

        died = self._reap_current()
        if died is not None and died.returncode == 98:
            self._offline_port_held()
            return
        self._wait_for_port_free()

        self.attempts += 1
        self._spawn()
        # stop() may have landed between the budget check and the spawn: kill
        # the just-spawned engine NOW, or the loop below would exit (its
        # condition is already false) leaving a fresh mettled orphaned — the
        # exact VRAM-holding orphan this supervisor exists to prevent.
        if self._stop:
            self._reap_current()
            return

        # Probe until ready; stay patient while the child lives and keeps
        # starting up (weights download / model load), bail quickly on death.
        # Ceiling: a child that outlives engine_startup_window_s without ever
        # reaching 200 is hung — kill the group (reaps), count the attempt,
        # respawn on the next pass. The ceiling is long so a legitimate
        # first-run download is never misread as a hang.
        started = time.monotonic()
        died_code: int | None = None
        while not self._stop:
            probe = self._probe()
            if probe["alive"]:
                # Only a 503 means "still preparing"; 200 (ready) and any other
                # answer (404 = too old, a permanent verdict) are settled — the
                # loop's alive-watch takes over, never a 900 s wait + respawn.
                settled = probe["code"] != 503
                self._report_alive(probe, reset_attempts=settled)
                if settled:
                    return
            proc = self._proc
            if proc is None or proc.poll() is not None:
                died_code = proc.poll() if proc is not None else None
                self._proc = None
                self.pid = None
                break  # child died before answering -> genuine attempt failure
            if time.monotonic() - started > self._settings.engine_startup_window_s:
                reaped = self._reap_current()
                died_code = reaped.returncode if reaped is not None else None
                break
            time.sleep(0.5)
            if self._stop:
                return

        if died_code == 98:
            self._offline_port_held()
            return

        # Child dead (crash) or killed for hanging: back off, retry next pass.
        self._wake.wait(timeout=self._BACKOFF_S[min(self.attempts, len(self._BACKOFF_S)) - 1])
        self._wake.clear()

    def _offline_port_held(self) -> None:
        """mettled exit 98 (§1.6): a foreign engine holds the port. Sets the
        ``_port_held`` flag, which :meth:`_loop` treats as terminal — ``offline``,
        no respawn loop; the operator must free the port, point docket
        elsewhere, or call :meth:`restart` once they have."""
        with self._lock:
            self.state = "offline"
            self.reason = "port held by a foreign engine (mettled exit 98)"
            self.phase, self.detail = {}, None
            self.active, self.switching = None, None
            self._port_held = True


_SUPERVISOR: EngineSupervisor | None = None
_SUPERVISOR_LOCK = threading.Lock()


def get_supervisor(settings: Settings | None = None) -> EngineSupervisor:
    """The process-wide supervisor (one engine, one supervisor — CLAUDE.md).
    Constructed once with ``settings_fn=load_settings`` so every later
    supervision cycle re-reads settings — a Settings-page change takes effect
    on the next probe/spawn without a restart. ``settings`` seeds the FIRST
    construction only (matching the previous contract); later calls with a
    different ``settings`` no longer silently do nothing — the supervisor
    already live-reloads on its own."""
    global _SUPERVISOR
    with _SUPERVISOR_LOCK:
        if _SUPERVISOR is None:
            _SUPERVISOR = EngineSupervisor(settings or load_settings(), settings_fn=load_settings)
        return _SUPERVISOR
