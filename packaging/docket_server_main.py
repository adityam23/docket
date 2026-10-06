"""Frozen server entry point — what PyInstaller packages as ``docket-server``,
the Tauri desktop shell's sidecar. The dev path stays `dk serve`; this entry
exists only for the desktop bundle.

Contract with the shell (METTLE_ONLY_CUTOVER_CONTRACT.md §8):
- binds 127.0.0.1 on the port passed as ``--port`` (the shell picks a free
  port — never hardcode 8760, a dev `dk serve` may hold it);
- env carries the resource paths: ``DK_ENGINE_BINARY`` (bundled mettled) and
  optionally ``DK_METTLE_CONFIG``;
- the shell sends SIGTERM on quit; this stops mettled's process group FIRST
  (see ``_EngineFirstServer`` below), then uvicorn runs its own graceful
  shutdown (connection drain + the FastAPI shutdown hook).

Ordering note (why "first" matters): uvicorn's ``Server.shutdown()`` waits
for open connections — including the long-lived SSE streams
``/api/engine/events`` and ``/api/ask/stream`` — to close BEFORE it runs the
lifespan ``shutdown`` event (``docket.web.app._stop_engine_supervisor``). With
either stream open, that wait can consume the whole ``timeout_graceful_shutdown``
window before the engine is ever stopped — past the shell's SIGKILL deadline,
orphaning mettled (verified by reading the installed uvicorn's
``uvicorn/server.py``: ``shutdown()`` → ``_wait_tasks_to_complete()`` → *then*
``self.lifespan.shutdown()``). ``_EngineFirstServer`` stops the engine
synchronously on the first captured signal, before any of that drain even
starts counting, so the kill no longer depends on connections being idle.

Exit-code note: uvicorn's ``capture_signals`` re-raises a captured SIGTERM
after the fully graceful shutdown (POSIX killed-by-signal convention), so the
observed code is 143 in every TERM path — direct or frozen (verified). The
shell treats any exit on quit as success and falls back to SIGKILL after its
own grace period (desktop/src-tauri/src/main.rs ``GRACE_PERIOD``); nothing
consumes the code.

The app object is passed directly (not by import string) so PyInstaller's
static analysis collects the whole docket package.
"""

from __future__ import annotations

import argparse
from types import FrameType

import uvicorn

from docket.web.app import app as web_app
from docket.web.app import stop_engine_if_ephemeral


class _EngineFirstServer(uvicorn.Server):
    """``uvicorn.Server`` that stops the engine BEFORE the graceful drain.

    ``Server.handle_exit`` is installed as the SIGTERM/SIGINT handler by
    ``Server.capture_signals()`` and runs synchronously on the first signal
    the process receives. Overriding it to call the shared, gated
    ``stop_engine_if_ephemeral()`` first — and only on the first signal, via
    the same ``_captured_signals`` list the base class tracks — guarantees
    mettled's process group is already gone before uvicorn's own shutdown
    sequence (which waits on open connections) ever runs. Idempotent with the
    FastAPI shutdown hook in ``docket.web.app``, which still runs afterwards
    as the dev-path behaviour / safety net (``EngineSupervisor.stop()`` is a
    no-op once already stopped — see that module's docstring for the shared
    gating logic).
    """

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        if not self._captured_signals:
            stop_engine_if_ephemeral()
        super().handle_exit(sig, frame)


def main() -> None:
    ap = argparse.ArgumentParser(description="docket desktop server (sidecar)")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    # Mirrors uvicorn.run()'s single-process, no-reload path (Config + Server
    # + .run()) so swapping in _EngineFirstServer is the only difference.
    config = uvicorn.Config(
        web_app, host=args.host, port=args.port, reload=False,
        log_level="info",
        # Engine safety no longer depends on this window (the engine is
        # already stopped before this drain starts — see module docstring),
        # so it's kept short: just enough for an in-flight response to finish,
        # not a budget the shell needs to wait out.
        timeout_graceful_shutdown=5,
    )
    config.load_app()
    server = _EngineFirstServer(config=config)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    if not server.started:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
