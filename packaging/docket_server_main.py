"""Frozen server entry point — what PyInstaller packages as ``docket-server``,
the Tauri desktop shell's sidecar. The dev path stays `dk serve`; this entry
exists only for the desktop bundle.

Contract with the shell (METTLE_ONLY_CUTOVER_CONTRACT.md §8):
- binds 127.0.0.1 on the port passed as ``--port`` (the shell picks a free
  port — never hardcode 8760, a dev `dk serve` may hold it);
- env carries the resource paths: ``DK_ENGINE_BINARY`` (bundled mettled) and
  optionally ``DK_METTLE_CONFIG``;
- the shell sends SIGTERM on quit; uvicorn's graceful path runs the FastAPI
  shutdown hook (which stops the engine supervisor, killing mettled's process
  group — proven by the baked-in smoke in packaging/build-sidecar.sh).

Exit-code note: uvicorn's ``capture_signals`` re-raises a captured SIGTERM
after the fully graceful shutdown (POSIX killed-by-signal convention), so the
observed code is 143 in every TERM path — direct or frozen (verified). The
shell treats any exit on quit as success and falls back to SIGKILL after
10 s; nothing consumes the code.

The app object is passed directly (not by import string) so PyInstaller's
static analysis collects the whole docket package.
"""

from __future__ import annotations

import argparse

import uvicorn

from docket.web.app import app as web_app


def main() -> None:
    ap = argparse.ArgumentParser(description="docket desktop server (sidecar)")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    uvicorn.run(web_app, host=args.host, port=args.port, reload=False,
                log_level="info",
                # The FastAPI shutdown hook stops the engine supervisor
                # (TERM 3 s + KILL + reap). Give uvicorn a window comfortably
                # larger than that so the KILL always gets its turn.
                timeout_graceful_shutdown=20)


if __name__ == "__main__":
    main()