"""Frozen server entry point — what PyInstaller packages as ``docket-server``,
the Tauri desktop shell's sidecar. The dev path stays `dk serve`; this entry
exists only for the desktop bundle.

Contract with the shell (METTLE_ONLY_CUTOVER_CONTRACT.md §8):
- binds 127.0.0.1 on the port passed as ``--port`` (the shell picks a free
  port — never hardcode 8760, a dev `dk serve` may hold it);
- env carries the resource paths: ``DK_ENGINE_BINARY`` (bundled mettled) and
  optionally ``DK_METTLE_CONFIG``;
- on SIGTERM/SIGINT (the shell quitting) uvicorn shuts down and the FastAPI
  shutdown hook stops the engine supervisor, killing mettled's process group.
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
                log_level="info")


if __name__ == "__main__":
    main()