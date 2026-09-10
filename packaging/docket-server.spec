# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the docket desktop sidecar (onedir).

The dev/CLI path stays `dk serve`; this spec exists only for the desktop
bundle (METTLE_ONLY_CUTOVER_CONTRACT.md §8). Build with the repo-root script:
``packaging/build-sidecar.sh`` — it prepares the venv and frontend build, then
invokes PyInstaller with this spec.

Layout contract with the shell:
- the analysis collects the whole docket package (the entry imports the app
  OBJECT directly, so static analysis sees every import — no hiddenimport
  guessing for the app itself);
- the SvelteKit SPA is bundled as DATA at the same relative path the source
  layout computes at import time (``docket/web/frontend/build``), so
  ``app.py``'s ``_FRONTEND`` resolves inside ``sys._MEIPASS`` frozen too;
- mettled is NOT bundled here — it is a sibling Tauri resource (see
  src-tauri/tauri.conf.json), located via DK_ENGINE_BINARY;
- langchain-core/langgraph use dynamic (string-based) imports internally, so
  they are collected wholesale below — that dynamism is exactly what static
  analysis cannot see.
"""

import os
from PyInstaller.utils.hooks import collect_all

SPECPATH = os.path.dirname(SPEC)
REPO = os.path.abspath(os.path.join(SPECPATH, ".."))
FRONTEND_BUILD = os.path.join(REPO, "docket", "web", "frontend", "build")

datas = [(os.path.join(FRONTEND_BUILD), "docket/web/frontend/build")]
binaries = []
hiddenimports = ["docket", "docket.web.app"]
for pkg in ("langchain_core", "langgraph", "pypdf", "cryptography"):
    r, b, d = collect_all(pkg)
    datas += r
    binaries += b
    hiddenimports += d

a = Analysis(
    ["docket_server_main.py"],
    pathex=[REPO],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="docket-server",
    console=True,          # logs stream to the shell's log while debugging
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="docket-server",
)