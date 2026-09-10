#!/usr/bin/env bash
# Build the docket desktop sidecar (packaging/docket-server/…) with PyInstaller.
# Run from the repo root:  packaging/build-sidecar.sh
# Prereqs: uv, node/npm, and the frontend deps installed once
#          (cd docket/web/frontend && npm ci).
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Python env: everything the server can import at runtime.
uv sync --extra ingest --extra agent --quiet

# 2. Frontend: the SPA is served by the frozen server (same relative layout).
if [ ! -f docket/web/frontend/build/index.html ]; then
  (cd docket/web/frontend && npm ci --silent && npm run build --silent)
fi

# 3. Freeze. Output: dist/docket-server/
uv run --with pyinstaller pyinstaller --noconfirm --clean packaging/docket-server.spec

echo "sidecar ready: dist/docket-server/docket-server"