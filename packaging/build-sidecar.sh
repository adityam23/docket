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

# 4. Boot smoke on the FROZEN binary (the _MEIPASS trap only bites after
# freezing, so a green build proves nothing without a boot):
#   - /api/health answers with the provider JSON
#   - / serves the real SPA (contains a hashed /_app/immutable asset, which
#     the "dashboard not built" fallback page never does)
#   - a hashed immutable asset actually loads (the /_app mount resolved)
#   - SIGTERM produces a graceful uvicorn shutdown
smoke_sidecar() {
  local port
  port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
  ./dist/docket-server/docket-server --port "$port" > /tmp/dksidecar-smoke.log 2>&1 &
  local spid=$!
  for _ in $(seq 1 60); do
    curl -sf -m 2 "http://127.0.0.1:$port/api/health" > /dev/null 2>&1 && break
    sleep 1
  done
  local fail=0
  curl -sf -m 5 "http://127.0.0.1:$port/api/health" | grep -q '"ok"[[:space:]]*:[[:space:]]*true' || { echo "smoke: /api/health failed" >&2; fail=1; }
  local asset
  asset=$(curl -sf -m 5 "http://127.0.0.1:$port/" | grep -o '/_app/immutable/entry/[^"]*' | head -1)
  [ -n "$asset" ] || { echo "smoke: / does not serve the built SPA" >&2; fail=1; }
  if [ -n "$asset" ]; then
    curl -sf -m 5 -o /dev/null "http://127.0.0.1:$port$asset" || { echo "smoke: immutable asset failed: $asset" >&2; fail=1; }
  fi
  kill -TERM "$spid" 2>/dev/null || true
  for _ in $(seq 1 10); do kill -0 "$spid" 2>/dev/null || break; sleep 0.5; done
  grep -q "Finished server process" /tmp/dksidecar-smoke.log || { echo "smoke: SIGTERM did not shut down gracefully" >&2; fail=1; }
  rm -f /tmp/dksidecar-smoke.log
  return $fail
}
smoke_sidecar || { echo "FROZEN SIDECAR SMOKE FAILED — see /tmp/dksidecar-smoke.log" >&2; exit 1; }

echo "sidecar ready: dist/docket-server/docket-server"