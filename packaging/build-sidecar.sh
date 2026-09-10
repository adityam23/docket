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
  local port fail spid fake_dir fake_port fake fakepid
  fail=0
  port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')

  # --- boot markers --------------------------------------------------------
  ./dist/docket-server/docket-server --port "$port" > /tmp/dksidecar-smoke.log 2>&1 &
  spid=$!
  for _ in $(seq 1 60); do
    curl -sf -m 2 "http://127.0.0.1:$port/api/health" > /dev/null 2>&1 && break
    sleep 1
  done
  # "provider" (config-derived) instead of "ok": CI has no live backend, and
  # ok reflects backend reachability, not sidecar health.
  curl -sf -m 5 "http://127.0.0.1:$port/api/health" | grep -q '"provider"' || { echo "smoke: /api/health failed" >&2; fail=1; }
  local asset
  asset=$(curl -sf -m 5 "http://127.0.0.1:$port/" | grep -o '/_app/immutable/entry/[^"]*' | head -1)
  [ -n "$asset" ] || { echo "smoke: / does not serve the built SPA" >&2; fail=1; }
  if [ -n "$asset" ]; then
    curl -sf -m 5 -o /dev/null "http://127.0.0.1:$port$asset" || { echo "smoke: immutable asset failed: $asset" >&2; fail=1; }
  fi
  kill -TERM "$spid" 2>/dev/null || true
  for _ in $(seq 1 10); do kill -0 "$spid" 2>/dev/null || break; sleep 0.5; done
  grep -q "Finished server process" /tmp/dksidecar-smoke.log || { echo "smoke: SIGTERM did not shut down gracefully" >&2; fail=1; }

  # --- quit-on-close (cutover contract §8): the sidecar takes the spawned
  # engine's process group down with it. Drive the REAL path: fake engine
  # via DK_ENGINE_BINARY, teardown opt-in via DK_ENGINE_EPHEMERAL=1 — both
  # exactly what the Tauri shell sets at spawn.
  fake_dir=$(mktemp -d)
  fake="$fake_dir/fake-mettled"
  printf '#!/bin/sh\necho $$ > %s/fake.pid\nsleep 600\n' "$fake_dir" > "$fake"
  chmod +x "$fake"
  fake_port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
  # DK_BACKEND_URL points at a closed port: the supervisor must see
  # "probe fails + binary configured" to take the spawn path (a live engine
  # at the default URL would be honoured, not double-spawned).
  DK_BACKEND_URL="http://127.0.0.1:1/v1" DK_ENGINE_BINARY="$fake" \
    DK_ENGINE_EPHEMERAL=1 \
    ./dist/docket-server/docket-server --port "$fake_port" > /tmp/dksidecar-smoke2.log 2>&1 &
  spid=$!
  for _ in $(seq 1 60); do
    curl -sf -m 2 "http://127.0.0.1:$fake_port/api/health" > /dev/null 2>&1 && break
    sleep 1
  done
  sleep 2  # let the supervisor spawn the (never-answering) fake engine
  fakepid=$(cat "$fake_dir/fake.pid" 2>/dev/null || true)
  if [ -n "$fakepid" ] && [ -d "/proc/$fakepid" ]; then
    kill -TERM "$spid" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$spid" 2>/dev/null || break; sleep 0.5; done
    sleep 1
    if [ -d "/proc/$fakepid" ]; then
      echo "smoke: engine group SURVIVED sidecar quit (pid $fakepid)" >&2
      kill -9 "$fakepid" 2>/dev/null || true
      fail=1
    fi
  else
    echo "smoke: fake engine was never spawned (pid file missing)" >&2
    kill -9 "$spid" 2>/dev/null || true
    fail=1
  fi
  rm -rf "$fake_dir" /tmp/dksidecar-smoke.log /tmp/dksidecar-smoke2.log
  return $fail
}
smoke_sidecar || { echo "FROZEN SIDECAR SMOKE FAILED" >&2; exit 1; }

echo "sidecar ready: dist/docket-server/docket-server"