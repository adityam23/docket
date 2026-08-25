#!/usr/bin/env bash
# docket installer (Linux/macOS).
#
# Installs the latest GitHub release as an isolated `uv tool` (the `dk`
# command lands on PATH) plus a fully self-contained dashboard — no Node,
# no separate engine required. Any OpenAI-compatible /v1 backend works;
# generation also runs on a free Cerebras/Groq key.
#
# Overrides:
#   DK_WHEEL=path/to/docket.whl   install a local wheel instead of downloading
#   DK_REPO=owner/repo            release repo       (default adityam23/docket)
#   DK_PREFIX=~/.local/share/docket                 manifests + downloads
#   DK_BIN_DIR=~/.local/bin                         where the `dk` shim lands
#   ./install.sh --with-engine URL|PATH             optional local-engine hook:
#     installs the binary as `docket-engine` and records it in the manifest
#     (it is NOT started or configured here).
set -euo pipefail

PREFIX=${DK_PREFIX:-$HOME/.local/share/docket}
REPO=${DK_REPO:-adityam23/docket}
BIN_DIR=${DK_BIN_DIR:-$HOME/.local/bin}
MANIFEST="$PREFIX/.install-manifest.txt"
mkdir -p "$PREFIX"
: > "$MANIFEST"
record() { printf '%s\n' "$1" >> "$MANIFEST"; }

WHEEL=${DK_WHEEL:-}
if [ -z "$WHEEL" ]; then
  command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }
  api="https://api.github.com/repos/$REPO/releases/latest"
  echo "fetching latest release from $api"
  asset=$(curl -fsSL "$api" \
    | sed -n 's/.*"browser_download_url": *"\([^"]*\.whl\)".*/\1/p' | head -n 1)
  [ -n "$asset" ] || { echo "error: no .whl asset on the latest release" >&2; exit 1; }
  WHEEL="$PREFIX/$(basename "$asset")"
  curl -fsSL "$asset" -o "$WHEEL"
fi
record "$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")"

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Pin where the `dk` shim lands: uv otherwise derives the bin dir from
# XDG_DATA_HOME, which surprises anyone with a custom data home.
export UV_TOOL_BIN_DIR="$BIN_DIR"
mkdir -p "$UV_TOOL_BIN_DIR"

uv tool install --force "$WHEEL"
record "uv-tool:docket"
record "$UV_TOOL_BIN_DIR/dk"

if [ "${1:-}" = "--with-engine" ]; then
  shift
  src=${1:?--with-engine requires a URL or local path to an engine binary}
  dst="$BIN_DIR/docket-engine"
  mkdir -p "$BIN_DIR"
  case "$src" in
    http://*|https://*) curl -fsSL "$src" -o "$dst" ;;
                    *) cp "$src" "$dst" ;;
  esac
  chmod +x "$dst"
  record "$dst"
  echo "engine binary installed at $dst"
  echo "start it yourself (e.g. \"$dst serve\") and point DK_BACKEND_URL at its /v1."
fi

echo
echo "installed. try:"
echo "  $UV_TOOL_BIN_DIR/dk serve   # dashboard + API on http://127.0.0.1:8760"
echo "  $UV_TOOL_BIN_DIR/dk health  # check the configured /v1 backend"
echo "uninstall: uninstall.sh (this dir), or: uv tool uninstall docket"
