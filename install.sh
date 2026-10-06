#!/usr/bin/env bash
# docket installer (Linux/macOS).
#
# Installs the latest GitHub release as an isolated `uv tool` (the `dk`
# command lands on PATH) plus a fully self-contained dashboard — no Node
# required. docket needs the mettle engine (`mettled`) to generate: point
# DK_ENGINE_BINARY at a mettled binary and `dk serve` supervises it for you,
# or point DK_BACKEND_URL at an already-running mettled's /v1. Set
# DK_ENGINE_KEY only if that engine enforces auth.
#
# Overrides:
#   DK_WHEEL=path/to/docket.whl   install a local wheel instead of downloading
#   DK_REPO=owner/repo            release repo       (default adityam23/docket)
#   DK_PREFIX=~/.local/share/docket-install   installer bookkeeping only —
#                                              manifest + a downloaded wheel.
#                                              Never the data dir: the corpus/
#                                              traces/chats live under
#                                              $XDG_DATA_HOME/docket (a sibling
#                                              dir — see docket/config.py) so
#                                              uninstall.sh can never reach
#                                              them without --data.
#   DK_BIN_DIR=~/.local/bin                   where the `dk` shim lands
#   ./install.sh --with-engine URL|PATH       install a local mettled binary:
#     copies/downloads it to `docket-engine` next to `dk` and records it in
#     the manifest (it is NOT started or configured here — set
#     DK_ENGINE_BINARY to its path for `dk serve` to supervise it).
set -euo pipefail

PREFIX=${DK_PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/docket-install}
REPO=${DK_REPO:-adityam23/docket}
BIN_DIR=${DK_BIN_DIR:-$HOME/.local/bin}
MANIFEST="$PREFIX/.install-manifest.txt"

# The install prefix is bookkeeping-only (manifest + a downloaded wheel) and
# must never coincide with or contain the data dir docket/config.py writes
# the corpus/traces/chats under — otherwise uninstall.sh's cleanup could
# destroy user data. Guard it rather than rely on the default staying right.
DATA_DIR=${XDG_DATA_HOME:-$HOME/.local/share}/docket
case "$PREFIX/" in
  "$DATA_DIR/")
    echo "error: DK_PREFIX ($PREFIX) must not be the data dir ($DATA_DIR)" >&2
    exit 1
    ;;
  "$DATA_DIR"/*)
    echo "error: DK_PREFIX ($PREFIX) must not be inside the data dir ($DATA_DIR)" >&2
    exit 1
    ;;
esac
case "$DATA_DIR/" in
  "$PREFIX"/*)
    echo "error: the data dir ($DATA_DIR) must not be inside DK_PREFIX ($PREFIX)" >&2
    exit 1
    ;;
esac

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
  # Only record paths the installer itself created under the prefix — a
  # user-supplied DK_WHEEL is the user's own file and must never land in the
  # manifest (uninstall.sh would otherwise delete it from under them).
  record "$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")"
fi

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
  src=${1:?--with-engine requires a URL or local path to a mettled binary}
  dst="$BIN_DIR/docket-engine"
  mkdir -p "$BIN_DIR"
  case "$src" in
    http://*|https://*) curl -fsSL "$src" -o "$dst" ;;
                    *) cp "$src" "$dst" ;;
  esac
  chmod +x "$dst"
  record "$dst"
  echo "engine binary installed at $dst"
  echo "set DK_ENGINE_BINARY=$dst (e.g. in ~/.config/docket/env.local) so"
  echo "\"dk serve\" supervises it, or run it yourself and point"
  echo "DK_BACKEND_URL at its /v1."
fi

echo
echo "installed. try:"
echo "  $UV_TOOL_BIN_DIR/dk serve   # dashboard + API on http://127.0.0.1:8760"
echo "  $UV_TOOL_BIN_DIR/dk health  # check the configured mettle engine"
echo "uninstall: uninstall.sh (this dir), or: uv tool uninstall docket"
