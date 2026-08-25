#!/usr/bin/env bash
# docket installer (Linux/macOS). Prebuilt-binary-first — no Rust
# compile for end users (docs/decisions.md Q18). Records an install manifest so
# uninstall.sh removes everything cleanly.
#
# STATUS: skeleton. TODO(phase-2): pin the real prebuilt-backend release URL and
# wire model auto-pull. Do not ship until the TODOs are resolved.
set -euo pipefail

PREFIX="${DK_PREFIX:-$HOME/.local/share/docket}"
BIN_DIR="${DK_BIN_DIR:-$HOME/.local/bin}"
MANIFEST="$PREFIX/.install-manifest.txt"

OS="$(uname -s)"; ARCH="$(uname -m)"
echo "docket installer — os=$OS arch=$ARCH prefix=$PREFIX"

mkdir -p "$PREFIX" "$BIN_DIR"
: > "$MANIFEST"
record() { echo "$1" >> "$MANIFEST"; }
record "$PREFIX"

# --- backend detection (docs/decisions.md Q18) --------------------------------
BACKEND="cpu"
if command -v nvidia-smi >/dev/null 2>&1; then BACKEND="cuda"
elif [ "$OS" = "Darwin" ]; then BACKEND="metal"; fi
echo "detected backend: $BACKEND (weak hardware can fall back to a free-tier API)"

# --- TODO(phase-2): download prebuilt llama-server/infengine binary -----------
# URL="https://<release-host>/llama-server-$OS-$ARCH-$BACKEND"
# curl -fsSL "$URL" -o "$PREFIX/llama-server" && chmod +x "$PREFIX/llama-server"
# record "$PREFIX/llama-server"
echo "TODO: pin backend binary release URL (see docs/roadmap.md)."

# --- python app via uv --------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv…"; curl -LsSf https://astral.sh/uv/install.sh | sh
fi
( cd "$(dirname "$0")" && uv sync )

echo "installed. manifest: $MANIFEST"
echo "run:  uv run dk serve   # then open http://127.0.0.1:8760"
