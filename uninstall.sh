#!/usr/bin/env bash
# docket uninstaller. Removes exactly what install.sh put down:
#   - the `uv tool` environment providing `dk`
#   - any engine binary installed via --with-engine
#   - the manifest + install prefix (bookkeeping only — see install.sh)
#   - with --data: the ingested index (corpus/traces/chats) under
#     $XDG_DATA_HOME/docket, AND the config dir under $XDG_CONFIG_HOME/docket
#     (which may hold env.local, and with it DK_ENGINE_KEY)
#
# The install prefix is a dedicated bookkeeping dir, deliberately never the
# data dir (docket/config.py puts the corpus/traces/chats under
# $XDG_DATA_HOME/docket) — so removing it can never touch user data, with or
# without --data. Only the paths install.sh itself recorded in the manifest
# are deleted, and the prefix directory is removed only if that leaves it
# empty (never `rm -rf` on it).
set -euo pipefail

PREFIX=${DK_PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/docket-install}
# Installs made before the prefix moved off the data dir: still uninstallable
# (manifest-only cleanup below), never collapsed into a blind `rm -rf` of
# what is, for those installs, $XDG_DATA_HOME/docket itself.
LEGACY_PREFIX=$HOME/.local/share/docket
DATA_DIR=${XDG_DATA_HOME:-$HOME/.local/share}/docket
CONFIG_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/docket

if command -v uv >/dev/null 2>&1; then
  uv tool uninstall docket 2>/dev/null || true
fi

remove_manifest() {
  prefix=$1
  manifest="$prefix/.install-manifest.txt"
  [ -f "$manifest" ] || return 0
  while IFS= read -r entry; do
    case $entry in
      uv-tool:*) ;;                                  # handled above
      # Pre-fix manifests recorded a user-supplied DK_WHEEL; only a wheel
      # the installer downloaded (under the prefix) is ours to delete.
      "$prefix"/*.whl) rm -f "$entry" ;;
      *.whl) ;;
      /*) rm -f "$entry" ;;
    esac
  done < "$manifest"
  rm -f "$manifest"
  # Bookkeeping-only dir: drop it iff that was everything in it. Never
  # rm -rf — for the legacy prefix this directory doubles as the data dir.
  rmdir "$prefix" 2>/dev/null || true
}

remove_manifest "$PREFIX"
if [ "$LEGACY_PREFIX" != "$PREFIX" ]; then
  remove_manifest "$LEGACY_PREFIX"
fi

if [ "${1:-}" = "--data" ]; then
  rm -rf "$DATA_DIR/index"
  echo "removed index: $DATA_DIR/index"
  rm -rf "$CONFIG_DIR"
  echo "removed config: $CONFIG_DIR"
fi

echo "docket removed."
