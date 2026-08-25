#!/usr/bin/env bash
# docket uninstaller. Removes exactly what install.sh put down:
#   - the `uv tool` environment providing `dk`
#   - any engine binary installed via --with-engine
#   - the manifest + download prefix
#   - with --data: the ingested index too (~/.local/share/docket/index)
# Configuration never lives outside env vars and those files.
set -euo pipefail

PREFIX=${DK_PREFIX:-$HOME/.local/share/docket}
MANIFEST="$PREFIX/.install-manifest.txt"

if command -v uv >/dev/null 2>&1; then
  uv tool uninstall docket 2>/dev/null || true
fi

if [ -f "$MANIFEST" ]; then
  while IFS= read -r entry; do
    case $entry in
      uv-tool:*) ;;                                  # handled above
      /*) rm -f "$entry" ;;
    esac
  done < "$MANIFEST"
fi

if [ "${1:-}" = "--data" ]; then
  data=${XDG_DATA_HOME:-$HOME/.local/share}/docket/index
  rm -rf "$data"
  echo "removed index: $data"
fi

rm -rf "$PREFIX"
echo "docket removed."
