#!/usr/bin/env bash
# docket uninstaller — shipped from day one (docs/decisions.md Q18).
# Removes exactly what install.sh recorded in the manifest; asks first.
set -euo pipefail

PREFIX="${DK_PREFIX:-$HOME/.local/share/docket}"
MANIFEST="$PREFIX/.install-manifest.txt"

if [ ! -f "$MANIFEST" ]; then
  echo "no install manifest at $MANIFEST — nothing to uninstall."
  exit 0
fi

echo "This will remove:"; sed 's/^/  - /' "$MANIFEST"
printf "proceed? [y/N] "; read -r ans
case "$ans" in
  y|Y|yes) ;;
  *) echo "aborted."; exit 0 ;;
esac

# Remove recorded paths (reverse order so files go before their dirs).
tac "$MANIFEST" | while IFS= read -r path; do
  [ -e "$path" ] && rm -rf "$path" && echo "removed $path"
done
rm -f "$MANIFEST"
echo "docket uninstalled. (Downloaded models, if any, were left in place.)"
