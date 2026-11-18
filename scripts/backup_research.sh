#!/usr/bin/env bash
# Exclude prior restore trees/bundles so repeated research backups do not nest.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env.sh"
REAL_FS="$(readlink -f /root/autodl-fs)"
mountpoint -q "$REAL_FS"
LABEL="${1:?research phase label required}"
[[ "$LABEL" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 1
mkdir -p "$TSP_ARCHIVE/backups"
OUT="$TSP_ARCHIVE/backups/$LABEL-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar --exclude=code/.git --exclude='*/__pycache__' --exclude='*.tmp' \
  -czf "$OUT.tmp" -C "$TSP_ROOT" code artifacts logs checkpoints data/prepared/stage1
mv "$OUT.tmp" "$OUT"
gzip -t "$OUT"
sha256sum "$OUT" > "$OUT.sha256"
echo "$OUT"
cat "$OUT.sha256"
