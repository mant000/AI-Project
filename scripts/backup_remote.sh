#!/usr/bin/env bash
# Archive code and measured artifacts to the verified file-storage mount.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env.sh"
FS_REAL="$(readlink -f /root/autodl-fs)"
mountpoint -q "$FS_REAL" || { echo 'File storage is not mounted; backup refused'; exit 1; }
FSTYPE="$(findmnt -n -o FSTYPE -T "$FS_REAL")"
[[ "$FSTYPE" != overlay && "$FSTYPE" != tmpfs ]] || exit 1
mkdir -p "$TSP_ARCHIVE/backups"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$TSP_ARCHIVE/backups/stage0-$STAMP.tar.gz"
ITEMS=(code artifacts logs checkpoints)
[[ ! -d "$TSP_ROOT/operations" ]] || ITEMS+=(operations)
tar --exclude=code/.git --exclude='*/__pycache__' --exclude='*.tmp' \
    -czf "$OUT.tmp" -C "$TSP_ROOT" "${ITEMS[@]}"
mv "$OUT.tmp" "$OUT"
sha256sum "$OUT" > "$OUT.sha256"
gzip -t "$OUT"
echo "$OUT"
cat "$OUT.sha256"
# Pull this archive and checksum to the local D: drive for independent backup.
