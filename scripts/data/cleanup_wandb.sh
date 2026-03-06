#!/usr/bin/env bash
# Clean up old W&B offline run directories.
# Usage: bash scripts/data/cleanup_wandb.sh [--days N] [--delete]
#   --days N   Delete dirs older than N days (default: 14)
#   --delete   Actually delete (default: dry-run)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DAYS=14
DELETE=false
for arg in "$@"; do
    case "$arg" in
        --days) shift; DAYS="$1"; shift ;;
        --days=*) DAYS="${arg#*=}" ;;
        --delete) DELETE=true ;;
    esac
done

OLD_DIRS=$(find wandb/ -maxdepth 1 -type d -name "offline-run-*" -mtime +"$DAYS" 2>/dev/null || true)
COUNT=$(echo "$OLD_DIRS" | grep -c . 2>/dev/null || echo 0)

if [[ "$COUNT" -eq 0 ]]; then
    echo "No offline-run dirs older than $DAYS days."
    exit 0
fi

echo "Found $COUNT offline-run dirs older than $DAYS days:"
echo "$OLD_DIRS" | head -10
[[ "$COUNT" -gt 10 ]] && echo "  ... and $((COUNT - 10)) more"

SIZE=$(du -sh $(echo "$OLD_DIRS" | tr '\n' ' ') 2>/dev/null | tail -1 | cut -f1)
echo "Total size: ~$SIZE"

if $DELETE; then
    echo "$OLD_DIRS" | xargs rm -r
    echo "Deleted $COUNT dirs."
else
    echo "(dry-run) Pass --delete to remove."
fi
