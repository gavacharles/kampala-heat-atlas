#!/bin/zsh
# Back up the working copy (incl. private notes/) and the data folder to OneDrive with rclone.
# The OneDrive app is NOT used: rclone talks to the OneDrive API directly, into a separate
# Backups/ folder, so it never competes with the app. Files that would be overwritten or deleted
# are moved to Backups/_versions/<timestamp>/ first, so nothing is lost by a sync.
#   ./scripts/backup.sh            code + notes + data (data/raw excluded: re-downloadable)
#   ./scripts/backup.sh --all      also back up data/raw (~2 GB, Overture + PIS + literature)
set -e
REMOTE=${KAMPALA_REMOTE:-onedrive}
STAMP=$(date +%Y%m%d-%H%M%S)
VERS="$REMOTE:Backups/_versions/$STAMP"
ROOT="${0:A:h}/.."
COMMON=(--fast-list --transfers 8 --checkers 16 --exclude "__pycache__/**" --exclude "*.egg-info/**" --exclude ".DS_Store")
echo "== code + notes"
rclone sync "$ROOT" "$REMOTE:Backups/kampala-heat-atlas" "${COMMON[@]}" --backup-dir "$VERS/kampala-heat-atlas" --stats-one-line -v 2>&1 | tail -2
echo "== data"
EXTRA=(--exclude "raw/**")
[[ "$1" == "--all" ]] && EXTRA=()
rclone sync ~/kampala-thesis-data "$REMOTE:Backups/kampala-thesis-data" "${COMMON[@]}" "${EXTRA[@]}" --backup-dir "$VERS/kampala-thesis-data" --stats-one-line -v 2>&1 | tail -2
echo "== done $(date +%H:%M)"
