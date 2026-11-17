#!/usr/bin/env bash
# Usage: bash scripts/start_job.sh SESSION COMMAND [ARG ...]
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env.sh"
NAME="${1:?tmux session name required}"
shift
[[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid tmux name'; exit 1; }
tmux has-session -t "=$NAME" 2>/dev/null && { echo "Already running: $NAME"; exit 1; }
mkdir -p "$TSP_ROOT/logs"
[[ ! -e "$TSP_ROOT/logs/$NAME.log" ]] || { echo "Previous log exists: $NAME; use a new session name"; exit 1; }
printf -v CMD '%q ' "$@"
printf -v ENV_FILE '%q' "$CODE/scripts/env.sh"
printf -v CODE_Q '%q' "$CODE"
printf -v LOG '%q' "$TSP_ROOT/logs/$NAME.log"
printf -v EXIT_FILE '%q' "$TSP_ROOT/logs/$NAME.exit"
# Record exit code even on failure; retain logs after session exits.
tmux new-session -d -s "$NAME" "bash -c 'source $ENV_FILE; cd $CODE_Q; $CMD > $LOG 2>&1; result=\$?; printf \"%s\\n\" \"\$result\" > $EXIT_FILE; exit \$result'"
echo "Session: $NAME; log: $TSP_ROOT/logs/$NAME.log"
