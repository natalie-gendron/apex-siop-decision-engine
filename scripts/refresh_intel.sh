#!/bin/zsh
# Local external-intelligence automation for the Apex SIOP Decision Engine.
# Usage: refresh_intel.sh monthly|weekly
# Scheduled by launchd (see scripts/launchd/); runs Claude Code headlessly.
set -euo pipefail

MODE="${1:?usage: refresh_intel.sh monthly|weekly}"
REPO="/Users/nataliegendron/Developer/10-projects/Apex SIOP Decision Engine"
LOGDIR="$HOME/Library/Logs/apex-siop"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

mkdir -p "$LOGDIR"
LOG="$LOGDIR/intel_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec >> "$LOG" 2>&1

echo "=== $(date) — intel $MODE run starting ==="
cd "$REPO"
git checkout main --quiet
git pull --quiet

claude -p "$(cat "scripts/intel_prompt_${MODE}.md")" \
  --model sonnet \
  --allowedTools "WebSearch,WebFetch,Read,Glob,Grep,Write,Edit,Bash(git:*),Bash(gh:*),Bash(.venv/bin/python:*)"

echo "=== $(date) — intel $MODE run finished (exit $?) ==="
