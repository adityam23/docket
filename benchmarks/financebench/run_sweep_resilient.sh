#!/usr/bin/env bash
# Auto-restart wrapper for the Phase-A recall sweep.
#
# The shared infengine backend is restarted by co-tenant agents and the box runs
# under memory pressure (swap-heavy), so the sweep python is periodically killed.
# The sweep itself is fully resumable (corpora cached to .sweep_cache/, per-config
# checkpoint to --out, wait-and-retry on backend restarts), so the robust answer is
# to relaunch it on any non-zero exit until it exits 0 (clean completion).
set -u

cd /personal-projects/docket
LOG=/tmp/sweep_full.log
BASE_URL=${BASE_URL:-http://127.0.0.1:11434/v1}
DOCS=benchmarks/financebench/filings
OUT=benchmarks/financebench/results/recall_sweep.json
MAX_ATTEMPTS=${MAX_ATTEMPTS:-40}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== [runner] attempt $attempt starting $(date -Iseconds) ===" >> "$LOG"
  uv run python -u benchmarks/financebench/sweep_recall.py \
      --base-url "$BASE_URL" --docs-dir "$DOCS" --out "$OUT" >> "$LOG" 2>&1
  rc=$?
  echo "=== [runner] attempt $attempt exited rc=$rc $(date -Iseconds) ===" >> "$LOG"
  if [ "$rc" -eq 0 ]; then
    echo "=== [runner] sweep completed cleanly ===" >> "$LOG"
    exit 0
  fi
  sleep 15
done
echo "=== [runner] gave up after $MAX_ATTEMPTS attempts ===" >> "$LOG"
exit 1
