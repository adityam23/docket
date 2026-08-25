#!/usr/bin/env bash
# Self-healing watchdog for a shared OpenAI-/v1 backend during long sweeps.
#
# A llama.cpp-based engine can die spontaneously or under load right after a
# fresh start; when it does, the whole recall sweep stalls in its wait-retry
# loop until someone restarts the backend. This watchdog fills that role: poll
# /v1/models, and if the backend is down and no engine process is alive,
# relaunch it. It only RUNS the existing binary (never modifies anything).
# Bounded restart count as a backstop.
#
# Configuration via environment:
#   BASE          /v1 base URL            (default http://127.0.0.1:11434/v1)
#   ENGINE_BIN    absolute path to the engine binary (required)
#   ENGINE_ARGS   arguments after the binary (default "serve")
#   ENGINE_LOG    engine stdout/stderr    (default /tmp/docket-engine.log)
set -u

BASE=${BASE:-http://127.0.0.1:11434/v1}
ENGINE_BIN=${ENGINE_BIN:?set ENGINE_BIN to the absolute path of your engine binary}
ENGINE_ARGS=${ENGINE_ARGS:-serve}
ENGINE_LOG=${ENGINE_LOG:-/tmp/docket-engine.log}
WD_LOG=${WD_LOG:-/tmp/engine_watchdog.log}
MAX_RESTARTS=${MAX_RESTARTS:-40}
restarts=0

echo "=== [watchdog] started $(date -Iseconds) ===" >> "$WD_LOG"
while true; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$BASE/models" 2>/dev/null)
  if [ "$code" != "200" ]; then
    # Give an in-flight start a moment; only act if truly no engine process.
    if ! pgrep -f "$(basename "$ENGINE_BIN") $ENGINE_ARGS" >/dev/null 2>&1; then
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        echo "=== [watchdog] hit MAX_RESTARTS=$MAX_RESTARTS, giving up $(date -Iseconds) ===" >> "$WD_LOG"
        exit 1
      fi
      restarts=$((restarts+1))
      echo "=== [watchdog] backend down (code=$code), restart #$restarts $(date -Iseconds) ===" >> "$WD_LOG"
      # Roll the engine log so a fresh crash backtrace isn't buried.
      [ -f "$ENGINE_LOG" ] && mv "$ENGINE_LOG" "${ENGINE_LOG}.$(date +%s)" 2>/dev/null
      setsid nohup "$ENGINE_BIN" $ENGINE_ARGS > "$ENGINE_LOG" 2>&1 < /dev/null &
    fi
  fi
  sleep 20
done
