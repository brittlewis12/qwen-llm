#!/usr/bin/env bash
# Boot `qwen serve` and wait for readiness, failing fast when the process
# dies. Written after a readiness loop polled an already-exited server for
# 33 minutes because it only probed the socket and never checked liveness.
#
# Usage: scripts/serve/boot.sh <log-path> [serve args...]
set -euo pipefail

BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/target/release/qwen"
LOG="${1:?usage: boot.sh <log-path> [serve args...]}"
shift

[[ -x "$BIN" ]] || { echo "boot: $BIN is not executable" >&2; exit 1; }

QWEN_METAL_LEASE_WAIT="${QWEN_METAL_LEASE_WAIT:-1}" \
RUST_LOG="${RUST_LOG:-warn,qwen_diag=info}" \
  "$BIN" serve "$@" >/dev/null 2>"$LOG" &
PID=$!
echo "boot: pid=$PID binary=$BIN"

ADDR="127.0.0.1:8737"
for arg in "$@"; do
  [[ "${PREV:-}" == "--addr" ]] && ADDR="$arg"
  PREV="$arg"
done

DEADLINE=$((SECONDS + ${BOOT_TIMEOUT_S:-1800}))
while true; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "boot: FAILED — process exited during startup:" >&2
    tail -5 "$LOG" >&2
    exit 1
  fi
  if curl -s -m 3 "http://${ADDR}/v1/models" >/dev/null 2>&1; then
    echo "boot: ready after $((SECONDS)) s"
    exit 0
  fi
  if (( SECONDS > DEADLINE )); then
    echo "boot: FAILED — not ready within ${BOOT_TIMEOUT_S:-1800}s" >&2
    tail -5 "$LOG" >&2
    kill -TERM "$PID" 2>/dev/null || true
    exit 1
  fi
  sleep 5
done
