#!/bin/bash
# Xcode/Instruments decode capture launcher (see
# docs/bench/2026-07-03-xcode-decode-capture/README.md).
#
# Usage: scripts/profile/xcode_capture_decode_window.sh {a3b|27b|a10b} [ctx] [window]
#
# Ramps qwen-bench decode-window to the target context, prints the PID for
# profiler attachment, and waits. Release the decode window with:
#   touch /tmp/qwen-capture.go
set -euo pipefail

MODEL_KEY="${1:-a3b}"
CTX="${2:-16384}"
WINDOW="${3:-256}"

case "$MODEL_KEY" in
  a3b)  MODEL="$HOME/models/Qwen3.5-35B-A3B-Q4_K_M.gguf" ;;
  27b)  MODEL="$HOME/models/Qwen3.5-27B-Q4_K_M.gguf" ;;
  a10b) MODEL="$HOME/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf" ;;
  *) echo "unknown model key: $MODEL_KEY (want a3b|27b|a10b)"; exit 1 ;;
esac

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="$ROOT/target/release/qwen-bench"
[ -x "$BENCH" ] || { echo "build first: cargo build --release -p qwen-cli --bin qwen-bench"; exit 1; }

READY=/tmp/qwen-capture.ready
GO=/tmp/qwen-capture.go
rm -f "$READY" "$GO"

echo "[capture] launching decode-window: $MODEL_KEY ctx=$CTX window=$WINDOW"
echo "[capture] (MTL_CAPTURE_ENABLED=1 for the Xcode GPU-capture path)"
MTL_CAPTURE_ENABLED=1 "$BENCH" decode-window \
  -m "$MODEL" \
  --target-ctx "$CTX" \
  --window "$WINDOW" \
  --ready-file "$READY" \
  --go-file "$GO" &
PID=$!

echo "[capture] ramping to ctx=$CTX (this takes a bit)..."
while [ ! -f "$READY" ]; do
  kill -0 "$PID" 2>/dev/null || { echo "[capture] bench exited early"; exit 1; }
  sleep 1
done

cat <<EOF

[capture] READY.  PID: $PID
[capture] 1. Attach Instruments (Metal System Trace + GPU Counters) or
[capture]    Xcode (Debug > Attach to Process by PID: $PID) now.
[capture] 2. Start recording.
[capture] 3. Release the window:   touch $GO
[capture] 4. Stop recording after the window completes (~3-10 s).

EOF
wait "$PID"
