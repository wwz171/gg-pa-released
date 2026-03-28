#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/phi4_example.yaml}"
shift || true
LOG_DIR="$ROOT/examples/phi4/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/critical_scaling.log"
PID_FILE="$LOG_DIR/critical_scaling.pid"

EXTRA_ARGS_STR=""
if [ "$#" -gt 0 ]; then
  printf -v EXTRA_ARGS_STR '%q ' "$@"
fi

read -r -d '' JOB_SCRIPT <<EOF || true
set -euo pipefail
trap 'rm -f "$PID_FILE"' EXIT
python -u "$ROOT/examples/phi4/phi4_re_scan.py" critical-scaling --config "$CONFIG" $EXTRA_ARGS_STR
echo
echo "Critical-scaling scan complete. Generating figures..."
python -u "$ROOT/examples/phi4/plot_phi4_results.py" --config "$CONFIG" critical-scaling
EOF

nohup conda run --no-capture-output -n pyg bash -lc "$JOB_SCRIPT" >"$LOG_FILE" 2>&1 &

echo $! >"$PID_FILE"
echo "Started critical-scaling run in background."
echo "  log: $LOG_FILE"
echo "  pid: $(cat "$PID_FILE")"
