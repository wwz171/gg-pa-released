#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/configs/phi4_example.yaml"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
  CONFIG="$1"
  shift
fi
ENV_NAME="${GGPA_CONDA_ENV:-}"
LOG_DIR="$ROOT/examples/phi4/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/critical_scaling.log"
PID_FILE="$LOG_DIR/critical_scaling.pid"
JOB_FILE="$LOG_DIR/critical_scaling_job.sh"

EXTRA_ARGS_STR=""
if [ "$#" -gt 0 ]; then
  printf -v EXTRA_ARGS_STR '%q ' "$@"
fi

cat >"$JOB_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
trap 'rm -f "$PID_FILE" "$JOB_FILE"' EXIT
python -u "$ROOT/examples/phi4/phi4_re_scan.py" critical-scaling --config "$CONFIG" $EXTRA_ARGS_STR
echo
echo "Critical-scaling scan complete. Generating figures..."
python -u "$ROOT/examples/phi4/plot_phi4_results.py" --config "$CONFIG" critical-scaling
EOF
chmod +x "$JOB_FILE"

if [ -n "$ENV_NAME" ]; then
  nohup conda run --no-capture-output -n "$ENV_NAME" bash "$JOB_FILE" >"$LOG_FILE" 2>&1 &
  ENV_MSG="conda env: $ENV_NAME"
else
  nohup bash "$JOB_FILE" >"$LOG_FILE" 2>&1 &
  ENV_MSG="current environment"
fi

echo $! >"$PID_FILE"
echo "Started critical-scaling run in background."
echo "  env: $ENV_MSG"
echo "  log: $LOG_FILE"
echo "  pid: $(cat "$PID_FILE")"
