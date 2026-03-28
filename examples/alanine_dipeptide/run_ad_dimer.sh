#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/alanine_dipeptide_example.yaml}"
shift || true
ENV_NAME="${GGPA_CONDA_ENV:-}"
LOG_DIR="$ROOT/examples/alanine_dipeptide/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/ad_dimer.log"
PID_FILE="$LOG_DIR/ad_dimer.pid"
JOB_FILE="$LOG_DIR/ad_dimer_job.sh"

EXTRA_ARGS_STR=""
if [ "$#" -gt 0 ]; then
  printf -v EXTRA_ARGS_STR '%q ' "$@"
fi

cat >"$JOB_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
trap 'rm -f "$PID_FILE" "$JOB_FILE"' EXIT
python -u "$ROOT/examples/alanine_dipeptide/run_ad_dimer.py" --config "$CONFIG" $EXTRA_ARGS_STR
echo
echo "AD dimer RE example complete. Regenerating figures..."
python -u "$ROOT/examples/alanine_dipeptide/plot_alanine_results.py" --config "$CONFIG" ad-dimer
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
echo "Started AD dimer RE example in background."
echo "  env: $ENV_MSG"
echo "  log: $LOG_FILE"
echo "  pid: $(cat "$PID_FILE")"
