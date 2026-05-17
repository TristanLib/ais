#!/bin/bash
# Run the conservative publication experiment scaffold.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiment_conservative.yaml}"

mkdir -p outputs/audit outputs/final

{
  echo "timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "python=$PYTHON_BIN"
  echo "config=$CONFIG_PATH"
  echo "command=$0 $*"
} > outputs/audit/run_command.txt

git status --short > outputs/audit/git_status.txt || true

"$PYTHON_BIN" scripts/check_env.py --report-only --output outputs/audit/environment.json
AUDIT_ARGS=(--config "$CONFIG_PATH")
if [[ "${DATA_AUDIT_HASH:-true}" == "true" ]]; then
  AUDIT_ARGS+=(--hash)
fi
if [[ "${DATA_AUDIT_PROFILE_RAW:-true}" == "true" ]]; then
  AUDIT_ARGS+=(--profile-raw)
fi
"$PYTHON_BIN" scripts/audit_dataset.py "${AUDIT_ARGS[@]}"
"$PYTHON_BIN" scripts/final_train_eval.py --config "$CONFIG_PATH" "$@"
if [[ $# -eq 0 ]]; then
  "$PYTHON_BIN" scripts/check_metric_stability.py \
    --config "$CONFIG_PATH" \
    --output outputs/final/reproducibility_check.json \
    --python-bin "$PYTHON_BIN" \
    --strict
fi
"$PYTHON_BIN" scripts/final_stats.py \
  --errors outputs/final/per_sample_errors.csv \
  --output outputs/final/statistical_tests.json \
  --reference constant_velocity
"$PYTHON_BIN" scripts/make_figures.py \
  --output-dir outputs/final
"$PYTHON_BIN" scripts/make_paper_assets.py \
  --output-dir outputs/final \
  --audit-dir outputs/audit \
  --paper-dir paper
"$PYTHON_BIN" scripts/make_manuscript.py \
  --output-dir outputs/final \
  --audit-dir outputs/audit \
  --paper-dir paper \
  --output paper/conservative_manuscript.md
READINESS_ARGS=(
  --config "$CONFIG_PATH"
  --output-dir outputs/final
  --audit-dir outputs/audit
  --paper-dir paper
  --output outputs/final/publication_readiness_report.json
)
if [[ $# -eq 0 ]]; then
  READINESS_ARGS+=(--strict)
fi
"$PYTHON_BIN" scripts/audit_publication_pack.py "${READINESS_ARGS[@]}"

echo "Conservative publication experiment complete."
