#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/experiment_multiday.yaml}"
MAX_ROWS_PER_FILE="${MAX_ROWS_PER_FILE:-}"
SAMPLE_ROWS_PER_FILE="${SAMPLE_ROWS_PER_FILE:-}"
SAMPLE_TIME_BLOCKS="${SAMPLE_TIME_BLOCKS:-}"
CHUNKSIZE="${CHUNKSIZE:-250000}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-}"
MAX_RISK_EVAL_SAMPLES="${MAX_RISK_EVAL_SAMPLES:-5000}"
MAX_RISK_SCENARIOS="${MAX_RISK_SCENARIOS:-2000}"
RUN_NEURAL_TUNING="${RUN_NEURAL_TUNING:-true}"
TUNING_MAX_CONFIGS_PER_MODEL="${TUNING_MAX_CONFIGS_PER_MODEL:-2}"
TUNING_PROXY_TRAIN_SAMPLES="${TUNING_PROXY_TRAIN_SAMPLES:-8192}"
TUNING_PROXY_VAL_SAMPLES="${TUNING_PROXY_VAL_SAMPLES:-2048}"
RUN_LATEST_PREDICTION="${RUN_LATEST_PREDICTION:-false}"
LATEST_MAX_ROWS="${LATEST_MAX_ROWS:-1000000}"
LATEST_MAX_SAMPLES="${LATEST_MAX_SAMPLES:-200}"
DOWNLOAD_DATES="${DOWNLOAD_DATES:-false}"
DATES="${DATES:-}"
SKIP_RAW_CHECKSUMS="${SKIP_RAW_CHECKSUMS:-false}"
UPDATE_RAW_CHECKSUMS="${UPDATE_RAW_CHECKSUMS:-false}"

echo "==> Checking Python scripts"
"$PYTHON_BIN" -m py_compile \
  scripts/build_multiday_dataset.py \
  scripts/final_train_eval.py \
  scripts/final_train_eval_multiday.py \
  scripts/tune_neural_baselines.py \
  scripts/evaluate_risk_warning.py \
  scripts/predict_latest_ais.py \
  scripts/update_raw_checksums.py \
  scripts/make_submission_pack.py \
  scripts/audit_high_quality_pack.py

echo "==> Building metadata-rich AIS dataset"
build_args=(scripts/build_multiday_dataset.py --config "$CONFIG" --chunksize "$CHUNKSIZE")
if [[ "$DOWNLOAD_DATES" == "true" ]]; then
  build_args+=(--download)
  if [[ -n "$DATES" ]]; then
    read -r -a date_args <<< "$DATES"
    build_args+=(--dates "${date_args[@]}")
  fi
fi
if [[ -n "$MAX_ROWS_PER_FILE" ]]; then
  build_args+=(--max-rows-per-file "$MAX_ROWS_PER_FILE")
fi
if [[ -n "$SAMPLE_ROWS_PER_FILE" ]]; then
  build_args+=(--sample-rows-per-file "$SAMPLE_ROWS_PER_FILE")
fi
if [[ -n "$SAMPLE_TIME_BLOCKS" ]]; then
  build_args+=(--sample-time-blocks "$SAMPLE_TIME_BLOCKS")
fi
if [[ "$SKIP_RAW_CHECKSUMS" == "true" ]]; then
  build_args+=(--skip-raw-checksums)
fi
"$PYTHON_BIN" "${build_args[@]}"

if [[ "$UPDATE_RAW_CHECKSUMS" == "true" ]]; then
  echo "==> Computing raw-file checksums"
  "$PYTHON_BIN" scripts/update_raw_checksums.py
fi

echo "==> Running expanded multiday benchmark"
bench_args=(scripts/final_train_eval_multiday.py --config "$CONFIG")
if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
  bench_args+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "$MAX_TEST_SAMPLES" ]]; then
  bench_args+=(--max-test-samples "$MAX_TEST_SAMPLES")
fi
"$PYTHON_BIN" "${bench_args[@]}"

if [[ "$RUN_NEURAL_TUNING" == "true" ]]; then
  echo "==> Running neural validation-set tuning protocol"
  "$PYTHON_BIN" scripts/tune_neural_baselines.py \
    --config "$CONFIG" \
    --max-configs-per-model "$TUNING_MAX_CONFIGS_PER_MODEL" \
    --proxy-train-samples "$TUNING_PROXY_TRAIN_SAMPLES" \
    --proxy-val-samples "$TUNING_PROXY_VAL_SAMPLES"
fi

echo "==> Running AIS-derived risk-warning evaluation"
"$PYTHON_BIN" scripts/evaluate_risk_warning.py \
  --config "$CONFIG" \
  --max-eval-samples "$MAX_RISK_EVAL_SAMPLES" \
  --max-scenarios "$MAX_RISK_SCENARIOS"

if [[ "$RUN_LATEST_PREDICTION" == "true" ]]; then
  echo "==> Exporting latest/offline AIS predictions"
  "$PYTHON_BIN" scripts/predict_latest_ais.py \
    --config "$CONFIG" \
    --max-rows "$LATEST_MAX_ROWS" \
    --max-samples "$LATEST_MAX_SAMPLES"
fi

echo "==> Generating synchronized submission artifacts"
"$PYTHON_BIN" scripts/make_submission_pack.py

echo "==> Auditing high-quality submission readiness"
"$PYTHON_BIN" scripts/audit_high_quality_pack.py

echo "High-quality pipeline finished."
