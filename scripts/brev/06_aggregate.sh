#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${ASR_BENCHMARK_PYTHON_BIN:-${PROJECT_DIR}/work/.venv/bin/python}"
cd "${PROJECT_DIR}"

ACCURACY_DIR=""
ACCURACY_STATUS_DIR=""
PERFORMANCE_DIR=""
PERFORMANCE_STATUS_DIR=""
OUTPUT_DIR="results/aggregated"
FIGURE_DIR="results/figures"

usage() {
  cat <<'EOF'
Usage:
  06_aggregate.sh \
    [--accuracy-dir DIR] [--accuracy-status-dir DIR] \
    [--performance-dir DIR] [--performance-status-dir DIR] \
    [--output-dir DIR] [--figure-dir DIR]

Specify at least one result directory. Status directories are optional checks.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accuracy-dir)
      ACCURACY_DIR="$2"
      shift 2
      ;;
    --accuracy-status-dir)
      ACCURACY_STATUS_DIR="$2"
      shift 2
      ;;
    --performance-dir)
      PERFORMANCE_DIR="$2"
      shift 2
      ;;
    --performance-status-dir)
      PERFORMANCE_STATUS_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --figure-dir)
      FIGURE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${ACCURACY_DIR}" && -z "${PERFORMANCE_DIR}" ]]; then
  echo "Specify at least one of --accuracy-dir or --performance-dir" >&2
  usage >&2
  exit 2
fi
if [[ -z "${ACCURACY_DIR}" && -n "${ACCURACY_STATUS_DIR}" ]]; then
  echo "--accuracy-status-dir requires --accuracy-dir" >&2
  usage >&2
  exit 2
fi
if [[ -z "${PERFORMANCE_DIR}" && -n "${PERFORMANCE_STATUS_DIR}" ]]; then
  echo "--performance-status-dir requires --performance-dir" >&2
  usage >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${FIGURE_DIR}"

AGGREGATE_ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --figure-dir "${FIGURE_DIR}"
)

if [[ -n "${ACCURACY_DIR}" ]]; then
  VERIFY_ACCURACY_ARGS=(
    --matrix configs/matrix.yaml
    --input-dir "${ACCURACY_DIR}"
    --available-only
    --output "${OUTPUT_DIR}/accuracy-verification.json"
  )
  if [[ -n "${ACCURACY_STATUS_DIR}" ]]; then
    VERIFY_ACCURACY_ARGS+=(--status-dir "${ACCURACY_STATUS_DIR}")
  fi
  "${PYTHON_BIN}" scripts/verify_results.py "${VERIFY_ACCURACY_ARGS[@]}"
  AGGREGATE_ARGS+=(--accuracy-input-dir "${ACCURACY_DIR}")
fi

if [[ -n "${PERFORMANCE_DIR}" ]]; then
  VERIFY_PERFORMANCE_ARGS=(
    --matrix configs/matrix.yaml
    --input-dir "${PERFORMANCE_DIR}"
    --available-only
    --output "${OUTPUT_DIR}/performance-verification.json"
  )
  if [[ -n "${PERFORMANCE_STATUS_DIR}" ]]; then
    VERIFY_PERFORMANCE_ARGS+=(--status-dir "${PERFORMANCE_STATUS_DIR}")
  fi
  "${PYTHON_BIN}" scripts/verify_results.py "${VERIFY_PERFORMANCE_ARGS[@]}"
  AGGREGATE_ARGS+=(--performance-input-dir "${PERFORMANCE_DIR}")
fi

"${PYTHON_BIN}" scripts/aggregate_results.py "${AGGREGATE_ARGS[@]}"
