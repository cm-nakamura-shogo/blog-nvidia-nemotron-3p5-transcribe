#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${ASR_BENCHMARK_PYTHON_BIN:-${PROJECT_DIR}/work/.venv/bin/python}"
LOG_PATH="${PROJECT_DIR}/results/nemotron-offline-benchmark.log"
PID_PATH="${PROJECT_DIR}/results/nemotron-offline-benchmark.pid"
MODEL_ID="nemotron-3.5-asr-offline-transformers"

cd "${PROJECT_DIR}"
mkdir -p results/raw results/status

if [[ -f "${PID_PATH}" ]]; then
  existing_pid="$(tr -d '[:space:]' < "${PID_PATH}")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Nemotron offline benchmark is already running with PID ${existing_pid}" >&2
    exit 1
  fi
fi

nohup "${PYTHON_BIN}" scripts/run_matrix.py \
  --matrix configs/matrix.yaml \
  --only "${MODEL_ID}" \
  --output-dir results/raw \
  --status-dir results/status \
  --resume \
  > "${LOG_PATH}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" > "${PID_PATH}"

echo "Started ${MODEL_ID} benchmark with PID ${pid}"
echo "Log: ${LOG_PATH}"
