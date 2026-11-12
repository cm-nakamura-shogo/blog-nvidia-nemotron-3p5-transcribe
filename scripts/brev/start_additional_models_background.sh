#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${ASR_BENCHMARK_PYTHON_BIN:-${PROJECT_DIR}/work/.venv/bin/python}"
LOG_PATH="${PROJECT_DIR}/results/additional-models-benchmark.log"
PID_PATH="${PROJECT_DIR}/results/additional-models-benchmark.pid"

cd "${PROJECT_DIR}"
mkdir -p results
nohup "${PYTHON_BIN}" scripts/run_matrix.py \
  --matrix configs/matrix.yaml \
  --only parakeet-tdt-ctc-0.6b-ja reazonspeech-k2-v2 \
  --output-dir results/raw \
  --status-dir results/status \
  --resume \
  > "${LOG_PATH}" 2>&1 &
pid=$!
echo "${pid}" > "${PID_PATH}"
echo "Started additional model benchmark with PID ${pid}"
echo "Log: ${LOG_PATH}"
