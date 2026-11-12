#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PID_FILE="${PROJECT_DIR}/results/full-benchmark.pid"
LOG_FILE="${PROJECT_DIR}/results/full-benchmark.log"

cd "${PROJECT_DIR}"
mkdir -p results

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(<"${PID_FILE}")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "benchmark already running: pid=${existing_pid}"
    exit 0
  fi
fi

nohup bash scripts/brev/05_run_performance.sh >"${LOG_FILE}" 2>&1 </dev/null &
benchmark_pid=$!
echo "${benchmark_pid}" >"${PID_FILE}"
echo "benchmark started: pid=${benchmark_pid} log=${LOG_FILE}"
