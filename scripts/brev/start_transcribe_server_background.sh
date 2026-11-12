#!/usr/bin/env bash
# vLLMサーバーをバックグラウンドで起動し、/v1/models が応答するまで待つ。
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG="${PROJECT_DIR}/results/nemotron-transcribe-server.log"
cd "${PROJECT_DIR}"
mkdir -p results

if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "server already running"
  exit 0
fi
started=$(date +%s)
nohup bash scripts/serve_nemotron_transcribe.sh >"${LOG}" 2>&1 &
echo $! > results/nemotron-transcribe-server.pid
for _ in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
    echo "server ready in $(( $(date +%s) - started ))s (cold start)"
    exit 0
  fi
  if ! kill -0 "$(cat results/nemotron-transcribe-server.pid)" 2>/dev/null; then
    tail -50 "${LOG}" >&2
    exit 1
  fi
  sleep 15
done
echo "server did not become ready within 30 minutes" >&2
exit 1
