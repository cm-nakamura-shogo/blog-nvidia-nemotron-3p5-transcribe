#!/usr/bin/env bash
# 2つのセットアップを順番に実行する。brev execの再試行で二重起動しないようflockで排他する。
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p results
nohup flock -n results/.setup.lock bash -c '
  bash scripts/setup_brev.sh > results/setup-brev.log 2>&1; echo "EXIT=$?" >> results/setup-brev.log
  bash scripts/setup_nemotron_transcribe.sh > results/setup-transcribe.log 2>&1; echo "EXIT=$?" >> results/setup-transcribe.log
' >/dev/null 2>&1 &
echo "started (or already running)"
