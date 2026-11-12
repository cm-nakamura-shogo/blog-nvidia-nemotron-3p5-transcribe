#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_NAME="${ASR_BENCHMARK_BREV_WORKSPACE:-cm-nakamura-asr-benchmark}"
REMOTE_DIR="${ASR_BENCHMARK_REMOTE_DIR:-/home/ubuntu/workspace/asr-benchmark}"

brev exec "${WORKSPACE_NAME}" "cd ${REMOTE_DIR} && bash scripts/brev/run_all.sh"

