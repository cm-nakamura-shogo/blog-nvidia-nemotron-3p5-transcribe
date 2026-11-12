#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_NAME="${ASR_BENCHMARK_BREV_WORKSPACE:-cm-nakamura-asr-benchmark}"
REMOTE_DIR="${ASR_BENCHMARK_REMOTE_DIR:-/home/ubuntu/workspace/asr-benchmark}"

brev exec "${WORKSPACE_NAME}" "mkdir -p ${REMOTE_DIR}"
brev copy "${PROJECT_DIR}/" "${WORKSPACE_NAME}:${REMOTE_DIR}/"

