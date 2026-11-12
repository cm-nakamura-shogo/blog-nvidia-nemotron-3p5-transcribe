#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_NAME="${ASR_BENCHMARK_BREV_WORKSPACE:-cm-nakamura-asr-benchmark}"
REMOTE_DIR="${ASR_BENCHMARK_REMOTE_DIR:-/home/ubuntu/workspace/asr-benchmark}"

brev copy "${WORKSPACE_NAME}:${REMOTE_DIR}/results/" "${PROJECT_DIR}/results/"
brev copy "${WORKSPACE_NAME}:${REMOTE_DIR}/environment/" "${PROJECT_DIR}/environment/"
brev copy \
  "${WORKSPACE_NAME}:${REMOTE_DIR}/manifests/fleurs-ja_jp-test.metadata.json" \
  "${PROJECT_DIR}/manifests/"
brev copy \
  "${WORKSPACE_NAME}:${REMOTE_DIR}/manifests/fleurs-ja_jp-test.validation.json" \
  "${PROJECT_DIR}/manifests/"
