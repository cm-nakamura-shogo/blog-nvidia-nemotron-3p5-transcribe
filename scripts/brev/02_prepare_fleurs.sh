#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${ASR_BENCHMARK_PYTHON_BIN:-${PROJECT_DIR}/work/.venv/bin/python}"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" scripts/prepare_fleurs.py \
  --audio-dir work/fleurs-ja_jp-test/audio \
  --manifest manifests/fleurs-ja_jp-test.jsonl \
  --metadata manifests/fleurs-ja_jp-test.metadata.json

"${PYTHON_BIN}" scripts/validate_manifest.py \
  --manifest manifests/fleurs-ja_jp-test.jsonl \
  --output manifests/fleurs-ja_jp-test.validation.json \
  --expected-files 650 \
  --expected-sample-rate 16000
