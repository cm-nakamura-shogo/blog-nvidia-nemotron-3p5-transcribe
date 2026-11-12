#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${ASR_BENCHMARK_PYTHON_BIN:-${PROJECT_DIR}/work/.venv/bin/python}"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" scripts/collect_environment.py \
  --output environment/environment-info.json
"${PYTHON_BIN}" -m pip freeze > environment/pip-freeze.txt

NEMOTRON_TRANSFORMERS_PYTHON="${PROJECT_DIR}/work/.venv-nemotron-transformers/bin/python"
if [[ -x "${NEMOTRON_TRANSFORMERS_PYTHON}" ]]; then
  "${NEMOTRON_TRANSFORMERS_PYTHON}" scripts/collect_environment.py \
    --output environment/environment-info-nemotron-transformers.json
  "${NEMOTRON_TRANSFORMERS_PYTHON}" -m pip freeze \
    > environment/pip-freeze-nemotron-transformers.txt
fi

REAZONSPEECH_PYTHON="${PROJECT_DIR}/scripts/reazonspeech_python.sh"
if [[ -x "${PROJECT_DIR}/work/.venv-reazonspeech/bin/python" ]]; then
  "${REAZONSPEECH_PYTHON}" scripts/collect_environment.py \
    --output environment/environment-info-reazonspeech.json
  "${REAZONSPEECH_PYTHON}" -m pip freeze \
    > environment/pip-freeze-reazonspeech.txt
fi
