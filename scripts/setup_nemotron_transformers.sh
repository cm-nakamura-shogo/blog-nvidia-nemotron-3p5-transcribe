#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${ASR_BENCHMARK_NEMOTRON_TRANSFORMERS_VENV:-${PROJECT_DIR}/work/.venv-nemotron-transformers}"

cd "${PROJECT_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade \
  "pip==26.2.1" \
  "setuptools==81.0.0" \
  "wheel==0.48.0"
"${VENV_DIR}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu129 \
  "torch==2.11.0+cu129"
"${VENV_DIR}/bin/python" -m pip install -r requirements-nemotron-transformers.txt
"${VENV_DIR}/bin/python" -m pip check
"${VENV_DIR}/bin/python" -m pip freeze > environment/pip-freeze-nemotron-transformers.txt

"${VENV_DIR}/bin/python" -c \
  'from transformers import AutoModelForRNNT; print("AutoModelForRNNT import: OK")'
"${VENV_DIR}/bin/python" scripts/collect_environment.py \
  --output environment/environment-info-nemotron-transformers.json
