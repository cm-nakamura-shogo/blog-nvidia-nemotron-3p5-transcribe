#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${ASR_BENCHMARK_REAZONSPEECH_VENV:-${PROJECT_DIR}/work/.venv-reazonspeech}"
SHERPA_ONNX_VERSION="1.13.4+cuda12.cudnn9"

cd "${PROJECT_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade \
  "pip==26.2.1" \
  "setuptools==81.0.0" \
  "wheel==0.48.0"
"${VENV_DIR}/bin/python" -m pip install -r requirements-reazonspeech.txt
"${VENV_DIR}/bin/python" -m pip install \
  --no-index \
  --find-links https://k2-fsa.github.io/sherpa/onnx/cuda.html \
  "sherpa-onnx==${SHERPA_ONNX_VERSION}"
"${VENV_DIR}/bin/python" -m pip check
scripts/reazonspeech_python.sh -c \
  'import sherpa_onnx; print("sherpa-onnx:", sherpa_onnx.__version__)'
"${VENV_DIR}/bin/python" -m pip freeze > environment/pip-freeze-reazonspeech.txt
scripts/reazonspeech_python.sh scripts/collect_environment.py \
  --output environment/environment-info-reazonspeech.json
