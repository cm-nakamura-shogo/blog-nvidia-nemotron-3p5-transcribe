#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_DIR="${ASR_BENCHMARK_WORK_DIR:-${PROJECT_DIR}/work}"
VENV_DIR="${ASR_BENCHMARK_VENV_DIR:-${WORK_DIR}/.venv}"
NEMO_DIR="${ASR_BENCHMARK_NEMO_DIR:-${WORK_DIR}/deps/NeMo}"
NEMO_REVISION="e0a284b67db4da54f90a7873dd723aa2106a503e"

cd "${PROJECT_DIR}"

sudo apt-get update
sudo apt-get install -y \
  ffmpeg git jq libsndfile1 python3 python3-pip python3-venv sox libsox-fmt-mp3

mkdir -p "${WORK_DIR}" "${WORK_DIR}/deps" environment
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade \
  "pip==26.2.1" \
  "setuptools==81.0.0" \
  "wheel==0.48.0" \
  "Cython==3.2.9" \
  "packaging==24.2"
"${VENV_DIR}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu129 \
  "torch==2.11.0+cu129"
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

if [[ ! -d "${NEMO_DIR}/.git" ]]; then
  git init "${NEMO_DIR}"
  git -C "${NEMO_DIR}" remote add origin https://github.com/NVIDIA-NeMo/NeMo.git
fi
git -C "${NEMO_DIR}" fetch --depth 1 origin "${NEMO_REVISION}"
git -C "${NEMO_DIR}" checkout --detach FETCH_HEAD

"${VENV_DIR}/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  -e "${NEMO_DIR}[asr,cu12]"
"${VENV_DIR}/bin/python" -m pip check
"${VENV_DIR}/bin/python" -m pip freeze > environment/pip-freeze.txt
git -C "${NEMO_DIR}" rev-parse HEAD > environment/nemo-revision.txt

"${VENV_DIR}/bin/python" scripts/collect_environment.py \
  --output environment/environment-info.json
