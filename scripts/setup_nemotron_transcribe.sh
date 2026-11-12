#!/usr/bin/env bash
# モデルカードの手順（vLLM 0.32.0 + NeMo Speech main）を、コミット固定で再現する。
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_DIR="${ASR_BENCHMARK_WORK_DIR:-${PROJECT_DIR}/work}"
SPEECH_DIR="${ASR_BENCHMARK_NEMO_SPEECH_DIR:-${WORK_DIR}/deps/nemo-speech}"
SPEECH_REVISION="e5d79d5a49cbec49d79440b0070aac0c32a8dbb8"
MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8"
MODEL_REVISION="2b6209f89bec0da576d69183bfdabb3b6c05ade2"
MODEL_DIR="${WORK_DIR}/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8"

cd "${PROJECT_DIR}"
mkdir -p "${WORK_DIR}/deps" "${WORK_DIR}/models" environment

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# vLLM 0.32.0はtorch cu130を使い、FlashInferのFP8 kernel JITにはCUDA 13のnvccが必要（モデルカード記載）。
# Brevのl40s-48gb.1xはドライバ565（CUDA 12.7）なので、データセンターGPU向けの
# forward compatibilityパッケージでCUDA 13ランタイムを使う。
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if (( DRIVER_MAJOR < 580 )); then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cuda-compat-13-0 cuda-minimal-build-13-0 cuda-libraries-dev-13-0
fi
export PATH="/usr/local/cuda-13.0/bin:${PATH}"
if ! nvcc --version | grep -q "release 13"; then
  echo "WARNING: CUDA 13 nvcc not found on PATH; FlashInfer JIT may fail." >&2
  nvcc --version >&2 || true
fi

if [[ ! -d "${SPEECH_DIR}/.git" ]]; then
  git init "${SPEECH_DIR}"
  git -C "${SPEECH_DIR}" remote add origin https://github.com/NVIDIA-NeMo/Speech.git
fi
git -C "${SPEECH_DIR}" fetch --depth 1 origin "${SPEECH_REVISION}"
git -C "${SPEECH_DIR}" checkout --detach FETCH_HEAD

cd "${SPEECH_DIR}"
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.13 .venv
fi
UV_PROJECT_ENVIRONMENT="${PWD}/.venv" uv sync --extra asr --extra vllm --locked --no-install-project
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python --no-deps av==19.0.2
uv pip install --python .venv/bin/python --no-deps opencv-python-headless==5.0.0.93

.venv/bin/hf download "${MODEL_ID}" --revision "${MODEL_REVISION}" --local-dir "${MODEL_DIR}"

cd "${PROJECT_DIR}"
uv pip freeze --python "${SPEECH_DIR}/.venv/bin/python" > environment/pip-freeze-nemotron-transcribe.txt
git -C "${SPEECH_DIR}" rev-parse HEAD > environment/nemo-speech-revision.txt
"${SPEECH_DIR}/.venv/bin/python" -c 'import vllm; print("vllm", vllm.__version__)'
