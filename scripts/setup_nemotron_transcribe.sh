#!/usr/bin/env bash
# モデルカード（FP8版）のインストール手順を、そのまま実行する。
# 公式手順からの変更は、ドライバ565向けのcuda-compat-13-0の導入だけ。
# 再現できるように、NeMo Speechのコミットとモデルのrevisionは実行時点のものに固定している。
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_DIR="${ASR_BENCHMARK_WORK_DIR:-${PROJECT_DIR}/work}"
SPEECH_DIR="${ASR_BENCHMARK_NEMO_SPEECH_DIR:-${WORK_DIR}/nemo-speech}"
SPEECH_REVISION="079e48770098a4cffcf542a4bc974cafc84502a3"
MODEL_REVISION="2b6209f89bec0da576d69183bfdabb3b6c05ade2"

# 前提条件：uv（モデルカードの手順には含まれない）
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# 変更1：ドライバ565（CUDA 12.7）では、vLLM 0.32.0が入れるtorch cu130が動かないため、
# データセンター向けGPUのforward compatibilityパッケージでCUDA 13のランタイムを使う。
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if (( DRIVER_MAJOR < 580 )); then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-compat-13-0
fi

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"
# ---- ここからモデルカードの手順（コミットとrevisionの固定だけを追加） ----
git clone https://github.com/NVIDIA-NeMo/Speech.git "${SPEECH_DIR}"
cd "${SPEECH_DIR}"
git checkout "${SPEECH_REVISION}"
uv venv --python 3.13 .venv
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --extra asr --extra vllm --locked --no-install-project
uv pip install --python .venv/bin/python --no-deps -e .
# PyAV decodes compressed audio formats, such as M4A, without a system FFmpeg.
uv pip install --python .venv/bin/python --no-deps av==19.0.2
# vLLM requires OpenCV, which NeMo's dependency settings exclude.
uv pip install --python .venv/bin/python --no-deps opencv-python-headless==5.0.0.93
# activateスクリプトは未定義の変数を参照するため、その間だけset -uを外す
set +u
source .venv/bin/activate
set -u
hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8 --revision "${MODEL_REVISION}" --local-dir ./NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8
# ---- ここまで ----

# NeMo Speechの.python-version（3.12）に合わせて、uv syncで3.12の環境に作り直される。
python --version
cd "${PROJECT_DIR}"
mkdir -p environment
uv pip freeze --python "${SPEECH_DIR}/.venv/bin/python" > environment/pip-freeze-nemotron-transcribe.txt
