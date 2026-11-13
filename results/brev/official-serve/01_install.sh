#!/usr/bin/env bash
# 前提条件: uv（モデルカードの手順には含まれない）
set -x
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# ---- ここからモデルカードの手順そのまま ----
cd ~/official
git clone https://github.com/NVIDIA-NeMo/Speech.git nemo-speech
cd nemo-speech
uv venv --python 3.13 .venv
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --extra asr --extra vllm --locked --no-install-project
uv pip install --python .venv/bin/python --no-deps -e .
# PyAV decodes compressed audio formats, such as M4A, without a system FFmpeg.
uv pip install --python .venv/bin/python --no-deps av==19.0.2
# vLLM requires OpenCV, which NeMo s dependency settings exclude.
uv pip install --python .venv/bin/python --no-deps opencv-python-headless==5.0.0.93
source .venv/bin/activate
hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8 --local-dir ./NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8
# ---- ここまで ----

git rev-parse HEAD
python --version
python -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, \"cuda\", torch.version.cuda)"
