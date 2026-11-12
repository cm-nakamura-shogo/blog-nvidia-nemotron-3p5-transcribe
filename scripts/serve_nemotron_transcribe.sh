#!/usr/bin/env bash
# モデルカードのFP8起動コマンド（RTX PRO 6000 96GB向け）をL40S 48GB向けに調整したもの。
# 変更点: 1リクエストの最大音声長を短くし（既定600秒）、エンコーダのプロファイル時メモリを抑える。
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_DIR="${ASR_BENCHMARK_WORK_DIR:-${PROJECT_DIR}/work}"
SPEECH_DIR="${ASR_BENCHMARK_NEMO_SPEECH_DIR:-${WORK_DIR}/deps/nemo-speech}"
MODEL_DIR="${WORK_DIR}/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8"
MAX_AUDIO_SEC="${NEMOTRON_MAX_AUDIO_SEC:-600}"
GPU_MEMORY_UTILIZATION="${NEMOTRON_GPU_MEMORY_UTILIZATION:-0.90}"
# L40S（sm_89）ではFlashInferのReplaySSMカーネル（checkpointing_ssu）がsm_90専用命令
# （griddepcontrol）をガードなしで含み、MTPの有無にかかわらずコンパイルできない。
# そのためMambaバックエンドをTritonにし、FlashInferが必須のMTPも無効にする。
SPECULATIVE="${NEMOTRON_SPECULATIVE:-none}"
MAMBA_BACKEND="${NEMOTRON_MAMBA_BACKEND:-triton}"

# ドライバ565のL40Sではforward compatibilityのlibcudaでCUDA 13ランタイムを使う（setupを参照）。
# FlashInferのJITはvenv内のninjaを使うため、venvのbinもPATHに入れる。
export PATH="${SPEECH_DIR}/.venv/bin:/usr/local/cuda-13.0/bin:${PATH}"
if [[ -d /usr/local/cuda-13.0/compat ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export FLASHINFER_WORKSPACE_BASE="${WORK_DIR}/flashinfer"
export VLLM_PLUGINS=nemo_speechlm
export VLLM_MAX_AUDIO_DECODE_DURATION_S="${MAX_AUDIO_SEC}"
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
export VLLM_USE_V2_MODEL_RUNNER=0
mkdir -p "${FLASHINFER_WORKSPACE_BASE}"

SPEC_ARGS=()
if [[ "${SPECULATIVE}" == "mtp8" ]]; then
  SPEC_ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":8}')
fi

exec "${SPEECH_DIR}/.venv/bin/vllm" serve "${MODEL_DIR}" \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8 \
  --host 127.0.0.1 --port 8000 \
  --trust-remote-code --dtype bfloat16 --kv-cache-dtype auto \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len 65536 --max-num-batched-tokens 4096 --max-num-seqs 64 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --use-replayssm --mamba-backend "${MAMBA_BACKEND}" --mamba-cache-mode none \
  --mm-processor-cache-gb 0 --no-async-scheduling \
  --limit-mm-per-prompt "{\"audio\":{\"count\":1,\"length\":$((MAX_AUDIO_SEC * 16000))}}" \
  "${SPEC_ARGS[@]}" \
  --reasoning-parser nemotron_v3 \
  --trust-request-mm-kwargs \
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
