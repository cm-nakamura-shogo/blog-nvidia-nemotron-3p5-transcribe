#!/usr/bin/env bash
# モデルカードのFP8版の起動コマンド（RTX PRO 6000向け）に、L40Sで動かすための最小限の変更を加えたもの。
# 変更点は「変更」とコメントした3か所だけで、それ以外はモデルカードのまま。
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPEECH_DIR="${ASR_BENCHMARK_NEMO_SPEECH_DIR:-${PROJECT_DIR}/work/nemo-speech}"

cd "${SPEECH_DIR}"
# activateスクリプトは未定義の変数を参照するため、その間だけset -uを外す
set +u
source .venv/bin/activate
set -u
# モデルカードの注記：FlashInferのJITコンパイルにはnvccがPATHに必要
export PATH=/usr/local/cuda/bin:$PATH
# 変更1：ドライバ565でCUDA 13を使うため、forward compatibilityのlibcudaを追加
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export VLLM_PLUGINS=nemo_speechlm
# Allow vLLM to decode recordings up to one hour long (the default is 600 seconds).
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
# Word timestamps require the V1 model runner.
export VLLM_USE_V2_MODEL_RUNNER=0
# Keep the optimized CUTLASS FP8 kernels validated on RTX PRO 6000.
export VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
# 変更2：--mamba-backendをflashinferからtritonに変更（FlashInferのReplaySSMカーネルがsm_90専用命令を含み、L40S（sm_89）でコンパイルできない）
# 変更3：--speculative-config（MTP）を削除（ReplaySSMの投機的デコードにはFlashInferが必須）
exec vllm serve ./NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8 \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8 \
  --trust-remote-code --dtype bfloat16 --kv-cache-dtype auto --gpu-memory-utilization 0.85 \
  --max-model-len 65536 --max-num-batched-tokens 4096 --max-num-seqs 64 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --use-replayssm --mamba-backend triton --mamba-cache-mode none \
  --mm-processor-cache-gb 0 --no-async-scheduling \
  --limit-mm-per-prompt '{"audio":{"count":1,"length":57600000}}' \
  --reasoning-parser nemotron_v3 \
  --trust-request-mm-kwargs \
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
