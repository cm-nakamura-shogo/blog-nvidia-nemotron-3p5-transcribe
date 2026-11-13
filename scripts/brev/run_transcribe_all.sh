#!/usr/bin/env bash
# Nemotron 3.5 Transcribe検証をBrev（L40S）上で一括実行する。
# 前提: scripts/setup_brev.sh と scripts/setup_nemotron_transcribe.sh が完了していること。
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${ASR_BENCHMARK_PYTHON_BIN:-${PROJECT_DIR}/work/.venv/bin/python}"
cd "${PROJECT_DIR}"

# 1. データ準備（FLEURS ja_jp / en_us test）
for lang in ja_jp en_us; do
  if [[ ! -f "manifests/fleurs-${lang}-test.jsonl" ]]; then
    "${PYTHON_BIN}" scripts/prepare_fleurs.py --config "${lang}" \
      --audio-dir "work/fleurs-${lang}-test/audio" \
      --manifest "manifests/fleurs-${lang}-test.jsonl" \
      --metadata "manifests/fleurs-${lang}-test.metadata.json"
  fi
done

# 2. vLLMサーバー起動
bash scripts/brev/start_transcribe_server_background.sh

# 3. 認識精度（1件ずつ、650件）
"${PYTHON_BIN}" scripts/run_matrix.py --matrix configs/matrix-transcribe.yaml \
  --only nemotron-3.5-transcribe-fp8-vllm --batch-sizes 1 --repeats 1 \
  --output-dir results/accuracy --status-dir results/status-accuracy --resume

# 4. プロンプトを変えた実行と採点（1件ずつ）
bash scripts/brev/08_run_probes.sh
