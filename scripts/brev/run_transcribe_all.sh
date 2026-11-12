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

# 2. vLLMサーバー起動（コールドスタート時間はログに残る）
bash scripts/brev/start_transcribe_server_background.sh | tee results/nemotron-transcribe-cold-start.txt

# 3. スモーク → 精度（BS1）→ 性能（BS1〜64 × 3回）
"${PYTHON_BIN}" scripts/run_matrix.py --matrix configs/matrix-transcribe.yaml \
  --only nemotron-3.5-transcribe-fp8-vllm --batch-sizes 1 4 --repeats 1 --limit 10 \
  --output-dir results/smoke-transcribe --status-dir results/status-smoke-transcribe --resume
"${PYTHON_BIN}" scripts/run_matrix.py --matrix configs/matrix-transcribe.yaml \
  --only nemotron-3.5-transcribe-fp8-vllm --batch-sizes 1 --repeats 1 \
  --output-dir results/accuracy --status-dir results/status-accuracy --resume
"${PYTHON_BIN}" scripts/run_matrix.py --matrix configs/matrix-transcribe.yaml \
  --only nemotron-3.5-transcribe-fp8-vllm \
  --output-dir results/raw --status-dir results/status --resume

# 4. 翻訳・プロンプト制御
bash scripts/brev/08_run_probes.sh

# 5. Parakeetの再測定はGPUを空けてから行う
kill "$(cat results/nemotron-transcribe-server.pid)" || true
sleep 20
"${PYTHON_BIN}" scripts/run_matrix.py --matrix configs/matrix-transcribe.yaml \
  --only parakeet-tdt-ctc-0.6b-ja \
  --output-dir results/raw --status-dir results/status --resume
