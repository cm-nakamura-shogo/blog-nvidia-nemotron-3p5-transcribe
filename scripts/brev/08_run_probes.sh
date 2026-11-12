#!/usr/bin/env bash
# 翻訳・プロンプト制御のプローブを実行して採点する。vLLMサーバーは起動済みであること。
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${ASR_BENCHMARK_PYTHON_BIN:-${PROJECT_DIR}/work/.venv/bin/python}"
OUT=results/probes
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" scripts/run_prompt_probe.py \
  --config configs/nemotron-3.5-transcribe-fp8-vllm.yaml \
  --probes configs/prompt-probes.yaml \
  --output-dir "${OUT}" "$@"

for name in terms_verbatim terms_context terms_verbatim_context; do
  "${PYTHON_BIN}" scripts/evaluate_term_recognition.py \
    --sentences data/discovery/discovery_sentences.csv \
    --predictions "${OUT}/${name}.jsonl" \
    --details "${OUT}/${name}-term-details.csv" \
    --summary "${OUT}/${name}-term-summary.json" >/dev/null
  echo "${name}: $(jq -c '{strict_recall, control_false_insertion_rate, target_other_term_insertion_rate}' "${OUT}/${name}-term-summary.json")"
done
"${PYTHON_BIN}" scripts/analyze_format_probes.py \
  "${OUT}"/fleurs_numbers_written.jsonl \
  "${OUT}"/fleurs_numbers_spoken.jsonl \
  "${OUT}"/fleurs_no_punctuation.jsonl \
  --output "${OUT}/format-summary.json"
"${PYTHON_BIN}" scripts/score_translation.py \
  --predictions "${OUT}/st_ja_to_en.jsonl" \
  --reference-manifest manifests/fleurs-en_us-test.jsonl \
  --target-lang en --output "${OUT}/st_ja_to_en-score.json"
"${PYTHON_BIN}" scripts/score_translation.py \
  --predictions "${OUT}/st_en_to_ja.jsonl" \
  --reference-manifest manifests/fleurs-ja_jp-test.jsonl \
  --target-lang ja --output "${OUT}/st_en_to_ja-score.json"
