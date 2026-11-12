#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
bash scripts/setup_brev.sh
bash scripts/setup_nemotron_transformers.sh
bash scripts/setup_reazonspeech.sh
