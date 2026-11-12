#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"

bash scripts/brev/00_preflight.sh
bash scripts/brev/01_setup.sh
bash scripts/brev/02_prepare_fleurs.sh
bash scripts/brev/03_smoke_test.sh
bash scripts/brev/04_run_accuracy.sh
bash scripts/brev/05_run_performance.sh
bash scripts/brev/06_aggregate.sh \
  --accuracy-dir results/accuracy \
  --accuracy-status-dir results/status-accuracy \
  --performance-dir results/raw \
  --performance-status-dir results/status
bash scripts/brev/07_collect_environment.sh
