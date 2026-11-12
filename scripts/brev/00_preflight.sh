#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${ASR_BENCHMARK_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"

test "$(nvidia-smi --query-gpu=name --format=csv,noheader --id=0 | xargs)" = "NVIDIA L40S"
nvidia-smi
df -h .
uname -a
python3 --version

