#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${DOSSIER_PYTHON:-/opt/conda/bin/python}"
OUTPUT_DIR="${DOSSIER_OUTPUT_DIR:-${PROJECT_ROOT}/logs/runs/dossier_full}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen3.5-27B}"
export OPENAI_DEFAULT_TEMPERATURE="${OPENAI_DEFAULT_TEMPERATURE:-0}"
export OPENAI_THINKING_MODULES="${OPENAI_THINKING_MODULES:-}"
export OPENAI_TIMEOUT_SECONDS="${OPENAI_TIMEOUT_SECONDS:-3600}"

TQDM_DISABLE=1 "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_pipeline.py" \
  --backend openai \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
