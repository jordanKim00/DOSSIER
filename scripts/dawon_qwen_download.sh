#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${DAWON_VENV_DIR:-${PROJECT_ROOT}/../venvs/dossiervenv}"
PYTHON_BIN="${DAWON_PYTHON:-${VENV_DIR}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${DAWON_DOWNLOAD_PYTHON:-python3}"
fi

MODEL_ID="${DAWON_QWEN_MODEL_ID:-Qwen/Qwen3.5-27B}"
MODEL_DIR="${DAWON_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3.5-27B}"
REVISION="${DAWON_QWEN_REVISION:-main}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import huggingface_hub
PY
then
  "${PYTHON_BIN}" -m pip install --upgrade "huggingface-hub[hf_transfer]"
fi

mkdir -p "${MODEL_DIR}"

"${PYTHON_BIN}" - "${MODEL_ID}" "${MODEL_DIR}" "${REVISION}" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id, model_dir, revision = sys.argv[1:4]
print(f"Downloading {model_id}@{revision} -> {model_dir}", flush=True)
snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_dir=model_dir,
    resume_download=True,
)
print("Download complete.", flush=True)
PY

echo "Model ready: ${MODEL_DIR}"
