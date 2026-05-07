#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
REQ_FILE="${REQ_FILE:-${PROJECT_ROOT}/requirements.txt}"

if [[ "${1:-}" == "--minimal" ]]; then
  REQ_FILE="${PROJECT_ROOT}/requirements-minimal.txt"
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ_FILE}"
python -m pip install -e "${PROJECT_ROOT}"

python - <<'PY'
import sys
print("DOSSIER venv ready:", sys.executable)
PY
