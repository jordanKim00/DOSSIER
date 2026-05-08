#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${DOSSIER_PYTHON:-/opt/conda/bin/python}"
VLLM_PYTHON="${DOSSIER_VLLM_PYTHON:-/opt/conda/bin/python}"
OUTPUT_DIR="${DOSSIER_OUTPUT_DIR:-${PROJECT_ROOT}/logs/runs/dossier_full}"
RUN_LOG="${DOSSIER_RUN_LOG:-${OUTPUT_DIR}/run.log}"
VLLM_LOG="${DOSSIER_VLLM_LOG:-${OUTPUT_DIR}/vllm.log}"

HOST="${DOSSIER_VLLM_HOST:-127.0.0.1}"
PORT="${DOSSIER_VLLM_PORT:-8000}"
MODEL_PATH="${DOSSIER_VLLM_MODEL_PATH:-${PROJECT_ROOT}/models/Llama-3.1-8B-Instruct}"
SERVED_MODEL_NAME="${DOSSIER_VLLM_SERVED_MODEL_NAME:-${MODEL_PATH}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${DOSSIER_VLLM_TP_SIZE:-4}"
MAX_MODEL_LEN="${DOSSIER_VLLM_MAX_MODEL_LEN:-260000}"
GPU_MEMORY_UTILIZATION="${DOSSIER_VLLM_GPU_MEMORY_UTILIZATION:-0.93}"
MAX_NUM_SEQS="${DOSSIER_VLLM_MAX_NUM_SEQS:-1}"
HEALTH_TIMEOUT="${DOSSIER_VLLM_HEALTH_TIMEOUT:-1800}"
# Optional reasoning parser (set to "qwen3" for Qwen3 family; leave empty for Llama)
REASONING_PARSER="${DOSSIER_VLLM_REASONING_PARSER:-}"
# Extra raw vLLM CLI args (e.g. --hf-overrides for rope_scaling override).
# For Llama-3.1-8B at >128K, default to a 2x linear extension.
EXTRA_ARGS="${DOSSIER_VLLM_EXTRA_ARGS:---hf-overrides {\"rope_scaling\":{\"rope_type\":\"linear\",\"factor\":2.0}}}"

SERVER_PID=""

cleanup() {
  local status=$?
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping vLLM server (pid=${SERVER_PID})" | tee -a "${RUN_LOG}"
    kill -TERM -"${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    sleep 5
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -KILL -"${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  exit "${status}"
}

wait_for_vllm() {
  "${PYTHON_BIN}" - <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

host = os.environ["DOSSIER_VLLM_HOST"]
port = os.environ["DOSSIER_VLLM_PORT"]
timeout_s = float(os.environ.get("DOSSIER_VLLM_HEALTH_TIMEOUT", "1800"))
server_pid = os.environ.get("DOSSIER_VLLM_SERVER_PID", "").strip()
url = f"http://{host}:{port}/health"
deadline = time.time() + timeout_s
last_error = ""

def server_exited(pid: str) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    try:
        state = open(f"/proc/{pid}/stat", encoding="utf-8").read().split()[2]
    except Exception:
        return False
    return state == "Z"

while time.time() < deadline:
    if server_pid and server_exited(server_pid):
        print(f"vLLM server exited before becoming ready: pid={server_pid}", file=sys.stderr)
        sys.exit(1)
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if 200 <= response.status < 300:
                print(f"vLLM server ready: {url}")
                sys.exit(0)
            last_error = f"status={response.status}"
    except urllib.error.HTTPError as exc:
        last_error = f"status={exc.code}"
    except Exception as exc:
        last_error = repr(exc)
    time.sleep(2)

print(f"vLLM server did not become ready within {timeout_s:.0f}s: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

mkdir -p "${OUTPUT_DIR}"
trap cleanup EXIT INT TERM

export DOSSIER_VLLM_HOST="${HOST}"
export DOSSIER_VLLM_PORT="${PORT}"
export DOSSIER_VLLM_HEALTH_TIMEOUT="${HEALTH_TIMEOUT}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_MODEL="${OPENAI_MODEL:-${SERVED_MODEL_NAME}}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://${HOST}:${PORT}/v1}"
export OPENAI_DEFAULT_TEMPERATURE="${OPENAI_DEFAULT_TEMPERATURE:-0}"
export OPENAI_THINKING_MODULES="${OPENAI_THINKING_MODULES:-}"
export OPENAI_TIMEOUT_SECONDS="${OPENAI_TIMEOUT_SECONDS:-3600}"

export DOSSIER_TOC_BUILDER_MAX_OUTPUT_TOKENS="${DOSSIER_TOC_BUILDER_MAX_OUTPUT_TOKENS:-150000}"
export DOSSIER_SEARCH_AGENT_MAX_OUTPUT_TOKENS="${DOSSIER_SEARCH_AGENT_MAX_OUTPUT_TOKENS:-125000}"
export DOSSIER_COMPOSER_MAX_OUTPUT_TOKENS="${DOSSIER_COMPOSER_MAX_OUTPUT_TOKENS:-50000}"
export DOSSIER_FORMATTER_MAX_OUTPUT_TOKENS="${DOSSIER_FORMATTER_MAX_OUTPUT_TOKENS:-8192}"
export DOSSIER_LLM_JUDGE_MAX_OUTPUT_TOKENS="${DOSSIER_LLM_JUDGE_MAX_OUTPUT_TOKENS:-400}"

# Optional CLI fragments — empty by default for Llama-family models.
REASONING_PARSER_ARG=""
if [[ -n "${REASONING_PARSER}" ]]; then
  REASONING_PARSER_ARG="--reasoning-parser ${REASONING_PARSER}"
fi

echo "Starting vLLM server on ${HOST}:${PORT}" | tee "${RUN_LOG}"
# shellcheck disable=SC2086
setsid env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  ${REASONING_PARSER_ARG} \
  ${EXTRA_ARGS} \
  > "${VLLM_LOG}" 2>&1 &
SERVER_PID=$!
export DOSSIER_VLLM_SERVER_PID="${SERVER_PID}"

if ! wait_for_vllm; then
  echo "vLLM failed to start. Last log lines:" | tee -a "${RUN_LOG}"
  tail -n 120 "${VLLM_LOG}" | tee -a "${RUN_LOG}" || true
  exit 1
fi

echo "Running DOSSIER full Loong experiment" | tee -a "${RUN_LOG}"
set +e
TQDM_DISABLE=1 "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_pipeline.py" \
  --backend openai \
  --output_dir "${OUTPUT_DIR}" \
  --force \
  "$@" 2>&1 | tee -a "${RUN_LOG}"
RUN_STATUS=${PIPESTATUS[0]}
set -e

exit "${RUN_STATUS}"
