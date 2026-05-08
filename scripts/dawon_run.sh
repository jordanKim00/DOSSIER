#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${DAWON_VENV_DIR:-${PROJECT_ROOT}/.venv-dawon}"
PYTHON_BIN="${DAWON_PYTHON:-${VENV_DIR}/bin/python}"
REQ_FILE="${DAWON_REQ_FILE:-${PROJECT_ROOT}/dawon_req.txt}"

INPUT_PATH="${DAWON_INPUT_PATH:-${PROJECT_ROOT}/Loong/full/loong_process.jsonl}"
OUTPUT_DIR="${DAWON_OUTPUT_DIR:-${PROJECT_ROOT}/logs/runs/dawon_qwen35_27b_full}"
RUN_LOG="${DAWON_RUN_LOG:-${OUTPUT_DIR}/run.log}"
VLLM_LOG="${DAWON_VLLM_LOG:-${OUTPUT_DIR}/vllm.log}"

HOST="${DAWON_VLLM_HOST:-127.0.0.1}"
PORT="${DAWON_VLLM_PORT:-8000}"
MODEL_PATH="${DAWON_MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3.5-27B}"
SERVED_MODEL_NAME="${DAWON_SERVED_MODEL_NAME:-Qwen3.5-27B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TENSOR_PARALLEL_SIZE="${DAWON_VLLM_TP_SIZE:-}"
MAX_MODEL_LEN="${DAWON_MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${DAWON_GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${DAWON_MAX_NUM_SEQS:-1}"
DTYPE="${DAWON_DTYPE:-bfloat16}"
HEALTH_TIMEOUT="${DAWON_VLLM_HEALTH_TIMEOUT:-2400}"
INSTALL_TIMEOUT_SECONDS="${DAWON_INSTALL_TIMEOUT_SECONDS:-7200}"

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

visible_gpu_count() {
  local devices="${CUDA_VISIBLE_DEVICES}"
  if [[ -z "${devices}" || "${devices}" == "all" ]]; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '
    return
  fi
  awk -F',' '{print NF}' <<<"${devices}"
}

ensure_venv() {
  if [[ "${DAWON_SKIP_INSTALL:-0}" == "1" ]]; then
    return
  fi
  local marker="${VENV_DIR}/.dawon_req_installed"
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi
  if [[ "${DAWON_REINSTALL:-0}" != "1" && -f "${marker}" && "${marker}" -nt "${REQ_FILE}" ]]; then
    return
  fi
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  timeout "${INSTALL_TIMEOUT_SECONDS}" "${PYTHON_BIN}" -m pip install -r "${REQ_FILE}"
  "${PYTHON_BIN}" -m pip install -e "${PROJECT_ROOT}"
  touch "${marker}"
}

vllm_supports_arg() {
  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q -- "$1"
}

wait_for_vllm() {
  "${PYTHON_BIN}" - <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

host = os.environ["DAWON_VLLM_HOST"]
port = os.environ["DAWON_VLLM_PORT"]
timeout_s = float(os.environ.get("DAWON_VLLM_HEALTH_TIMEOUT", "2400"))
server_pid = os.environ.get("DAWON_VLLM_SERVER_PID", "").strip()
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

if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "Input file not found: ${INPUT_PATH}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory not found: ${MODEL_PATH}" >&2
  echo "Run: bash scripts/dawon_qwen_download.sh" >&2
  echo "Or set DAWON_MODEL_PATH=/path/to/Qwen3.5-27B" >&2
  exit 1
fi

if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
  TENSOR_PARALLEL_SIZE="$(visible_gpu_count)"
fi

mkdir -p "${OUTPUT_DIR}"
trap cleanup EXIT INT TERM

ensure_venv

export DAWON_VLLM_HOST="${HOST}"
export DAWON_VLLM_PORT="${PORT}"
export DAWON_VLLM_HEALTH_TIMEOUT="${HEALTH_TIMEOUT}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://${HOST}:${PORT}/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-${SERVED_MODEL_NAME}}"
export OPENAI_DEFAULT_TEMPERATURE="${OPENAI_DEFAULT_TEMPERATURE:-0}"
export OPENAI_THINKING_MODULES="${OPENAI_THINKING_MODULES:-}"
export OPENAI_TIMEOUT_SECONDS="${OPENAI_TIMEOUT_SECONDS:-3600}"

export DOSSIER_TOC_BUILDER_MAX_OUTPUT_TOKENS="${DOSSIER_TOC_BUILDER_MAX_OUTPUT_TOKENS:-16384}"
export DOSSIER_SEARCH_AGENT_MAX_OUTPUT_TOKENS="${DOSSIER_SEARCH_AGENT_MAX_OUTPUT_TOKENS:-12288}"
export DOSSIER_COMPOSER_MAX_OUTPUT_TOKENS="${DOSSIER_COMPOSER_MAX_OUTPUT_TOKENS:-8192}"
export DOSSIER_FORMATTER_MAX_OUTPUT_TOKENS="${DOSSIER_FORMATTER_MAX_OUTPUT_TOKENS:-8192}"
export DOSSIER_LLM_JUDGE_MAX_OUTPUT_TOKENS="${DOSSIER_LLM_JUDGE_MAX_OUTPUT_TOKENS:-400}"

VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --dtype "${DTYPE}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
  --enforce-eager
  --reasoning-parser qwen3
  --generation-config vllm
)

if [[ "${DAWON_LANGUAGE_MODEL_ONLY:-1}" == "1" ]] && vllm_supports_arg "--language-model-only"; then
  VLLM_ARGS+=(--language-model-only)
fi

if [[ "${DAWON_DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]] && vllm_supports_arg "--disable-custom-all-reduce"; then
  VLLM_ARGS+=(--disable-custom-all-reduce)
fi

echo "Starting vLLM: model=${MODEL_PATH}, gpus=${CUDA_VISIBLE_DEVICES}, tp=${TENSOR_PARALLEL_SIZE}, max_len=${MAX_MODEL_LEN}" | tee "${RUN_LOG}"
setsid env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" "${VLLM_ARGS[@]}" \
  > "${VLLM_LOG}" 2>&1 &
SERVER_PID=$!
export DAWON_VLLM_SERVER_PID="${SERVER_PID}"

if ! wait_for_vllm; then
  echo "vLLM failed to start. Last log lines:" | tee -a "${RUN_LOG}"
  tail -n 160 "${VLLM_LOG}" | tee -a "${RUN_LOG}" || true
  exit 1
fi

PIPELINE_ARGS=(
  --backend openai
  --input_path "${INPUT_PATH}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${DAWON_FORCE:-0}" == "1" ]]; then
  PIPELINE_ARGS+=(--force)
fi

if [[ "${DAWON_LIMIT:-}" =~ ^[0-9]+$ ]] && [[ "${DAWON_LIMIT}" -gt 0 ]]; then
  PIPELINE_ARGS+=(--limit "${DAWON_LIMIT}")
fi

if [[ -n "${DAWON_MAX_REFINE_ROUNDS:-}" ]]; then
  PIPELINE_ARGS+=(--max_refine_rounds "${DAWON_MAX_REFINE_ROUNDS}")
fi

echo "Running DOSSIER on ${INPUT_PATH}" | tee -a "${RUN_LOG}"
set +e
TQDM_DISABLE=1 "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_pipeline.py" \
  "${PIPELINE_ARGS[@]}" \
  "$@" 2>&1 | tee -a "${RUN_LOG}"
RUN_STATUS=${PIPESTATUS[0]}
set -e

exit "${RUN_STATUS}"
