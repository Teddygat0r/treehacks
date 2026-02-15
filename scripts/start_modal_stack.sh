#!/usr/bin/env bash
set -euo pipefail

# Simple startup script for the distributed speculative decoding stack
# Architecture: Frontend -> Frontend Bridge -> Router -> (Modal Target + Draft Nodes)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/workers/logs"

# Configuration
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-8001}"
DRAFT_PORT="${DRAFT_PORT:-50052}"
BRIDGE_PORT="${BRIDGE_PORT:-8000}"
MODAL_APP_NAME="${MODAL_APP_NAME:-treehacks-verification-service}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

mkdir -p "${LOG_DIR}"

# Activate venv if available
if [[ -f "${HOME}/.venv-vllm-metal/bin/activate" ]]; then
  source "${HOME}/.venv-vllm-metal/bin/activate"
fi

# Track background processes
PIDS=()
cleanup() {
  echo "Cleaning up background processes..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

# Helper: wait for HTTP endpoint
wait_for_http() {
  local url="$1"
  local name="$2"
  local attempts="${3:-40}"

  for _ in $(seq 1 "${attempts}"); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "✓ ${name} is ready"
      return 0
    fi
    sleep 0.5
  done

  echo "✗ Timeout waiting for ${name} at ${url}"
  return 1
}

# =====================================================================
# Step 1: Clean up old Modal deployment
# =====================================================================
echo "Step 1/5: Stopping old modal deployment"
modal app stop "${MODAL_APP_NAME}" 2>/dev/null || echo "  (no existing deployment found)"

# =====================================================================
# Step 2: Start Router
# =====================================================================
echo "Step 2/5: Starting router (port ${ROUTER_PORT})"
(
  cd "${ROOT_DIR}"
  python router/server.py --host "${ROUTER_HOST}" --port "${ROUTER_PORT}"
) >"${LOG_DIR}/router.log" 2>&1 &
PIDS+=("$!")
wait_for_http "http://127.0.0.1:${ROUTER_PORT}/health" "Router"

# =====================================================================
# Step 3: Start Draft Node
# =====================================================================
echo "Step 3/5: Starting draft node (port ${DRAFT_PORT})"
(
  cd "${ROOT_DIR}"
  python workers/draft_node/client.py \
    --port "${DRAFT_PORT}" \
    --draft-model "${DRAFT_MODEL}" \
    --modal-app-name "${MODAL_APP_NAME}" \
    --modal-class-name "VerificationService"
) >"${LOG_DIR}/draft-node.log" 2>&1 &
PIDS+=("$!")
sleep 2  # Draft node takes a moment to initialize vLLM

# =====================================================================
# Step 4: Deploy Modal Target Server
# =====================================================================
echo "Step 4/5: Deploying modal target server"
(
  cd "${ROOT_DIR}"
  modal deploy workers/target_node/server_modal.py
) | tee "${LOG_DIR}/modal-deploy.log"

echo "  Note: Draft nodes will connect to Modal directly (no registration needed)"
sleep 2  # Give Modal time to deploy

# =====================================================================
# Step 5: Start Frontend Bridge
# =====================================================================
echo "Step 5/5: Starting frontend bridge (port ${BRIDGE_PORT})"
(
  cd "${ROOT_DIR}"
  DRAFT_MODEL_ID="${DRAFT_MODEL}" \
  TARGET_MODEL_ID="${TARGET_MODEL}" \
    python workers/frontend_bridge/server.py --port "${BRIDGE_PORT}"
) >"${LOG_DIR}/frontend-bridge.log" 2>&1 &
PIDS+=("$!")
wait_for_http "http://127.0.0.1:${BRIDGE_PORT}/api/health" "Frontend Bridge"

# =====================================================================
# All services ready
# =====================================================================
echo ""
echo "=========================================="
echo "All backend services are running!"
echo "=========================================="
echo "Logs:"
echo "  Router:          ${LOG_DIR}/router.log"
echo "  Draft Node:      ${LOG_DIR}/draft-node.log"
echo "  Modal Deploy:    ${LOG_DIR}/modal-deploy.log"
echo "  Frontend Bridge: ${LOG_DIR}/frontend-bridge.log"
echo ""
echo "Endpoints:"
echo "  Router:          http://127.0.0.1:${ROUTER_PORT}/state"
echo "  Frontend Bridge: http://127.0.0.1:${BRIDGE_PORT}/api/health"
echo ""
echo "Press Ctrl+C to stop all services."
echo ""

# Start frontend (this blocks)
cd "${ROOT_DIR}"
npm run dev
