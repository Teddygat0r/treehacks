#!/bin/bash
# Start the FastAPI frontend bridge server
cd "$(dirname "$0")/.."

export MODAL_APP_NAME="${MODAL_APP_NAME:-treehacks-verification-service}"
export MODAL_CLASS_NAME="${MODAL_CLASS_NAME:-VerificationService}"
export DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export BRIDGE_PORT="${BRIDGE_PORT:-8000}"

echo "Starting Frontend Bridge..."
echo "  Modal app: $MODAL_APP_NAME"
echo "  Modal class: $MODAL_CLASS_NAME"
echo "  Draft model: $DRAFT_MODEL"
echo "  Bridge port: $BRIDGE_PORT"

python -m uvicorn workers.frontend_bridge.server:app --host 0.0.0.0 --port "$BRIDGE_PORT" --reload
