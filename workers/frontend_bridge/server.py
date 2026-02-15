"""
Frontend Bridge - FastAPI server bridging Next.js frontend to distributed nodes.

- Frontend asks router for a draft node
- Bridge calls that draft node over gRPC
- Draft node handles speculative decoding against routed target nodes
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time
import uuid

import grpc
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROTO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "proto",
)
if PROTO_DIR not in sys.path:
    sys.path.insert(0, PROTO_DIR)

import common_pb2
import speculative_decoding_pb2
import speculative_decoding_pb2_grpc


# ── Configuration ──

# Hard-coded router IP as requested.
ROUTER_HTTP_BASE = "http://127.0.0.1:8001"

# Keep bridge model routing aligned with worker launch env names.
DRAFT_MODEL_ID = (
    os.getenv("DRAFT_MODEL_ID")
    or os.getenv("DRAFT_MODEL")
    or "Qwen/Qwen2.5-1.5B-Instruct"
)
TARGET_MODEL_ID = (
    os.getenv("TARGET_MODEL_ID")
    or os.getenv("TARGET_MODEL")
    or "Qwen/Qwen2.5-3B-Instruct"
)
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8000"))
MOCK_MODE = os.getenv("MOCK_MODE", "").lower() in ("1", "true", "yes")
TOKEN_SEND_BATCH_SIZE = max(1, int(os.getenv("TOKEN_SEND_BATCH_SIZE", "1")))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")
PROMPT_FORMAT = os.getenv("PROMPT_FORMAT", "chatml").lower()

FRONTEND_CLIENT_ID = os.getenv("FRONTEND_CLIENT_ID", f"frontend-{uuid.uuid4().hex[:8]}")
FRONTEND_CLIENT_ADDRESS = os.getenv("FRONTEND_CLIENT_ADDRESS", f"127.0.0.1:{BRIDGE_PORT}")

app = FastAPI(title="SpecNet Frontend Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ──

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=-1)
    draft_tokens: int = Field(default=5, ge=1, le=20)


class TokenEvent(BaseModel):
    text: str
    type: str  # "accepted" | "rejected" | "corrected"
    token_id: int = 0
    logprob: float = 0.0


class RoundEvent(BaseModel):
    round_num: int
    drafted: int
    accepted: int
    corrected: int
    verification_time_ms: float
    acceptance_rate: float


class InferenceResponse(BaseModel):
    request_id: str
    generated_text: str
    tokens: list[TokenEvent]
    total_tokens: int
    draft_tokens_generated: int
    draft_tokens_accepted: int
    generation_time_ms: float
    acceptance_rate: float
    speculation_rounds: int


class NodeInfo(BaseModel):
    id: str
    type: str  # "draft" | "target"
    hardware: str
    model: str
    status: str  # "online" | "offline" | "busy"
    latency: float = 0.0
    price: float = 0.0
    gpu_memory: str = ""


class NetworkStats(BaseModel):
    active_draft_nodes: int
    active_target_nodes: int
    total_tps: float
    avg_acceptance_rate: float
    avg_cost_per_1k: float


_registered_with_router = False
_draft_channels: dict[str, grpc.Channel] = {}
_draft_stubs: dict[str, speculative_decoding_pb2_grpc.DraftNodeServiceStub] = {}


def _router_post(path: str, payload: dict) -> dict | None:
    try:
        response = requests.post(f"{ROUTER_HTTP_BASE}{path}", json=payload, timeout=2.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _router_get(path: str) -> dict | None:
    try:
        response = requests.get(f"{ROUTER_HTTP_BASE}{path}", timeout=2.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _register_frontend_client() -> None:
    global _registered_with_router
    if _registered_with_router:
        return

    response = _router_post(
        "/register/client",
        {
            "client_id": FRONTEND_CLIENT_ID,
            "address": FRONTEND_CLIENT_ADDRESS,
            "metadata": "frontend-bridge",
        },
    )
    _registered_with_router = bool(response and response.get("accepted"))


def _route_to_draft_node(request_id: str, prompt: str) -> dict:
    _register_frontend_client()

    response = _router_post(
        "/route/draft-node",
        {
            "request_id": request_id,
            "prompt": prompt,
            "model_id": DRAFT_MODEL_ID,
            "priority": 0,
            "client_id": FRONTEND_CLIENT_ID,
            "client_address": FRONTEND_CLIENT_ADDRESS,
        },
    )
    if not response or response.get("status") != "success":
        raise RuntimeError("Router failed to assign a draft node")
    if not response.get("assigned_draft_node_address"):
        raise RuntimeError("Router returned draft node without address")
    return response


def _get_draft_stub(address: str) -> speculative_decoding_pb2_grpc.DraftNodeServiceStub:
    if address not in _draft_stubs:
        channel = grpc.insecure_channel(address)
        _draft_channels[address] = channel
        _draft_stubs[address] = speculative_decoding_pb2_grpc.DraftNodeServiceStub(channel)
    return _draft_stubs[address]


def _build_model_prompt(user_prompt: str) -> str:
    system_prompt = SYSTEM_PROMPT.strip()
    if not system_prompt:
        return user_prompt

    if PROMPT_FORMAT == "plain":
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

    return (
        f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── Mock inference (no GPU required) ──

MOCK_RESPONSES = {
    "default": (
        "The concept you're asking about is quite fascinating. It involves multiple layers of "
        "understanding that have been refined over decades of research. At its core, the idea "
        "relies on fundamental principles that connect seemingly disparate observations into a "
        "unified framework. Researchers have spent considerable effort developing mathematical "
        "models that capture these relationships with remarkable precision."
    ),
    "relativity": (
        "The theory of relativity, proposed by Albert Einstein in 1905 and 1915, fundamentally "
        "revolutionized our understanding of space and time. Special relativity demonstrates that "
        "the speed of light is constant for all observers, leading to the iconic equation E=mc². "
        "General relativity extends this by describing gravity not as a force, but as a curvature "
        "of spacetime caused by massive objects."
    ),
    "ai": (
        "Artificial intelligence has evolved dramatically since its inception in the 1950s. Modern "
        "deep learning approaches use neural networks with billions of parameters trained on vast "
        "datasets. Techniques like speculative decoding accelerate inference by using a smaller "
        "draft model to predict tokens that a larger target model then verifies, achieving "
        "significant speedups while maintaining output quality."
    ),
    "capital": (
        "The capital city serves as the political and administrative center of the country. It "
        "houses the primary government institutions, diplomatic missions, and often serves as a "
        "cultural hub. The population and economic significance can vary greatly depending on the "
        "nation's structure and historical development."
    ),
}


def _pick_mock_response(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for key, response in MOCK_RESPONSES.items():
        if key != "default" and key in prompt_lower:
            return response
    return MOCK_RESPONSES["default"]


def run_mock_inference(prompt: str, params: InferenceRequest):
    request_id = str(uuid.uuid4())[:8]
    response_text = _pick_mock_response(prompt)
    words = response_text.split(" ")
    start_time = time.time()

    all_token_events: list[TokenEvent] = []
    total_draft_generated = 0
    total_draft_accepted = 0
    speculation_rounds = 0

    i = 0
    output_token_count = 0
    while i < len(words) and output_token_count < params.max_tokens:
        speculation_rounds += 1
        round_token_events: list[TokenEvent] = []
        round_drafted = 0
        round_accepted = 0
        round_corrected = 0

        draft_count = min(params.draft_tokens, len(words) - i, params.max_tokens - output_token_count)

        for j in range(draft_count):
            word = words[i + j]
            prefix = " " if (i + j) > 0 else ""
            text = prefix + word
            round_drafted += 1
            total_draft_generated += 1

            roll = random.random()
            if roll < 0.80:
                round_token_events.append(TokenEvent(text=text, type="accepted", logprob=-0.1))
                round_accepted += 1
                total_draft_accepted += 1
                output_token_count += 1
            else:
                round_token_events.append(TokenEvent(text=text, type="rejected", logprob=-2.5))
                corrections = {
                    "fundamentally": "profoundly",
                    "changed": "transformed",
                    "shows": "demonstrates",
                    "famous": "well-known",
                    "explains": "describes",
                    "significant": "notable",
                    "vast": "enormous",
                    "dramatic": "remarkable",
                    "evolved": "progressed",
                    "modern": "contemporary",
                }
                corrected_word = corrections.get(word, word + "s" if not word.endswith("s") else word[:-1])
                round_corrected += 1
                total_draft_generated += 1
                total_draft_accepted += 1
                round_token_events.append(
                    TokenEvent(text=prefix + corrected_word, type="corrected", logprob=-0.3)
                )
                output_token_count += 1
                break

        i += draft_count if round_corrected == 0 else (j + 1)  # noqa: F821

        all_token_events.extend(round_token_events)
        verify_time = random.uniform(5, 25)
        rate = round_accepted / round_drafted if round_drafted > 0 else 0.0

        round_event = RoundEvent(
            round_num=speculation_rounds,
            drafted=round_drafted,
            accepted=round_accepted,
            corrected=round_corrected,
            verification_time_ms=verify_time,
            acceptance_rate=rate,
        )
        yield ("round", round_event, round_token_events)
        time.sleep(random.uniform(0.03, 0.08))

    elapsed = (time.time() - start_time) * 1000
    acceptance_rate = total_draft_accepted / total_draft_generated if total_draft_generated > 0 else 0.0
    generated_text = " ".join(
        te.text.strip() for te in all_token_events if te.type in ("accepted", "corrected")
    )
    summary = InferenceResponse(
        request_id=request_id,
        generated_text=generated_text,
        tokens=all_token_events,
        total_tokens=len(all_token_events),
        draft_tokens_generated=total_draft_generated,
        draft_tokens_accepted=total_draft_accepted,
        generation_time_ms=elapsed,
        acceptance_rate=acceptance_rate,
        speculation_rounds=speculation_rounds,
    )
    yield ("done", summary, [])


# ── Real inference (delegated to remote draft node) ──

def run_real_inference(prompt: str, params: InferenceRequest):
    print(f"Starting run_real_inference for prompt: {prompt[:50]}...")
    request_id = str(uuid.uuid4())[:8]
    model_prompt = _build_model_prompt(prompt)

    print(f"Routing request {request_id} to draft node...")
    route = _route_to_draft_node(request_id=request_id, prompt=model_prompt)
    draft_address = route["assigned_draft_node_address"]
    draft_id = route.get("assigned_draft_node_id", "unknown")
    print(f"Router assigned draft node: {draft_id} ({draft_address})")

    request = speculative_decoding_pb2.InferenceJobRequest(
        request_id=request_id,
        prompt=model_prompt,
        params=common_pb2.InferenceParams(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_k=params.top_k,
            draft_tokens=params.draft_tokens,
        ),
        # Draft node uses this model_id to route to a target node.
        model_id=TARGET_MODEL_ID,
        timestamp=int(time.time() * 1000),
    )

    stub = _get_draft_stub(draft_address)
    print(f"Calling StreamInference on draft node at {draft_address}")
    stream = stub.StreamInference(request)
    print("StreamInference call initiated, starting to iterate...")

    round_num = 0
    round_buffer: list[TokenEvent] = []
    final_response = None

    try:
        for chunk in stream:
            print(f"Received chunk from draft node: is_final={chunk.is_final}")
            if chunk.is_final:
                final_response = chunk.final_response
                break

            token_event = TokenEvent(
                text=chunk.token.text,
                type="accepted",
                token_id=chunk.token.token_id,
                logprob=chunk.token.logprob,
            )
            round_buffer.append(token_event)

            if len(round_buffer) >= params.draft_tokens:
                round_num += 1
                round_event = RoundEvent(
                    round_num=round_num,
                    drafted=len(round_buffer),
                    accepted=len(round_buffer),
                    corrected=0,
                    verification_time_ms=0.0,
                    acceptance_rate=1.0,
                )
                yield ("round", round_event, list(round_buffer))
                round_buffer.clear()
    except grpc.RpcError as exc:
        print(f"gRPC error: code={exc.code()}, details={exc.details()}")
        raise RuntimeError(f"Draft node RPC failed: {exc.code()} {exc.details()}") from exc
    except Exception as exc:
        print(f"Unexpected error during stream iteration: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        raise

    if round_buffer:
        round_num += 1
        round_event = RoundEvent(
            round_num=round_num,
            drafted=len(round_buffer),
            accepted=len(round_buffer),
            corrected=0,
            verification_time_ms=0.0,
            acceptance_rate=1.0,
        )
        yield ("round", round_event, list(round_buffer))

    if final_response is None:
        print("ERROR: Draft node stream ended without final response")
        raise RuntimeError("Draft node stream ended without final response")

    print("Draft node stream completed successfully, building summary...")
    summary_tokens = [
        TokenEvent(
            text=token.text,
            type="accepted",
            token_id=token.token_id,
            logprob=token.logprob,
        )
        for token in final_response.tokens
    ]
    summary = InferenceResponse(
        request_id=final_response.request_id,
        generated_text=final_response.generated_text,
        tokens=summary_tokens,
        total_tokens=final_response.total_tokens,
        draft_tokens_generated=final_response.draft_tokens_generated,
        draft_tokens_accepted=final_response.draft_tokens_accepted,
        generation_time_ms=final_response.generation_time_ms,
        acceptance_rate=final_response.acceptance_rate,
        speculation_rounds=final_response.speculation_rounds,
    )
    yield ("done", summary, [])


def run_inference(prompt: str, params: InferenceRequest):
    if MOCK_MODE:
        yield from run_mock_inference(prompt, params)
    else:
        yield from run_real_inference(prompt, params)


# ── REST endpoints ──

@app.on_event("startup")
def on_startup():
    _register_frontend_client()


@app.post("/api/inference", response_model=InferenceResponse)
def inference(req: InferenceRequest):
    result = None
    for event_type, data, _ in run_inference(req.prompt, req):
        if event_type == "done":
            result = data
    return result


@app.websocket("/api/inference/stream")
async def ws_stream_inference(websocket: WebSocket):
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        params = InferenceRequest(**json.loads(raw))

        token_buffer: list[TokenEvent] = []

        async def flush_tokens():
            nonlocal token_buffer
            if not token_buffer:
                return
            for token in token_buffer:
                await websocket.send_text(json.dumps({"type": "token", "data": token.model_dump()}))
            token_buffer = []

        gen = run_inference(params.prompt, params)
        while True:
            next_item = await asyncio.to_thread(lambda: next(gen, None))
            if next_item is None:
                break

            event_type, data, tokens = next_item
            if event_type == "round":
                for token in tokens:
                    token_buffer.append(token)
                    if len(token_buffer) >= TOKEN_SEND_BATCH_SIZE:
                        await flush_tokens()
                await flush_tokens()
                await websocket.send_text(json.dumps({"type": "round", "data": data.model_dump()}))
            elif event_type == "done":
                await flush_tokens()
                await websocket.send_text(json.dumps({"type": "done", "data": data.model_dump()}))

    except WebSocketDisconnect:
        print("WebSocket disconnected by client")
    except Exception as e:
        print(f"Error in WebSocket handler: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({"type": "error", "data": {"message": str(e)}}))
        except Exception as send_error:
            print(f"Failed to send error to WebSocket: {send_error}")


def _format_memory(memory_bytes: int) -> str:
    if memory_bytes <= 0:
        return "N/A"
    gib = memory_bytes / (1024 ** 3)
    return f"{gib:.0f} GB"


@app.get("/api/nodes", response_model=list[NodeInfo])
def get_nodes():
    router_state = _router_get("/state")
    if router_state:
        nodes: list[NodeInfo] = []

        for target in router_state.get("target_nodes", []):
            nodes.append(
                NodeInfo(
                    id=target.get("worker_id", "target-unknown"),
                    type="target",
                    hardware=target.get("gpu_model") or "Target GPU",
                    model=target.get("model_id") or target.get("model_name") or "unknown",
                    status="online" if target.get("online") else "offline",
                    latency=0.0,
                    price=0.0,
                    gpu_memory=_format_memory(int(target.get("gpu_memory_bytes", 0))),
                )
            )

        for draft in router_state.get("draft_nodes", []):
            nodes.append(
                NodeInfo(
                    id=draft.get("draft_node_id", "draft-unknown"),
                    type="draft",
                    hardware=draft.get("gpu_model") or ("Mock CPU" if MOCK_MODE else "Edge GPU"),
                    model=draft.get("model_id") or draft.get("model_name") or DRAFT_MODEL_ID,
                    status="online" if draft.get("online") else "offline",
                    latency=0.0,
                    price=0.0,
                    gpu_memory=_format_memory(int(draft.get("gpu_memory_bytes", 0))),
                )
            )
        return nodes

    return [
        NodeInfo(
            id="target-0",
            type="target",
            hardware="GPU Server",
            model=TARGET_MODEL_ID,
            status="online",
            latency=12,
            price=2.49,
            gpu_memory="80 GB",
        ),
        NodeInfo(
            id="draft-0",
            type="draft",
            hardware="Edge GPU" if not MOCK_MODE else "Mock CPU",
            model=DRAFT_MODEL_ID if not MOCK_MODE else "mock-model",
            status="online",
            latency=45,
            price=0.05,
            gpu_memory="12 GB" if not MOCK_MODE else "N/A",
        ),
    ]


@app.get("/api/stats", response_model=NetworkStats)
def get_stats():
    router_stats = _router_get("/stats")
    if router_stats:
        return NetworkStats(
            active_draft_nodes=int(router_stats.get("active_draft_nodes", 0)),
            active_target_nodes=int(router_stats.get("active_target_nodes", 0)),
            total_tps=145 if MOCK_MODE else 0.0,
            avg_acceptance_rate=0.82 if MOCK_MODE else 0.0,
            avg_cost_per_1k=0.0004,
        )

    return NetworkStats(
        active_draft_nodes=1,
        active_target_nodes=1,
        total_tps=145 if MOCK_MODE else 0.0,
        avg_acceptance_rate=0.82 if MOCK_MODE else 0.0,
        avg_cost_per_1k=0.0004,
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "mock": MOCK_MODE}


@app.post("/api/warmup")
async def warmup():
    if MOCK_MODE:
        return {"status": "ok", "mock": True, "warmed": False}
    _register_frontend_client()
    return {"status": "ok", "mock": False, "warmed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpecNet Frontend Bridge")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no GPU required)")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT, help="Port to listen on")
    args = parser.parse_args()

    if args.mock:
        MOCK_MODE = True

    print(f"\n{'='*60}")
    print("  SpecNet Frontend Bridge")
    print(f"  Port: {args.port}")
    print(f"  Mode: {'MOCK' if MOCK_MODE else 'LIVE (router + draft gRPC)'}")
    print(f"  Router: {ROUTER_HTTP_BASE}")
    if not MOCK_MODE:
        print(f"  Draft model filter: {DRAFT_MODEL_ID}")
        print(f"  Target model request: {TARGET_MODEL_ID}")
        print(f"  Prompt format: {PROMPT_FORMAT}")
        print(f"  System prompt enabled: {'yes' if SYSTEM_PROMPT.strip() else 'no'}")
    print(f"{'='*60}\n")

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)
