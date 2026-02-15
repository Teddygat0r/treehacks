"""
Frontend Bridge - FastAPI server bridging Next.js frontend to Modal backend.
Translates HTTP/WebSocket requests into calls to draft/target nodes.

Run with --mock to simulate speculative decoding without vLLM/GPU:
    python workers/frontend_bridge/server.py --mock
"""
import asyncio
import json
import random
import time
import uuid
import sys
import os
import argparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from collections import defaultdict

# ── Configuration ──

DRAFT_MODEL = os.getenv("DRAFT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "treehacks-verification-service")
MODAL_CLASS_NAME = os.getenv("MODAL_CLASS_NAME", "VerificationService")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8000"))
MOCK_MODE = os.getenv("MOCK_MODE", "").lower() in ("1", "true", "yes")
TOKEN_SEND_BATCH_SIZE = max(1, int(os.getenv("TOKEN_SEND_BATCH_SIZE", "1")))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")
PROMPT_FORMAT = os.getenv("PROMPT_FORMAT", "chatml").lower()

# ── Pricing Configuration ──

PRICING = {
    "draft_token_generated": 0.00005,  # $0.00005 per draft token
    "draft_token_accepted": 0.00002,   # Bonus for accepted tokens
    "target_token_verified": 0.0002,   # $0.0002 per verified token
    "inference_base": 0.001,           # Base fee per inference
}

app = FastAPI(title="Nexus Frontend Bridge")

# ── API Key Validation ──

def validate_api_key(authorization: Optional[str] = Header(None)):
    """Validate API key from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing API key. Include 'Authorization: Bearer YOUR_API_KEY' header.")

    # Extract the key (support both "Bearer nx_..." and just "nx_..." formats)
    api_key = authorization.replace("Bearer ", "").strip()

    # Validate key format (starts with nx_ and has reasonable length)
    if not api_key.startswith("nx_") or len(api_key) < 10:
        raise HTTPException(status_code=401, detail="Invalid API key format. Key must start with 'nx_'.")

    # For demo purposes, accept any key starting with nx_
    # In production, you'd validate against a database
    return api_key

# ── Earnings Storage ──

class EarningsTracker:
    def __init__(self):
        self.total_earnings = 0.0
        self.daily_earnings = defaultdict(float)  # date -> earnings
        self.hourly_activity = defaultdict(lambda: {"requests": 0, "tokens": 0, "earnings": 0.0})
        self.payouts = []  # List of payout records
        self.inference_history = []  # Recent inference records
        self.total_tokens = 0  # Track total tokens drafted
        self.total_acceptance_rate = 0.0  # Track cumulative acceptance rate for averaging
        self.active_connections = []  # Track active inference routes
        self.last_inference_route = None  # Track last inference for visualization

    def record_inference(self,
                        draft_tokens_generated: int,
                        draft_tokens_accepted: int,
                        target_tokens_verified: int,
                        acceptance_rate: float,
                        draft_node: str = "draft-us-east",
                        target_node: str = "target-us-west"):
        """Record an inference and calculate earnings"""
        # Calculate earnings
        draft_earnings = (
            draft_tokens_generated * PRICING["draft_token_generated"] +
            draft_tokens_accepted * PRICING["draft_token_accepted"]
        )
        target_earnings = target_tokens_verified * PRICING["target_token_verified"]
        total_inference_earnings = draft_earnings + target_earnings + PRICING["inference_base"]

        # Update totals
        self.total_earnings += total_inference_earnings
        self.total_tokens += draft_tokens_generated
        self.total_acceptance_rate += acceptance_rate

        # Track by date
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_earnings[today] += total_inference_earnings

        # Track by hour for activity chart
        current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
        self.hourly_activity[current_hour]["requests"] += 1
        self.hourly_activity[current_hour]["tokens"] += draft_tokens_generated + target_tokens_verified
        self.hourly_activity[current_hour]["earnings"] += total_inference_earnings

        # Track inference route for visualization
        self.last_inference_route = {
            "draft_node": draft_node,
            "target_node": target_node,
            "timestamp": datetime.now().isoformat(),
            "tokens_generated": draft_tokens_generated,
            "tokens_accepted": draft_tokens_accepted,
            "acceptance_rate": acceptance_rate,
        }

        # Store inference record
        record = {
            "timestamp": datetime.now().isoformat(),
            "draft_tokens_generated": draft_tokens_generated,
            "draft_tokens_accepted": draft_tokens_accepted,
            "target_tokens_verified": target_tokens_verified,
            "acceptance_rate": acceptance_rate,
            "draft_earnings": round(draft_earnings, 6),
            "target_earnings": round(target_earnings, 6),
            "total_earnings": round(total_inference_earnings, 6),
            "draft_node": draft_node,
            "target_node": target_node,
        }
        self.inference_history.append(record)

        # Keep only last 100 inferences
        if len(self.inference_history) > 100:
            self.inference_history = self.inference_history[-100:]

        return record

    def get_earnings_summary(self):
        """Get overall earnings summary"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_earnings = self.daily_earnings.get(today, 0.0)

        # Calculate percentage change (mock for now)
        yesterday_earnings = 15.20  # Mock data
        change_pct = ((today_earnings - yesterday_earnings) / yesterday_earnings * 100) if yesterday_earnings > 0 else 0

        # Calculate average acceptance rate
        num_inferences = len(self.inference_history)
        avg_acceptance_rate = (self.total_acceptance_rate / num_inferences * 100) if num_inferences > 0 else 0

        return {
            "total_earnings": round(self.total_earnings, 2),
            "today_earnings": round(today_earnings, 2),
            "change_percentage": round(change_pct, 1),
            "total_inferences": num_inferences,
            "total_tokens": self.total_tokens,
            "average_acceptance_rate": round(avg_acceptance_rate, 1),
        }

    def get_activity_data(self, hours: int = 24):
        """Get activity data for the last N hours"""
        activity_list = []
        for hour_key in sorted(self.hourly_activity.keys())[-hours:]:
            data = self.hourly_activity[hour_key]
            # Parse the hour_key (format: "YYYY-MM-DD HH:00") and extract just the time
            try:
                time_str = hour_key.split()[1][:5]  # Get "HH:00"
            except:
                time_str = hour_key

            activity_list.append({
                "time": time_str,
                "requests": data["requests"],
                "tokens": data["tokens"],
                "earnings": round(data["earnings"], 4),
            })
        return activity_list

    def get_recent_payouts(self, limit: int = 10):
        """Get recent payout records"""
        # For now, generate mock payouts based on earnings
        if not self.payouts and self.total_earnings > 0:
            # Generate some mock historical payouts
            from datetime import timedelta
            base_date = datetime.now()

            self.payouts = [
                {
                    "date": (base_date - timedelta(days=2)).strftime("%b %d, %Y"),
                    "amount": round(self.total_earnings * 0.35, 2),
                    "status": "Completed",
                },
                {
                    "date": (base_date - timedelta(days=5)).strftime("%b %d, %Y"),
                    "amount": round(self.total_earnings * 0.28, 2),
                    "status": "Completed",
                },
                {
                    "date": (base_date - timedelta(days=7)).strftime("%b %d, %Y"),
                    "amount": round(self.total_earnings * 0.22, 2),
                    "status": "Completed",
                },
            ]
        return self.payouts[:limit]

# Global earnings tracker instance
earnings_tracker = EarningsTracker()

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
    location: dict = {"lat": 0.0, "lng": 0.0, "city": "Unknown", "country": "Unknown"}
    earnings: float = 0.0
    uptime: float = 100.0

class NetworkStats(BaseModel):
    active_draft_nodes: int
    active_target_nodes: int
    total_tps: float
    avg_acceptance_rate: float
    avg_cost_per_1k: float

class ModelPair(BaseModel):
    id: str
    draft_model: str
    target_model: str
    draft_size: str
    target_size: str
    acceptance_rate: float
    speedup: str
    price_per_1m: float
    category: str
    available: bool


def _build_model_prompt(user_prompt: str) -> str:
    """Build a model prompt with a global system message."""
    system_prompt = SYSTEM_PROMPT.strip()
    if not system_prompt:
        return user_prompt

    if PROMPT_FORMAT == "plain":
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

    # ChatML format works well for Qwen instruct models.
    return (
        f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# ── Mock inference (no GPU required) ──

# Word bank for generating plausible mock responses
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
    """
    Simulate speculative decoding with realistic timing and token events.
    No GPU, vLLM, or gRPC required.
    """
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

        # Simulate a draft round of N tokens
        draft_count = min(params.draft_tokens, len(words) - i, params.max_tokens - output_token_count)

        for j in range(draft_count):
            word = words[i + j]
            prefix = " " if (i + j) > 0 else ""
            text = prefix + word
            round_drafted += 1
            total_draft_generated += 1

            # ~80% acceptance rate, ~15% rejected+corrected, ~5% just accepted anyway
            roll = random.random()
            if roll < 0.80:
                round_token_events.append(TokenEvent(text=text, type="accepted", logprob=-0.1))
                round_accepted += 1
                total_draft_accepted += 1
                output_token_count += 1
            else:
                # Rejected token, then a corrected replacement
                round_token_events.append(TokenEvent(text=text, type="rejected", logprob=-2.5))

                # Pick a plausible correction (synonym-ish)
                corrections = {
                    "fundamentally": "profoundly", "changed": "transformed",
                    "shows": "demonstrates", "famous": "well-known",
                    "explains": "describes", "significant": "notable",
                    "vast": "enormous", "dramatic": "remarkable",
                    "evolved": "progressed", "modern": "contemporary",
                }
                corrected_word = corrections.get(word, word + "s" if not word.endswith("s") else word[:-1])
                round_corrected += 1
                total_draft_generated += 1
                total_draft_accepted += 1
                round_token_events.append(TokenEvent(text=prefix + corrected_word, type="corrected", logprob=-0.3))
                output_token_count += 1
                break  # After a rejection, the round ends

        i += draft_count if round_corrected == 0 else (j + 1)  # noqa: F821

        all_token_events.extend(round_token_events)

        # Simulate verification latency
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

        # Small delay to simulate real timing
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

    # Record earnings for mock inference too
    earnings_tracker.record_inference(
        draft_tokens_generated=total_draft_generated,
        draft_tokens_accepted=total_draft_accepted,
        target_tokens_verified=total_draft_generated,
        acceptance_rate=acceptance_rate,
        draft_node="real-draft-modal",
        target_node="real-target-modal",
    )

    yield ("done", summary, [])

# ── Real inference (delegated to DraftNodeClient + Modal target) ──

# Cache for real worker locations (from Modal region); refreshed every 90s
_worker_location_cache = {
    "draft": {"lat": 37.7749, "lng": -122.4194, "city": "San Francisco", "country": "USA (Modal)"},
    "target": {"lat": 37.7749, "lng": -122.4194, "city": "San Francisco", "country": "USA (Modal)"},
    "fetched_at": 0,
}
WORKER_LOCATION_CACHE_TTL = 90  # seconds

def _get_modal_draft_service():
    """Get Modal draft service client."""
    if not hasattr(_get_modal_draft_service, "_client"):
        try:
            import modal
            _get_modal_draft_service._client = modal.Cls.from_name(
                "treehacks-draft-service",
                "DraftService"
            )()
            print("Connected to Modal draft service")
        except Exception as e:
            print(f"Failed to connect to Modal draft service: {e}")
            raise
    return _get_modal_draft_service._client

def _get_modal_target_service():
    """Get Modal verification (target) service client."""
    if not hasattr(_get_modal_target_service, "_client"):
        try:
            import modal
            _get_modal_target_service._client = modal.Cls.from_name(
                "treehacks-verification-service",
                "VerificationService"
            )()
            print("Connected to Modal verification service")
        except Exception as e:
            print(f"Failed to connect to Modal verification service: {e}")
            raise
    return _get_modal_target_service._client

def _get_real_worker_locations():
    """Fetch draft and target worker locations from Modal (cached). Returns (draft_location, target_location)."""
    global _worker_location_cache
    now = time.time()
    if now - _worker_location_cache["fetched_at"] < WORKER_LOCATION_CACHE_TTL:
        return _worker_location_cache["draft"], _worker_location_cache["target"]
    try:
        draft_svc = _get_modal_draft_service()
        target_svc = _get_modal_target_service()
        draft_loc = draft_svc.get_location.remote()
        target_loc = target_svc.get_location.remote()
        if isinstance(draft_loc, dict) and "lat" in draft_loc:
            _worker_location_cache["draft"] = draft_loc
        if isinstance(target_loc, dict) and "lat" in target_loc:
            _worker_location_cache["target"] = target_loc
        _worker_location_cache["fetched_at"] = time.time()
    except Exception as e:
        print(f"Could not fetch worker locations: {e}")
    return _worker_location_cache["draft"], _worker_location_cache["target"]

def _get_draft_client():
    if not hasattr(_get_draft_client, "_client"):
        from workers.draft_node.client import DraftNodeClient
        _get_draft_client._client = DraftNodeClient(
            draft_model=DRAFT_MODEL,
            num_draft_tokens=5,
            modal_app_name=MODAL_APP_NAME,
            modal_class_name=MODAL_CLASS_NAME,
        )
    return _get_draft_client._client


def run_real_inference(prompt: str, params: InferenceRequest):
    """
    Run speculative decoding through Modal DraftService.
    Yields (event_type, data, tokens) tuples.
    """
    request_id = str(uuid.uuid4())[:8]
    model_prompt = _build_model_prompt(prompt)

    # Get Modal draft service
    draft_service = _get_modal_draft_service()

    # Call Modal service
    result = draft_service.execute_inference.remote(
        request_id=request_id,
        prompt=model_prompt,
        max_tokens=params.max_tokens,
        temperature=params.temperature,
        top_k=params.top_k,
        draft_tokens=params.draft_tokens,
    )

    # Convert result to our format
    token_events = [
        TokenEvent(
            text=t["text"],
            type=t.get("type", "accepted"),
            token_id=t.get("token_id", 0),
            logprob=t.get("logprob", 0.0),
        )
        for t in result["tokens"]
    ]

    # Debug: Count token types
    type_counts = {"accepted": 0, "rejected": 0, "corrected": 0}
    for t in result["tokens"]:
        token_type = t.get("type", "accepted")
        type_counts[token_type] = type_counts.get(token_type, 0) + 1

    print(f"Token types in response: {type_counts}")
    print(f"Acceptance rate: {result['acceptance_rate']:.1%}")
    print(f"Draft generated: {result['draft_tokens_generated']}, accepted: {result['draft_tokens_accepted']}")

    round_event = RoundEvent(
        round_num=result["speculation_rounds"],
        drafted=result["draft_tokens_generated"],
        accepted=result["draft_tokens_accepted"],
        corrected=max(0, result["total_tokens"] - result["draft_tokens_accepted"]),
        verification_time_ms=0.0,
        acceptance_rate=result["acceptance_rate"],
    )
    yield ("round", round_event, token_events)

    summary = InferenceResponse(
        request_id=result["request_id"],
        generated_text=result["generated_text"],
        tokens=token_events,
        total_tokens=result["total_tokens"],
        draft_tokens_generated=result["draft_tokens_generated"],
        draft_tokens_accepted=result["draft_tokens_accepted"],
        generation_time_ms=result["generation_time_ms"],
        acceptance_rate=result["acceptance_rate"],
        speculation_rounds=result["speculation_rounds"],
    )

    # Record earnings for this inference
    earnings_tracker.record_inference(
        draft_tokens_generated=result["draft_tokens_generated"],
        draft_tokens_accepted=result["draft_tokens_accepted"],
        target_tokens_verified=result["draft_tokens_generated"],  # All drafted tokens get verified
        acceptance_rate=result["acceptance_rate"],
        draft_node="real-draft-modal",
        target_node="real-target-modal",
    )

    yield ("done", summary, [])

# ── Dispatch to mock or real ──

def run_inference(prompt: str, params: InferenceRequest):
    if MOCK_MODE:
        yield from run_mock_inference(prompt, params)
    else:
        yield from run_real_inference(prompt, params)

# ── REST endpoints ──

@app.post("/api/inference", response_model=InferenceResponse)
def inference(req: InferenceRequest, api_key: str = Depends(validate_api_key)):
    """Submit a prompt and get the full inference response. Requires API key."""
    result = None
    for event_type, data, _ in run_inference(req.prompt, req):
        if event_type == "done":
            result = data
    return result

@app.websocket("/api/inference/stream")
async def ws_stream_inference(websocket: WebSocket):
    """
    WebSocket endpoint for streaming inference.

    Client sends: {"api_key": "nx_...", "prompt": "...", "max_tokens": 64, ...}
    Server sends:
      - {"type": "token", "data": {"text": "...", "type": "accepted"}}
      - {"type": "round", "data": {"round_num": 1, "accepted": 3, ...}}
      - {"type": "done", "data": {"request_id": "...", ...}}
    """
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)

        # Validate API key from message
        api_key = data.get("api_key", "")
        if not api_key.startswith("nx_") or len(api_key) < 10:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"message": "Invalid or missing API key. Include 'api_key' field in request."},
            }))
            await websocket.close()
            return

        # Remove api_key from data before creating InferenceRequest
        data.pop("api_key", None)
        params = InferenceRequest(**data)

        token_buffer: list[TokenEvent] = []

        async def flush_tokens():
            nonlocal token_buffer
            if not token_buffer:
                return
            for token in token_buffer:
                await websocket.send_text(json.dumps({
                    "type": "token",
                    "data": token.model_dump(),
                }))
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
                await websocket.send_text(json.dumps({
                    "type": "round",
                    "data": data.model_dump(),
                }))
            elif event_type == "done":
                await flush_tokens()
                await websocket.send_text(json.dumps({
                    "type": "done",
                    "data": data.model_dump(),
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": {"message": str(e)},
            }))
        except:
            pass

@app.get("/api/nodes", response_model=list[NodeInfo])
def get_nodes(include_demo: bool = False):
    """Return active nodes with optional demo nodes for visualization."""
    # Real nodes - actual Modal infrastructure; locations from worker regions
    draft_loc, target_loc = _get_real_worker_locations()
    nodes = [
        NodeInfo(
            id="real-draft-modal",
            type="draft",
            hardware="Modal A10G 24GB",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            status="online",
            latency=8,
            price=0.05,
            gpu_memory="24 GB",
            location=draft_loc,
            earnings=round(earnings_tracker.total_earnings * 0.6, 2),
            uptime=100.0,
        ),
        NodeInfo(
            id="real-target-modal",
            type="target",
            hardware="Modal A100 80GB",
            model="Qwen/Qwen2.5-3B-Instruct",
            status="online",
            latency=12,
            price=2.49,
            gpu_memory="80 GB",
            location=target_loc,
            earnings=round(earnings_tracker.total_earnings * 0.4, 2),
            uptime=100.0,
        ),
    ]

    # Add demo nodes if requested - Global distribution
    if include_demo:
        demo_nodes = [
            # Demo Target nodes (verification) - Major data centers globally
            NodeInfo(id="demo-target-eu-west", type="target", hardware="NVIDIA A100 80GB", model="Qwen/Qwen2.5-3B-Instruct", status="online", latency=18, price=2.35, gpu_memory="80 GB", location={"lat": 51.5074, "lng": -0.1278, "city": "London", "country": "UK"}, earnings=0.0, uptime=99.5),
            NodeInfo(id="demo-target-asia-east", type="target", hardware="NVIDIA A100 80GB", model="Qwen/Qwen2.5-3B-Instruct", status="online", latency=25, price=2.29, gpu_memory="80 GB", location={"lat": 35.6762, "lng": 139.6503, "city": "Tokyo", "country": "Japan"}, earnings=0.0, uptime=99.2),
            NodeInfo(id="demo-target-us-east", type="target", hardware="NVIDIA A100 80GB", model="Qwen/Qwen2.5-3B-Instruct", status="online", latency=15, price=2.45, gpu_memory="80 GB", location={"lat": 40.7128, "lng": -74.0060, "city": "New York", "country": "USA"}, earnings=0.0, uptime=99.8),
            NodeInfo(id="demo-target-asia-south", type="target", hardware="NVIDIA A100 80GB", model="Qwen/Qwen2.5-3B-Instruct", status="online", latency=28, price=2.20, gpu_memory="80 GB", location={"lat": 1.3521, "lng": 103.8198, "city": "Singapore", "country": "Singapore"}, earnings=0.0, uptime=98.9),
            NodeInfo(id="demo-target-eu-central", type="target", hardware="NVIDIA A100 80GB", model="Qwen/Qwen2.5-3B-Instruct", status="online", latency=19, price=2.38, gpu_memory="80 GB", location={"lat": 52.5200, "lng": 13.4050, "city": "Berlin", "country": "Germany"}, earnings=0.0, uptime=99.6),
            NodeInfo(id="demo-target-oceania", type="target", hardware="NVIDIA A100 80GB", model="Qwen/Qwen2.5-3B-Instruct", status="online", latency=32, price=2.42, gpu_memory="80 GB", location={"lat": -33.8688, "lng": 151.2093, "city": "Sydney", "country": "Australia"}, earnings=0.0, uptime=99.1),

            # Demo Draft nodes - Distributed globally
            NodeInfo(id="demo-draft-us-east-1", type="draft", hardware="NVIDIA A10G 24GB", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=8, price=0.05, gpu_memory="24 GB", location={"lat": 39.5, "lng": -75.2, "city": "Philadelphia", "country": "USA"}, earnings=0.0, uptime=99.9),
            NodeInfo(id="demo-draft-us-central", type="draft", hardware="NVIDIA RTX 4090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=9, price=0.04, gpu_memory="24 GB", location={"lat": 41.8781, "lng": -87.6298, "city": "Chicago", "country": "USA"}, earnings=0.0, uptime=99.7),
            NodeInfo(id="demo-draft-us-south", type="draft", hardware="NVIDIA RTX 3090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=11, price=0.03, gpu_memory="24 GB", location={"lat": 30.2672, "lng": -97.7431, "city": "Austin", "country": "USA"}, earnings=0.0, uptime=99.3),
            NodeInfo(id="demo-draft-us-west-1", type="draft", hardware="NVIDIA A10G 24GB", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=10, price=0.05, gpu_memory="24 GB", location={"lat": 47.6062, "lng": -122.3321, "city": "Seattle", "country": "USA"}, earnings=0.0, uptime=98.7),
            NodeInfo(id="demo-draft-canada-east", type="draft", hardware="NVIDIA RTX 4080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=13, price=0.04, gpu_memory="16 GB", location={"lat": 43.6532, "lng": -79.3832, "city": "Toronto", "country": "Canada"}, earnings=0.0, uptime=99.4),
            NodeInfo(id="demo-draft-brazil", type="draft", hardware="NVIDIA RTX 3080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=28, price=0.03, gpu_memory="10 GB", location={"lat": -23.5505, "lng": -46.6333, "city": "São Paulo", "country": "Brazil"}, earnings=0.0, uptime=98.1),

            NodeInfo(id="demo-draft-uk", type="draft", hardware="NVIDIA RTX 4090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=15, price=0.04, gpu_memory="24 GB", location={"lat": 53.4808, "lng": -2.2426, "city": "Manchester", "country": "UK"}, earnings=0.0, uptime=99.5),
            NodeInfo(id="demo-draft-france", type="draft", hardware="NVIDIA RTX 3090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=17, price=0.03, gpu_memory="24 GB", location={"lat": 48.8566, "lng": 2.3522, "city": "Paris", "country": "France"}, earnings=0.0, uptime=99.2),
            NodeInfo(id="demo-draft-germany", type="draft", hardware="NVIDIA RTX 4080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=16, price=0.04, gpu_memory="16 GB", location={"lat": 50.1109, "lng": 8.6821, "city": "Frankfurt", "country": "Germany"}, earnings=0.0, uptime=99.6),
            NodeInfo(id="demo-draft-netherlands", type="draft", hardware="NVIDIA RTX 4080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=16, price=0.04, gpu_memory="16 GB", location={"lat": 52.3702, "lng": 4.8952, "city": "Amsterdam", "country": "Netherlands"}, earnings=0.0, uptime=99.6),
            NodeInfo(id="demo-draft-sweden", type="draft", hardware="NVIDIA RTX 3080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=19, price=0.03, gpu_memory="10 GB", location={"lat": 59.3293, "lng": 18.0686, "city": "Stockholm", "country": "Sweden"}, earnings=0.0, uptime=99.1),

            NodeInfo(id="demo-draft-singapore", type="draft", hardware="NVIDIA RTX 3090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=22, price=0.03, gpu_memory="24 GB", location={"lat": 3.1390, "lng": 101.6869, "city": "Kuala Lumpur", "country": "Malaysia"}, earnings=0.0, uptime=99.8),
            NodeInfo(id="demo-draft-hong-kong", type="draft", hardware="NVIDIA RTX 4080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=24, price=0.04, gpu_memory="16 GB", location={"lat": 22.3193, "lng": 114.1694, "city": "Hong Kong", "country": "Hong Kong"}, earnings=0.0, uptime=99.3),
            NodeInfo(id="demo-draft-tokyo", type="draft", hardware="NVIDIA RTX 4090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=26, price=0.04, gpu_memory="24 GB", location={"lat": 34.6937, "lng": 135.5023, "city": "Osaka", "country": "Japan"}, earnings=0.0, uptime=99.7),
            NodeInfo(id="demo-draft-seoul", type="draft", hardware="NVIDIA RTX 4090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=26, price=0.04, gpu_memory="24 GB", location={"lat": 37.5665, "lng": 126.9780, "city": "Seoul", "country": "South Korea"}, earnings=0.0, uptime=99.7),
            NodeInfo(id="demo-draft-mumbai", type="draft", hardware="NVIDIA RTX 3070", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=29, price=0.03, gpu_memory="8 GB", location={"lat": 19.0760, "lng": 72.8777, "city": "Mumbai", "country": "India"}, earnings=0.0, uptime=98.2),
            NodeInfo(id="demo-draft-dubai", type="draft", hardware="NVIDIA RTX 3090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=25, price=0.03, gpu_memory="24 GB", location={"lat": 25.2048, "lng": 55.2708, "city": "Dubai", "country": "UAE"}, earnings=0.0, uptime=99.0),

            NodeInfo(id="demo-draft-sydney", type="draft", hardware="NVIDIA RTX 4080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=33, price=0.04, gpu_memory="16 GB", location={"lat": -34.9285, "lng": 138.6007, "city": "Adelaide", "country": "Australia"}, earnings=0.0, uptime=99.0),
            NodeInfo(id="demo-draft-melbourne", type="draft", hardware="NVIDIA RTX 4080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=33, price=0.04, gpu_memory="16 GB", location={"lat": -37.8136, "lng": 144.9631, "city": "Melbourne", "country": "Australia"}, earnings=0.0, uptime=99.0),
            NodeInfo(id="demo-draft-cape-town", type="draft", hardware="NVIDIA RTX 3060", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=38, price=0.02, gpu_memory="12 GB", location={"lat": -33.9249, "lng": 18.4241, "city": "Cape Town", "country": "South Africa"}, earnings=0.0, uptime=97.8),

            # Additional draft nodes for denser network
            NodeInfo(id="demo-draft-los-angeles", type="draft", hardware="NVIDIA RTX 4090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=11, price=0.04, gpu_memory="24 GB", location={"lat": 34.0522, "lng": -118.2437, "city": "Los Angeles", "country": "USA"}, earnings=0.0, uptime=99.4),
            NodeInfo(id="demo-draft-miami", type="draft", hardware="NVIDIA RTX 3080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=13, price=0.03, gpu_memory="10 GB", location={"lat": 25.7617, "lng": -80.1918, "city": "Miami", "country": "USA"}, earnings=0.0, uptime=98.9),
            NodeInfo(id="demo-draft-boston", type="draft", hardware="NVIDIA RTX 3090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=12, price=0.03, gpu_memory="24 GB", location={"lat": 42.3601, "lng": -71.0589, "city": "Boston", "country": "USA"}, earnings=0.0, uptime=99.2),
            NodeInfo(id="demo-draft-denver", type="draft", hardware="NVIDIA RTX 3070", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=14, price=0.03, gpu_memory="8 GB", location={"lat": 39.7392, "lng": -104.9903, "city": "Denver", "country": "USA"}, earnings=0.0, uptime=98.8),
            NodeInfo(id="demo-draft-vancouver", type="draft", hardware="NVIDIA RTX 3080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=12, price=0.03, gpu_memory="10 GB", location={"lat": 49.2827, "lng": -123.1207, "city": "Vancouver", "country": "Canada"}, earnings=0.0, uptime=99.1),
            NodeInfo(id="demo-draft-mexico-city", type="draft", hardware="NVIDIA RTX 3070", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=18, price=0.03, gpu_memory="8 GB", location={"lat": 19.4326, "lng": -99.1332, "city": "Mexico City", "country": "Mexico"}, earnings=0.0, uptime=98.5),
            NodeInfo(id="demo-draft-buenos-aires", type="draft", hardware="NVIDIA RTX 3060", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=30, price=0.02, gpu_memory="12 GB", location={"lat": -34.6037, "lng": -58.3816, "city": "Buenos Aires", "country": "Argentina"}, earnings=0.0, uptime=97.9),

            NodeInfo(id="demo-draft-madrid", type="draft", hardware="NVIDIA RTX 3070", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=20, price=0.03, gpu_memory="8 GB", location={"lat": 40.4168, "lng": -3.7038, "city": "Madrid", "country": "Spain"}, earnings=0.0, uptime=98.7),
            NodeInfo(id="demo-draft-rome", type="draft", hardware="NVIDIA RTX 3070", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=21, price=0.03, gpu_memory="8 GB", location={"lat": 41.9028, "lng": 12.4964, "city": "Rome", "country": "Italy"}, earnings=0.0, uptime=98.4),
            NodeInfo(id="demo-draft-zurich", type="draft", hardware="NVIDIA RTX 4090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=17, price=0.04, gpu_memory="24 GB", location={"lat": 47.3769, "lng": 8.5417, "city": "Zurich", "country": "Switzerland"}, earnings=0.0, uptime=99.6),
            NodeInfo(id="demo-draft-warsaw", type="draft", hardware="NVIDIA RTX 3070", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=20, price=0.03, gpu_memory="8 GB", location={"lat": 52.2297, "lng": 21.0122, "city": "Warsaw", "country": "Poland"}, earnings=0.0, uptime=98.8),
            NodeInfo(id="demo-draft-copenhagen", type="draft", hardware="NVIDIA RTX 4080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=19, price=0.04, gpu_memory="16 GB", location={"lat": 55.6761, "lng": 12.5683, "city": "Copenhagen", "country": "Denmark"}, earnings=0.0, uptime=99.3),
            NodeInfo(id="demo-draft-oslo", type="draft", hardware="NVIDIA RTX 3090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=20, price=0.03, gpu_memory="24 GB", location={"lat": 59.9139, "lng": 10.7522, "city": "Oslo", "country": "Norway"}, earnings=0.0, uptime=99.2),

            NodeInfo(id="demo-draft-beijing", type="draft", hardware="NVIDIA RTX 4090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=28, price=0.04, gpu_memory="24 GB", location={"lat": 39.9042, "lng": 116.4074, "city": "Beijing", "country": "China"}, earnings=0.0, uptime=99.4),
            NodeInfo(id="demo-draft-shanghai", type="draft", hardware="NVIDIA RTX 3090", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=27, price=0.03, gpu_memory="24 GB", location={"lat": 31.2304, "lng": 121.4737, "city": "Shanghai", "country": "China"}, earnings=0.0, uptime=99.2),
            NodeInfo(id="demo-draft-taipei", type="draft", hardware="NVIDIA RTX 4080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=26, price=0.04, gpu_memory="16 GB", location={"lat": 25.0330, "lng": 121.5654, "city": "Taipei", "country": "Taiwan"}, earnings=0.0, uptime=99.5),
            NodeInfo(id="demo-draft-bangkok", type="draft", hardware="NVIDIA RTX 3080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=27, price=0.03, gpu_memory="10 GB", location={"lat": 13.7563, "lng": 100.5018, "city": "Bangkok", "country": "Thailand"}, earnings=0.0, uptime=98.8),
            NodeInfo(id="demo-draft-jakarta", type="draft", hardware="NVIDIA RTX 3070", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=29, price=0.03, gpu_memory="8 GB", location={"lat": -6.2088, "lng": 106.8456, "city": "Jakarta", "country": "Indonesia"}, earnings=0.0, uptime=98.5),
            NodeInfo(id="demo-draft-bangalore", type="draft", hardware="NVIDIA RTX 3080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=28, price=0.03, gpu_memory="10 GB", location={"lat": 12.9716, "lng": 77.5946, "city": "Bangalore", "country": "India"}, earnings=0.0, uptime=98.7),
            NodeInfo(id="demo-draft-tel-aviv", type="draft", hardware="NVIDIA RTX 4080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=24, price=0.04, gpu_memory="16 GB", location={"lat": 32.0853, "lng": 34.7818, "city": "Tel Aviv", "country": "Israel"}, earnings=0.0, uptime=99.1),

            NodeInfo(id="demo-draft-brisbane", type="draft", hardware="NVIDIA RTX 3080", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=34, price=0.03, gpu_memory="10 GB", location={"lat": -27.4698, "lng": 153.0251, "city": "Brisbane", "country": "Australia"}, earnings=0.0, uptime=98.9),
            NodeInfo(id="demo-draft-auckland", type="draft", hardware="NVIDIA RTX 3070", model="Qwen/Qwen2.5-1.5B-Instruct", status="online", latency=36, price=0.03, gpu_memory="8 GB", location={"lat": -36.8485, "lng": 174.7633, "city": "Auckland", "country": "New Zealand"}, earnings=0.0, uptime=98.6),
        ]
        nodes.extend(demo_nodes)

    return nodes

@app.get("/api/stats", response_model=NetworkStats)
def get_stats():
    """Return network-wide statistics."""
    # Calculate real stats from earnings tracker
    num_inferences = len(earnings_tracker.inference_history)

    # Calculate average acceptance rate from real data
    if num_inferences > 0:
        avg_acceptance = earnings_tracker.total_acceptance_rate / num_inferences
    else:
        avg_acceptance = 0.0

    # Count active nodes (nodes with status "online" or "busy")
    nodes_data = get_nodes()
    active_draft = sum(1 for n in nodes_data if n.type == "draft" and n.status in ["online", "busy"])
    active_target = sum(1 for n in nodes_data if n.type == "target" and n.status in ["online", "busy"])

    return NetworkStats(
        active_draft_nodes=active_draft,
        active_target_nodes=active_target,
        total_tps=0,  # Could calculate from recent inference rates
        avg_acceptance_rate=avg_acceptance,
        avg_cost_per_1k=0.0004,
    )

@app.get("/api/models/pairs", response_model=list[ModelPair])
def get_model_pairs():
    """Return available model pairs for speculative decoding"""
    return [
        ModelPair(
            id="opt-350m-6.7b",
            draft_model="OPT-350M",
            target_model="OPT-6.7B",
            draft_size="350M",
            target_size="6.7B",
            acceptance_rate=0.65,
            speedup="2.1x",
            price_per_1m=0.45,
            category="OPT",
            available=True,
        ),
        ModelPair(
            id="opt-125m-6.7b",
            draft_model="OPT-125M",
            target_model="OPT-6.7B",
            draft_size="125M",
            target_size="6.7B",
            acceptance_rate=0.55,
            speedup="1.8x",
            price_per_1m=0.38,
            category="OPT",
            available=True,
        ),
        ModelPair(
            id="llama-2-7b-70b",
            draft_model="Llama-2-7B",
            target_model="Llama-2-70B",
            draft_size="7B",
            target_size="70B",
            acceptance_rate=0.70,
            speedup="2.5x",
            price_per_1m=0.65,
            category="Llama",
            available=True,
        ),
        ModelPair(
            id="qwen-2.5-1.5b-72b",
            draft_model="Qwen-2.5-1.5B",
            target_model="Qwen-2.5-72B",
            draft_size="1.5B",
            target_size="72B",
            acceptance_rate=0.75,
            speedup="2.8x",
            price_per_1m=0.72,
            category="Qwen",
            available=True,
        ),
    ]

@app.get("/api/health")
def health():
    return {"status": "ok", "mock": MOCK_MODE}

# ── Provider Earnings Endpoints ──

@app.get("/api/provider/earnings")
def get_provider_earnings():
    """Get provider earnings summary"""
    return earnings_tracker.get_earnings_summary()

@app.get("/api/provider/activity")
def get_provider_activity(hours: int = 24):
    """Get provider activity data for charts"""
    return {
        "activity": earnings_tracker.get_activity_data(hours=hours),
        "pricing": PRICING,
    }

@app.get("/api/provider/payouts")
def get_provider_payouts(limit: int = 10):
    """Get recent payout records"""
    return {
        "payouts": earnings_tracker.get_recent_payouts(limit=limit)
    }

@app.get("/api/provider/stats")
def get_provider_stats():
    """Get detailed provider statistics"""
    summary = earnings_tracker.get_earnings_summary()
    recent_inferences = earnings_tracker.inference_history[-10:] if earnings_tracker.inference_history else []

    # Calculate average acceptance rate
    if recent_inferences:
        avg_acceptance = sum(inf["acceptance_rate"] for inf in recent_inferences) / len(recent_inferences)
    else:
        avg_acceptance = 0.0

    return {
        **summary,
        "avg_acceptance_rate": round(avg_acceptance, 3),
        "recent_inferences": recent_inferences,
    }


@app.get("/api/network/connections")
def get_network_connections(include_demo: bool = False):
    """Get network connections and active inference routes"""
    import time

    # Define static network topology (which nodes can connect to which)
    static_connections = []

    if include_demo:
        # Include demo network topology (Global) - all draft nodes connect to all target nodes
        draft_nodes = [
            "real-draft-modal",
            # North America
            "demo-draft-us-east-1", "demo-draft-us-central", "demo-draft-us-south", "demo-draft-us-west-1",
            "demo-draft-los-angeles", "demo-draft-miami", "demo-draft-boston", "demo-draft-denver",
            "demo-draft-canada-east", "demo-draft-vancouver", "demo-draft-mexico-city",
            # South America
            "demo-draft-brazil", "demo-draft-buenos-aires",
            # Europe
            "demo-draft-uk", "demo-draft-france", "demo-draft-germany", "demo-draft-netherlands", "demo-draft-sweden",
            "demo-draft-madrid", "demo-draft-rome", "demo-draft-zurich", "demo-draft-warsaw",
            "demo-draft-copenhagen", "demo-draft-oslo",
            # Asia
            "demo-draft-singapore", "demo-draft-hong-kong", "demo-draft-tokyo", "demo-draft-seoul",
            "demo-draft-beijing", "demo-draft-shanghai", "demo-draft-taipei", "demo-draft-bangkok",
            "demo-draft-jakarta", "demo-draft-mumbai", "demo-draft-bangalore",
            # Middle East
            "demo-draft-dubai", "demo-draft-tel-aviv",
            # Oceania
            "demo-draft-sydney", "demo-draft-melbourne", "demo-draft-brisbane", "demo-draft-auckland",
            # Africa
            "demo-draft-cape-town",
        ]
        target_nodes = [
            "real-target-modal",
            "demo-target-us-east", "demo-target-eu-west", "demo-target-eu-central",
            "demo-target-asia-east", "demo-target-asia-south", "demo-target-oceania",
        ]
    else:
        # Only real nodes
        draft_nodes = ["real-draft-modal"]
        target_nodes = ["real-target-modal"]

    for draft in draft_nodes:
        for target in target_nodes:
            static_connections.append({
                "from": draft,
                "to": target,
                "type": "potential"
            })

    # Check for active inference (within last 5 seconds)
    active_route = None
    if earnings_tracker.last_inference_route:
        route_time = datetime.fromisoformat(earnings_tracker.last_inference_route["timestamp"])
        time_diff = (datetime.now() - route_time).total_seconds()

        if time_diff < 5:  # Show as active for 5 seconds after inference
            active_route = {
                "from": earnings_tracker.last_inference_route["draft_node"],
                "to": earnings_tracker.last_inference_route["target_node"],
                "type": "active",
                "tokens": earnings_tracker.last_inference_route["tokens_generated"],
                "acceptance_rate": earnings_tracker.last_inference_route["acceptance_rate"],
                "timestamp": earnings_tracker.last_inference_route["timestamp"],
            }

    return {
        "static_connections": static_connections,
        "active_route": active_route,
        "total_inferences": len(earnings_tracker.inference_history),
    }


@app.post("/api/warmup")
async def warmup():
    """
    Warm up draft model/client so first prompt has low latency.
    Triggered by frontend on page load.
    """
    if MOCK_MODE:
        return {"status": "ok", "mock": True, "warmed": False}

    try:
        await asyncio.to_thread(_get_modal_draft_service)
        return {"status": "ok", "mock": False, "warmed": True}
    except Exception as e:
        print(f"Warmup failed: {e}")
        return {"status": "ok", "mock": False, "warmed": False, "error": str(e)}

# ── Startup ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nexus Frontend Bridge")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no GPU/vLLM required)")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT, help="Port to listen on")
    args = parser.parse_args()

    if args.mock:
        MOCK_MODE = True

    print(f"\n{'='*60}")
    print(f"  Nexus Frontend Bridge")
    print(f"  Port: {args.port}")
    print(f"  Mode: {'MOCK (no GPU)' if MOCK_MODE else 'LIVE (vLLM + Modal)'}")
    if not MOCK_MODE:
        print(f"  Modal app: {MODAL_APP_NAME}")
        print(f"  Modal class: {MODAL_CLASS_NAME}")
        print(f"  Draft model: {DRAFT_MODEL}")
        print(f"  Prompt format: {PROMPT_FORMAT}")
        print(f"  System prompt enabled: {'yes' if SYSTEM_PROMPT.strip() else 'no'}")
    print(f"{'='*60}\n")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
