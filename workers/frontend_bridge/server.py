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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")
PROMPT_FORMAT = os.getenv("PROMPT_FORMAT", "chatml").lower()

# ── Pricing Configuration ──

PRICING = {
    "draft_token_generated": 0.00005,  # $0.00005 per draft token
    "draft_token_accepted": 0.00002,   # Bonus for accepted tokens
    "target_token_verified": 0.0002,   # $0.0002 per verified token
    "inference_base": 0.001,           # Base fee per inference
}

app = FastAPI(title="SpecNet Frontend Bridge")

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
    max_tokens: int = Field(default=64, ge=1, le=512)
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
    while i < len(words) and len(all_token_events) < params.max_tokens:
        speculation_rounds += 1
        round_token_events: list[TokenEvent] = []
        round_drafted = 0
        round_accepted = 0
        round_corrected = 0

        # Simulate a draft round of N tokens
        draft_count = min(params.draft_tokens, len(words) - i, params.max_tokens - len(all_token_events))

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
def inference(req: InferenceRequest):
    """Submit a prompt and get the full inference response."""
    result = None
    for event_type, data, _ in run_inference(req.prompt, req):
        if event_type == "done":
            result = data
    return result

@app.websocket("/api/inference/stream")
async def ws_stream_inference(websocket: WebSocket):
    """
    WebSocket endpoint for streaming inference.

    Client sends: {"prompt": "...", "max_tokens": 64, ...}
    Server sends:
      - {"type": "token", "data": {"text": "...", "type": "accepted"}}
      - {"type": "round", "data": {"round_num": 1, "accepted": 3, ...}}
      - {"type": "done", "data": {"request_id": "...", ...}}
    """
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
    # Real nodes - actual Modal infrastructure
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
            location={"lat": 37.7749, "lng": -122.4194, "city": "San Francisco", "country": "USA (Modal)"},
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
            location={"lat": 37.7749, "lng": -122.4194, "city": "San Francisco", "country": "USA (Modal)"},
            earnings=round(earnings_tracker.total_earnings * 0.4, 2),
            uptime=100.0,
        ),
    ]

    # Add demo nodes if requested
    if include_demo:
        demo_nodes = [
            # Demo Target nodes (verification) - High-end GPUs
            NodeInfo(
                id="demo-target-eu-central",
                type="target",
                hardware="NVIDIA A100 80GB",
                model="Qwen/Qwen2.5-3B-Instruct",
                status="online",
                latency=18,
                price=2.35,
                gpu_memory="80 GB",
                location={"lat": 52.5200, "lng": 13.4050, "city": "Berlin", "country": "Germany (Demo)"},
                earnings=0.0,
                uptime=99.5,
            ),
            NodeInfo(
                id="demo-target-asia-east",
                type="target",
                hardware="NVIDIA A100 80GB",
                model="Qwen/Qwen2.5-3B-Instruct",
                status="busy",
                latency=25,
                price=2.29,
                gpu_memory="80 GB",
                location={"lat": 35.6762, "lng": 139.6503, "city": "Tokyo", "country": "Japan (Demo)"},
                earnings=0.0,
                uptime=99.2,
            ),

            # Demo Draft nodes - Distributed edge compute
            NodeInfo(
                id="demo-draft-us-east",
                type="draft",
                hardware="NVIDIA A10G 24GB",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                status="online",
                latency=8,
                price=0.05,
                gpu_memory="24 GB",
                location={"lat": 40.7128, "lng": -74.0060, "city": "New York", "country": "USA (Demo)"},
                earnings=0.0,
                uptime=99.9,
            ),
            NodeInfo(
                id="demo-draft-us-west",
                type="draft",
                hardware="NVIDIA A10G 24GB",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                status="online",
                latency=10,
                price=0.05,
                gpu_memory="24 GB",
                location={"lat": 47.6062, "lng": -122.3321, "city": "Seattle", "country": "USA (Demo)"},
                earnings=0.0,
                uptime=98.7,
            ),
            NodeInfo(
                id="demo-draft-eu-west",
                type="draft",
                hardware="NVIDIA RTX 4090 24GB",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                status="online",
                latency=15,
                price=0.04,
                gpu_memory="24 GB",
                location={"lat": 51.5074, "lng": -0.1278, "city": "London", "country": "UK (Demo)"},
                earnings=0.0,
                uptime=97.5,
            ),
            NodeInfo(
                id="demo-draft-asia-south",
                type="draft",
                hardware="NVIDIA RTX 3090 24GB",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                status="online",
                latency=22,
                price=0.03,
                gpu_memory="24 GB",
                location={"lat": 1.3521, "lng": 103.8198, "city": "Singapore", "country": "Singapore (Demo)"},
                earnings=0.0,
                uptime=96.8,
            ),
            NodeInfo(
                id="demo-draft-oceania",
                type="draft",
                hardware="NVIDIA RTX 4080 16GB",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                status="offline",
                latency=35,
                price=0.04,
                gpu_memory="16 GB",
                location={"lat": -33.8688, "lng": 151.2093, "city": "Sydney", "country": "Australia (Demo)"},
                earnings=0.0,
                uptime=95.2,
            ),
            NodeInfo(
                id="demo-draft-south-america",
                type="draft",
                hardware="NVIDIA RTX 3080 10GB",
                model="Qwen/Qwen2.5-1.5B-Instruct",
                status="online",
                latency=28,
                price=0.03,
                gpu_memory="10 GB",
                location={"lat": -23.5505, "lng": -46.6333, "city": "São Paulo", "country": "Brazil (Demo)"},
                earnings=0.0,
                uptime=94.1,
            ),
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
        # Include demo network topology
        draft_nodes = [
            "real-draft-modal",
            "demo-draft-us-east", "demo-draft-us-west", "demo-draft-eu-west",
            "demo-draft-asia-south", "demo-draft-oceania", "demo-draft-south-america"
        ]
        target_nodes = ["real-target-modal", "demo-target-eu-central", "demo-target-asia-east"]
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
    parser = argparse.ArgumentParser(description="SpecNet Frontend Bridge")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no GPU/vLLM required)")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT, help="Port to listen on")
    args = parser.parse_args()

    if args.mock:
        MOCK_MODE = True

    print(f"\n{'='*60}")
    print(f"  SpecNet Frontend Bridge")
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
