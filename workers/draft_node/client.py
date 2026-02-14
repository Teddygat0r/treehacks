"""
Draft Node implementation.

- Runs draft model generation locally
- Resolves target node address from router
- Verifies draft tokens against target node over gRPC
- Exposes DraftNodeService gRPC server for frontend bridge calls
"""
from __future__ import annotations

import argparse
from concurrent import futures
from dataclasses import dataclass
import os
import socket
import sys
import threading
import time
import uuid

import grpc
import requests
from vllm import LLM, SamplingParams

# Add proto directory to path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "proto",
    ),
)

import common_pb2
import speculative_decoding_pb2
import speculative_decoding_pb2_grpc


# Hard-coded router IP as requested.
ROUTER_HTTP_BASE = "http://127.0.0.1:8001"


def _router_post(path: str, payload: dict) -> dict | None:
    try:
        response = requests.post(f"{ROUTER_HTTP_BASE}{path}", json=payload, timeout=2.0)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(f"Router request failed ({path}): {exc}")
        return None


def _resolve_local_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


@dataclass
class Token:
    token_id: int
    text: str
    logprob: float


class DraftNodeClient:
    def __init__(
        self,
        draft_model: str = "Qwen/Qwen3-4B-Instruct",
        num_draft_tokens: int = 5,
        verification_server: str = "127.0.0.1:50051",
        draft_node_id: str | None = None,
        draft_node_address: str | None = None,
        register_with_router: bool = True,
    ):
        print(f"Initializing draft node with model: {draft_model}")
        self.llm = LLM(
            model=draft_model,
            gpu_memory_utilization=0.7,
            max_model_len=4096,
        )

        self.num_draft_tokens = num_draft_tokens
        self.verification_server = verification_server
        self.draft_model = draft_model
        self.register_with_router = register_with_router

        self.draft_node_id = draft_node_id or f"draft-{uuid.uuid4().hex[:8]}"
        self.draft_node_address = draft_node_address or ""
        self._active_requests = 0
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

        self._verification_channels: dict[str, grpc.Channel] = {}
        self._verification_stubs: dict[str, speculative_decoding_pb2_grpc.VerificationServiceStub] = {}

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(draft_model)

        if self.register_with_router:
            self.register_node()
            self._start_heartbeat()

        print("Draft node ready!")

    def register_node(self) -> None:
        payload = {
            "draft_node_id": self.draft_node_id,
            "address": self.draft_node_address,
            "model_id": self.draft_model,
            "model_name": self.draft_model,
            "gpu_model": "",
            "gpu_memory_bytes": 0,
            "max_draft_tokens": self.num_draft_tokens,
        }
        result = _router_post("/register/draft-node", payload)
        if result and result.get("accepted"):
            print(
                f"Registered draft node with router: id={self.draft_node_id}, "
                f"address={self.draft_node_address}"
            )

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        def _run():
            interval_s = 10.0
            while not self._heartbeat_stop.wait(interval_s):
                response = _router_post(
                    "/heartbeat/draft-node",
                    {
                        "draft_node_id": self.draft_node_id,
                        "available_capacity": 1,
                        "active_requests": self._active_requests,
                    },
                )
                if response and response.get("next_heartbeat_interval_ms"):
                    interval_s = max(1.0, response["next_heartbeat_interval_ms"] / 1000.0)

        self._heartbeat_thread = threading.Thread(
            target=_run,
            name="draft-router-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _resolve_target_address(self, request: speculative_decoding_pb2.InferenceJobRequest) -> str:
        if not self.register_with_router:
            return self.verification_server

        route = _router_post(
            "/route/target-node",
            {
                "request_id": request.request_id,
                "draft_node_id": self.draft_node_id,
                "model_id": request.model_id,
            },
        )
        if route and route.get("status") == "success" and route.get("worker_address"):
            return str(route["worker_address"])

        return self.verification_server

    def _get_verification_stub(
        self, address: str
    ) -> speculative_decoding_pb2_grpc.VerificationServiceStub:
        if address not in self._verification_stubs:
            channel = grpc.insecure_channel(address)
            self._verification_channels[address] = channel
            self._verification_stubs[address] = speculative_decoding_pb2_grpc.VerificationServiceStub(channel)
        return self._verification_stubs[address]

    def execute_inference_stream(self, request):
        """
        Execute inference with speculative decoding and stream round events.

        Returns:
            Yields tuples:
            - ("round", round_event_dict, token_event_dicts)
            - ("done", InferenceJobResponse, [])
        """
        start_time = time.time()
        self._active_requests += 1

        target_address = self._resolve_target_address(request)
        verification_stub = self._get_verification_stub(target_address)
        print(f"Using target node: {target_address}")

        current_text = request.prompt
        current_token_ids = self.tokenizer.encode(current_text)
        all_tokens = []

        total_draft_generated = 0
        total_draft_accepted = 0
        speculation_rounds = 0

        max_tokens = request.params.max_tokens if request.params.max_tokens > 0 else 16
        draft_tokens_per_round = (
            request.params.draft_tokens if request.params.draft_tokens > 0 else self.num_draft_tokens
        )

        eos_token_ids = set()
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            eos_token_ids.add(self.tokenizer.eos_token_id)
        for token in ("<|endoftext|>", "<|im_end|>"):
            if token in self.tokenizer.get_vocab():
                eos_token_ids.add(self.tokenizer.convert_tokens_to_ids(token))

        while len(all_tokens) < max_tokens:
            speculation_rounds += 1
            num_to_draft = min(draft_tokens_per_round, max_tokens - len(all_tokens))

            sampling_params = SamplingParams(
                temperature=request.params.temperature if request.params.temperature > 0 else 0.8,
                top_k=request.params.top_k if request.params.top_k > 0 else -1,
                top_p=0.95,
                max_tokens=num_to_draft,
                logprobs=5,
                seed=42,
            )

            outputs = self.llm.generate(
                prompts=[current_text],
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            draft_output = outputs[0].outputs[0]
            draft_token_ids = draft_output.token_ids
            if not draft_token_ids:
                break

            total_draft_generated += len(draft_token_ids)

            draft_logprobs = []
            if draft_output.logprobs:
                for token_logprobs in draft_output.logprobs:
                    token_id = list(token_logprobs.keys())[0]
                    draft_logprobs.append(token_logprobs[token_id].logprob)

            verify_request = speculative_decoding_pb2.VerificationRequest(
                request_id=request.request_id,
                session_id="session-0",
                prefix_token_ids=list(current_token_ids),
                draft_token_ids=list(draft_token_ids),
                draft_logprobs=list(draft_logprobs),
                temperature=request.params.temperature if request.params.temperature > 0 else 0.8,
                top_k=request.params.top_k if request.params.top_k > 0 else -1,
            )

            try:
                verify_response = verification_stub.VerifyDraft(verify_request)
            except grpc.RpcError as exc:
                print(f"Verification RPC failed: {exc.code()} {exc.details()}")
                break

            num_accepted = verify_response.num_accepted_tokens
            total_draft_accepted += num_accepted

            round_token_events = []
            accepted_tokens = draft_token_ids[:num_accepted]
            current_token_ids.extend(accepted_tokens)

            for token_id in accepted_tokens:
                round_token_events.append(
                    {
                        "text": self.tokenizer.decode([token_id]),
                        "type": "accepted",
                        "token_id": token_id,
                        "logprob": 0.0,
                    }
                )

            if num_accepted < len(draft_token_ids):
                rejected_token_id = draft_token_ids[num_accepted]
                round_token_events.append(
                    {
                        "text": self.tokenizer.decode([rejected_token_id]),
                        "type": "rejected",
                        "token_id": rejected_token_id,
                        "logprob": 0.0,
                    }
                )

            if verify_response.corrected_token_ids:
                current_token_ids.extend(verify_response.corrected_token_ids)
                for i, token_id in enumerate(verify_response.corrected_token_ids):
                    round_token_events.append(
                        {
                            "text": self.tokenizer.decode([token_id]),
                            "type": "corrected",
                            "token_id": token_id,
                            "logprob": (
                                verify_response.corrected_logprobs[i]
                                if i < len(verify_response.corrected_logprobs)
                                else 0.0
                            ),
                        }
                    )

            eos_reached = False
            for token_id in accepted_tokens:
                token = Token(
                    token_id=token_id,
                    text=self.tokenizer.decode([token_id]),
                    logprob=0.0,
                )
                all_tokens.append(token)
                if eos_token_ids and token_id in eos_token_ids:
                    eos_reached = True
                    break

            if not eos_reached:
                for i, token_id in enumerate(verify_response.corrected_token_ids):
                    token = Token(
                        token_id=token_id,
                        text=self.tokenizer.decode([token_id]),
                        logprob=(
                            verify_response.corrected_logprobs[i]
                            if i < len(verify_response.corrected_logprobs)
                            else 0.0
                        ),
                    )
                    all_tokens.append(token)
                    if eos_token_ids and token_id in eos_token_ids:
                        eos_reached = True
                        break

            if not eos_reached and eos_token_ids and verify_response.next_token_id in eos_token_ids:
                token = Token(
                    token_id=verify_response.next_token_id,
                    text=self.tokenizer.decode([verify_response.next_token_id]),
                    logprob=verify_response.next_token_logprob or 0.0,
                )
                all_tokens.append(token)
                current_token_ids.append(verify_response.next_token_id)
                eos_reached = True
                round_token_events.append(
                    {
                        "text": token.text,
                        "type": "accepted",
                        "token_id": token.token_id,
                        "logprob": token.logprob,
                    }
                )

            round_event = {
                "round_num": speculation_rounds,
                "drafted": len(draft_token_ids),
                "accepted": num_accepted,
                "corrected": len(verify_response.corrected_token_ids),
                "verification_time_ms": float(verify_response.verification_time_ms),
                "acceptance_rate": float(verify_response.acceptance_rate),
            }
            yield ("round", round_event, round_token_events)

            if eos_reached:
                break
            current_text = self.tokenizer.decode(current_token_ids, skip_special_tokens=True)

        elapsed = (time.time() - start_time) * 1000
        final_text = self.tokenizer.decode(current_token_ids, skip_special_tokens=True)
        acceptance_rate = total_draft_accepted / total_draft_generated if total_draft_generated > 0 else 0.0

        response = speculative_decoding_pb2.InferenceJobResponse(
            request_id=request.request_id,
            generated_text=final_text,
            tokens=[common_pb2.Token(token_id=t.token_id, text=t.text, logprob=t.logprob) for t in all_tokens],
            status=common_pb2.STATUS_SUCCESS,
            total_tokens=len(all_tokens),
            draft_tokens_generated=total_draft_generated,
            draft_tokens_accepted=total_draft_accepted,
            generation_time_ms=elapsed,
            acceptance_rate=acceptance_rate,
            speculation_rounds=speculation_rounds,
        )
        yield ("done", response, [])
        self._active_requests = max(0, self._active_requests - 1)

    def execute_inference(self, request):
        result = None
        for event_type, data, _ in self.execute_inference_stream(request):
            if event_type == "done":
                result = data
        return result

    def close(self):
        self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)

        for channel in self._verification_channels.values():
            channel.close()
        self._verification_channels.clear()
        self._verification_stubs.clear()

        if hasattr(self, "llm"):
            try:
                del self.llm.llm_engine
                del self.llm
            except Exception:
                pass

    def __del__(self):
        self.close()


class DraftNodeServiceImpl(speculative_decoding_pb2_grpc.DraftNodeServiceServicer):
    def __init__(self, draft_client: DraftNodeClient):
        self.client = draft_client

    def ExecuteInference(self, request, context):
        response = self.client.execute_inference(request)
        if response is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Draft inference failed")
            return speculative_decoding_pb2.InferenceJobResponse(
                request_id=request.request_id,
                status=common_pb2.STATUS_FAILED,
                error_message="Draft inference failed",
            )
        return response

    def StreamInference(self, request, context):
        final_response = None
        for event_type, data, token_events in self.client.execute_inference_stream(request):
            if event_type == "round":
                for token_event in token_events:
                    yield speculative_decoding_pb2.InferenceStreamChunk(
                        request_id=request.request_id,
                        token=common_pb2.Token(
                            token_id=int(token_event.get("token_id", 0)),
                            text=str(token_event.get("text", "")),
                            logprob=float(token_event.get("logprob", 0.0)),
                        ),
                        is_final=False,
                    )
            elif event_type == "done":
                final_response = data

        if final_response is None:
            final_response = speculative_decoding_pb2.InferenceJobResponse(
                request_id=request.request_id,
                status=common_pb2.STATUS_FAILED,
                error_message="Draft streaming failed",
            )

        yield speculative_decoding_pb2.InferenceStreamChunk(
            request_id=request.request_id,
            is_final=True,
            final_response=final_response,
        )


def serve(
    port: int = 50052,
    draft_model: str = "Qwen/Qwen3-4B-Instruct",
    num_draft_tokens: int = 5,
    verification_server: str = "127.0.0.1:50051",
    draft_node_id: str | None = None,
    draft_node_address: str | None = None,
    register_with_router: bool = True,
):
    bind_host = "0.0.0.0"
    if draft_node_address is None:
        draft_node_address = f"{_resolve_local_ip()}:{port}"

    client = DraftNodeClient(
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
        verification_server=verification_server,
        draft_node_id=draft_node_id,
        draft_node_address=draft_node_address,
        register_with_router=register_with_router,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    speculative_decoding_pb2_grpc.add_DraftNodeServiceServicer_to_server(
        DraftNodeServiceImpl(client),
        server,
    )
    server.add_insecure_port(f"{bind_host}:{port}")
    server.start()

    print(f"\n{'='*80}")
    print("📝 Draft Node Service started")
    print(f"   gRPC listen: {bind_host}:{port}")
    print(f"   Advertised address: {draft_node_address}")
    print(f"   Draft node ID: {client.draft_node_id}")
    print(f"   Draft model: {draft_model}")
    print(f"   Router: {ROUTER_HTTP_BASE}")
    print(f"{'='*80}\n")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down draft node service...")
        server.stop(0)
    finally:
        client.close()


def demo():
    print("\n" + "=" * 80)
    print("Draft Node Client - Speculative Decoding Demo")
    print("=" * 80 + "\n")

    client = DraftNodeClient(register_with_router=False)

    request = speculative_decoding_pb2.InferenceJobRequest(
        request_id=f"req-{uuid.uuid4().hex[:8]}",
        prompt="Explain speculative decoding in one paragraph.",
        params=common_pb2.InferenceParams(
            max_tokens=64,
            temperature=0.8,
            top_k=50,
            draft_tokens=5,
        ),
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        timestamp=int(time.time() * 1000),
    )

    response = client.execute_inference(request)
    if response is not None:
        print(f"Status: {common_pb2.StatusCode.Name(response.status)}")
        print(f"Acceptance: {response.acceptance_rate:.1%}")
        print(f"Output: {response.generated_text[:300]}")

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draft Node Service")
    parser.add_argument("--port", type=int, default=50052, help="Draft node gRPC port")
    parser.add_argument(
        "--draft-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct",
        help="Draft model name",
    )
    parser.add_argument("--num-draft-tokens", type=int, default=5, help="Draft tokens per round")
    parser.add_argument(
        "--verification-server",
        type=str,
        default="127.0.0.1:50051",
        help="Fallback target verification server when router lookup fails",
    )
    parser.add_argument("--draft-node-id", type=str, default=None, help="Optional draft node ID")
    parser.add_argument(
        "--draft-node-address",
        type=str,
        default=None,
        help="Address to register with router (defaults to local IP:port)",
    )
    parser.add_argument("--no-register", action="store_true", help="Disable router registration")
    parser.add_argument("--demo", action="store_true", help="Run a single local demo request")

    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        serve(
            port=args.port,
            draft_model=args.draft_model,
            num_draft_tokens=args.num_draft_tokens,
            verification_server=args.verification_server,
            draft_node_id=args.draft_node_id,
            draft_node_address=args.draft_node_address,
            register_with_router=not args.no_register,
        )
