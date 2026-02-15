"""
Target Node - VerificationService on Modal
Runs the powerful model on Modal GPUs to verify draft tokens from draft nodes.
"""
import modal
import sys
import os
import threading
import time
import uuid

import requests

app = modal.App("treehacks-verification-service")

# Router defaults (overridden by Modal secret env vars).
DEFAULT_ROUTER_HTTP_BASE = "http://127.0.0.1:8001"
DEFAULT_MODAL_MODEL = os.getenv("TARGET_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm",
        "transformers",
        "numpy",
        "torch",
        "requests",
    )
    .add_local_dir(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "proto"),
        remote_path="/app/proto",
    )
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "verification_strategies.py"),
        remote_path="/app/verification_strategies.py",
    )
)


@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    min_containers=1,
    max_containers=1,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("router-config")],
)
@modal.concurrent(max_inputs=10)
class VerificationService:
    model_name: str = DEFAULT_MODAL_MODEL
    strategy: str = "deterministic"

    def _router_post(self, path: str, payload: dict) -> dict | None:
        router_http_base = os.getenv("ROUTER_HTTP_BASE", DEFAULT_ROUTER_HTTP_BASE)
        try:
            response = requests.post(f"{router_http_base}{path}", json=payload, timeout=2.0)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            print(f"Router request failed ({path}): {exc}")
            return None

    def _heartbeat_loop(self):
        interval_s = 10.0
        while not self._heartbeat_stop.wait(interval_s):
            response = self._router_post(
                "/heartbeat/target-node",
                {
                    "worker_id": self.worker_id,
                    "active_requests": self._active_requests,
                },
            )
            if response and response.get("next_heartbeat_interval_ms"):
                interval_s = max(1.0, response["next_heartbeat_interval_ms"] / 1000.0)

    def _register_with_router(self):
        if not self.register_with_router:
            return

        payload = {
            "worker_id": self.worker_id,
            "address": self.worker_address,
            "transport": "modal",
            "modal_app_name": app.name,
            "modal_class_name": "VerificationService",
            "model_id": self.model_name,
            "model_name": self.model_name,
            "version": "",
            "gpu_model": "Modal GPU",
            "gpu_memory_bytes": 0,
            "gpu_count": 1,
            "max_concurrent_requests": 32,
            "max_batch_size": 16,
        }
        response = self._router_post("/register/target-node", payload)
        if response and response.get("accepted"):
            print(
                f"Registered Modal target node with router: id={self.worker_id}, "
                f"address={self.worker_address}"
            )
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="modal-target-router-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def _send_modal_online_alert(self, event: str = "container_started"):
        if not self.register_with_router:
            return

        payload = {
            "app_name": app.name,
            "worker_id": self.worker_id,
            "address": self.worker_address,
            "event": event,
            "model_name": self.model_name,
        }
        response = self._router_post("/alerts/modal-online", payload)
        if response and response.get("accepted"):
            print(
                f"Sent Modal lifecycle alert to router: event={event}, "
                f"id={self.worker_id}"
            )

    @modal.enter()
    def setup(self):
        """Load model and tokenizer on container start."""
        from vllm import LLM
        from transformers import AutoTokenizer

        sys.path.insert(0, "/app/proto")
        sys.path.insert(0, "/app")

        print(f"Initializing verification service with model: {self.model_name}")
        print(f"Using verification strategy: {self.strategy}")

        # Optional HF auth for private/gated models.
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        llm_kwargs = {
            "model": self.model_name,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 4096,
        }
        tokenizer_kwargs = {}
        if hf_token:
            llm_kwargs["hf_token"] = hf_token
            tokenizer_kwargs["token"] = hf_token

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)

        from verification_strategies import get_strategy
        self.verification_strategy = get_strategy(self.strategy)

        self._active_requests = 0
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None
        # Default to NOT registering since Modal containers can't reach localhost router
        # Registration should be handled by a local service instead
        self.register_with_router = os.getenv("REGISTER_WITH_ROUTER", "0") not in ("0", "false", "False")
        self.worker_id = os.getenv("MODAL_TARGET_WORKER_ID", f"target-modal-{uuid.uuid4().hex[:8]}")
        self.worker_address = os.getenv(
            "MODAL_TARGET_WORKER_ADDRESS",
            f"modal://{app.name}/VerificationService/{self.worker_id}",
        )
        if self.register_with_router:
            print(f"Note: REGISTER_WITH_ROUTER is enabled. Router must be publicly accessible at: {os.getenv('ROUTER_HTTP_BASE', DEFAULT_ROUTER_HTTP_BASE)}")
        self._send_modal_online_alert(event="container_started")
        self._register_with_router()

        print("Verification service ready!")

    @modal.exit()
    def teardown(self):
        if hasattr(self, "_heartbeat_stop"):
            self._heartbeat_stop.set()
        if hasattr(self, "_heartbeat_thread") and self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)

    @modal.method()
    def ping(self, reason: str = "manual") -> dict:
        """Lightweight callable used to wake the modal target and confirm liveness."""
        self._send_modal_online_alert(event=f"ping:{reason}")
        return {
            "ok": True,
            "worker_id": self.worker_id,
            "worker_address": self.worker_address,
            "model_name": self.model_name,
            "reason": reason,
        }

    @modal.method()
    def verify_draft(
        self,
        request_id: str,
        session_id: str,
        prefix_token_ids: list[int],
        draft_token_ids: list[int],
        draft_logprobs: list[float],
        temperature: float = 0.8,
        top_k: int = -1,
    ) -> dict:
        """Verify draft tokens with the powerful model."""
        from vllm import SamplingParams

        self._active_requests += 1
        start_time = time.time()

        try:
            num_draft_tokens = len(draft_token_ids)

            if num_draft_tokens == 0:
                return {
                    "request_id": request_id,
                    "session_id": session_id,
                    "num_accepted_tokens": 0,
                    "acceptance_mask": [],
                    "corrected_token_ids": [],
                    "corrected_logprobs": [],
                    "next_token_id": 0,
                    "next_token_logprob": 0.0,
                    "verification_time_ms": 0.0,
                    "acceptance_rate": 0.0,
                }

            prefix_text = self.tokenizer.decode(prefix_token_ids, skip_special_tokens=True)

            sampling_params = SamplingParams(
                temperature=temperature if temperature > 0 else 0.8,
                top_k=top_k if top_k > 0 else -1,
                top_p=0.95,
                max_tokens=num_draft_tokens + 1,
                logprobs=5,
                seed=42,
            )

            print(f"  Verifying {num_draft_tokens} draft tokens (temp={sampling_params.temperature}, top_k={sampling_params.top_k})")

            outputs = self.llm.generate(
                prompts=[prefix_text],
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            target_output = outputs[0].outputs[0]
            target_tokens = target_output.token_ids

            num_accepted, acceptance_mask, corrected_tokens, corrected_logprobs = \
                self.verification_strategy.verify(
                    draft_token_ids=draft_token_ids,
                    draft_logprobs=draft_logprobs,
                    target_token_ids=target_tokens,
                    target_logprobs=target_output.logprobs,
                )

            next_token_id = 0
            next_token_logprob = 0.0
            if len(target_tokens) > num_accepted:
                next_token_id = target_tokens[num_accepted]
                if target_output.logprobs and num_accepted < len(target_output.logprobs):
                    token_logprobs = target_output.logprobs[num_accepted]
                    if next_token_id in token_logprobs:
                        next_token_logprob = token_logprobs[next_token_id].logprob

            acceptance_rate = num_accepted / num_draft_tokens if num_draft_tokens > 0 else 0.0
            verification_time = (time.time() - start_time) * 1000

            print(f"Verified {num_accepted}/{num_draft_tokens} tokens ({acceptance_rate:.1%}) in {verification_time:.1f}ms")

            return {
                "request_id": request_id,
                "session_id": session_id,
                "num_accepted_tokens": num_accepted,
                "acceptance_mask": acceptance_mask,
                "corrected_token_ids": corrected_tokens,
                "corrected_logprobs": corrected_logprobs,
                "next_token_id": next_token_id,
                "next_token_logprob": next_token_logprob,
                "verification_time_ms": verification_time,
                "acceptance_rate": acceptance_rate,
            }
        finally:
            self._active_requests = max(0, self._active_requests - 1)

    @modal.method()
    def verify_multi_candidate(
        self,
        request_id: str,
        session_id: str,
        prefix_token_ids: list[int],
        candidates: list[dict],
        temperature: float = 0.8,
        top_k: int = -1,
    ) -> dict:
        """Verify multiple draft candidates against a single target forward pass.

        Args:
            candidates: list of {draft_token_ids: list[int], draft_logprobs: list[float]}
        """
        from vllm import SamplingParams

        self._active_requests += 1
        start_time = time.time()
        n = len(candidates)

        try:
            if n == 0:
                return {
                    "request_id": request_id,
                    "session_id": session_id,
                    "candidate_results": [],
                    "best_candidate_idx": -1,
                    "verification_time_ms": 0.0,
                }

            # Determine max draft length across all candidates
            max_draft_len = max(len(c["draft_token_ids"]) for c in candidates)

            if max_draft_len == 0:
                return {
                    "request_id": request_id,
                    "session_id": session_id,
                    "candidate_results": [
                        {
                            "num_accepted": 0,
                            "acceptance_mask": [],
                            "corrected_token_ids": [],
                            "corrected_logprobs": [],
                            "next_token_id": 0,
                            "next_token_logprob": 0.0,
                            "target_logprob_sum": 0.0,
                        }
                        for _ in range(n)
                    ],
                    "best_candidate_idx": 0,
                    "verification_time_ms": 0.0,
                }

            # Single target model forward pass
            prefix_text = self.tokenizer.decode(prefix_token_ids, skip_special_tokens=True)

            sampling_params = SamplingParams(
                temperature=temperature if temperature > 0 else 0.8,
                top_k=top_k if top_k > 0 else -1,
                top_p=0.95,
                max_tokens=max_draft_len + 1,
                logprobs=5,
                seed=42,
            )

            print(f"  Multi-candidate verify: {n} candidates, max_draft_len={max_draft_len}")

            outputs = self.llm.generate(
                prompts=[prefix_text],
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            target_output = outputs[0].outputs[0]
            target_tokens = list(target_output.token_ids)
            target_lps = target_output.logprobs

            # Vectorized batch verification
            batch_results = self.verification_strategy.verify_batch(
                candidates=candidates,
                target_token_ids=target_tokens,
                target_logprobs=target_lps,
            )

            # Build per-candidate results and find best
            candidate_results = []
            best_idx = 0
            best_accepted = -1
            best_logprob_sum = float("-inf")

            for i, (num_accepted, acceptance_mask, corrected_tokens, corrected_logprobs) in enumerate(batch_results):
                # Next token (when all draft tokens accepted)
                next_token_id = 0
                next_token_logprob = 0.0
                if len(target_tokens) > num_accepted:
                    next_token_id = target_tokens[num_accepted]
                    if target_lps and num_accepted < len(target_lps):
                        token_logprobs = target_lps[num_accepted]
                        if next_token_id in token_logprobs:
                            next_token_logprob = token_logprobs[next_token_id].logprob

                # Compute target logprob sum for accepted tokens (for tie-breaking)
                target_logprob_sum = 0.0
                if target_lps:
                    draft_ids = candidates[i]["draft_token_ids"]
                    for j in range(num_accepted):
                        if j < len(target_lps) and j < len(draft_ids):
                            tid = draft_ids[j]
                            if tid in target_lps[j]:
                                target_logprob_sum += target_lps[j][tid].logprob

                candidate_results.append(
                    {
                        "num_accepted": num_accepted,
                        "acceptance_mask": acceptance_mask,
                        "corrected_token_ids": corrected_tokens,
                        "corrected_logprobs": corrected_logprobs,
                        "next_token_id": next_token_id,
                        "next_token_logprob": next_token_logprob,
                        "target_logprob_sum": target_logprob_sum,
                    }
                )

                # Selection: longest accepted prefix, tie-break by target logprob sum
                if (num_accepted > best_accepted) or (
                    num_accepted == best_accepted and target_logprob_sum > best_logprob_sum
                ):
                    best_accepted = num_accepted
                    best_logprob_sum = target_logprob_sum
                    best_idx = i

            verification_time = (time.time() - start_time) * 1000
            accepted_lengths = [r["num_accepted"] for r in candidate_results]
            print(
                f"  Multi-candidate result: accepted_lengths={accepted_lengths}, "
                f"best_idx={best_idx} in {verification_time:.1f}ms"
            )

            return {
                "request_id": request_id,
                "session_id": session_id,
                "candidate_results": candidate_results,
                "best_candidate_idx": best_idx,
                "verification_time_ms": verification_time,
            }
        finally:
            self._active_requests = max(0, self._active_requests - 1)

    @modal.method()
    def batch_verify(self, requests: list[dict]) -> dict:
        """Batch verification for multiple sequences."""
        self._active_requests += 1
        start_time = time.time()
        try:
            responses = []
            for req in requests:
                response = self.verify_draft(**req)
                responses.append(response)

            total_time = (time.time() - start_time) * 1000

            return {
                "responses": responses,
                "total_batch_time_ms": total_time,
            }
        finally:
            self._active_requests = max(0, self._active_requests - 1)



# --- Convenience entrypoint for direct Modal function calls (no gRPC needed) ---

@app.local_entrypoint()
def ping_modal_target(reason: str = "router_wakeup"):
    """Wake up the modal target service and force initial registration/heartbeat."""
    service = VerificationService()
    result = service.ping.remote(reason=reason)
    print(result)


@app.local_entrypoint()
def main():
    """Test the verification service via Modal's native RPC (no gRPC needed)."""
    service = VerificationService()

    # Simple test: verify some dummy tokens
    result = service.verify_draft.remote(
        request_id="test-001",
        session_id="session-0",
        prefix_token_ids=[1, 2, 3],  # dummy prefix
        draft_token_ids=[4, 5, 6],   # dummy draft
        draft_logprobs=[-0.5, -0.3, -0.8],
        temperature=0.8,
        top_k=50,
    )

    print(f"\nVerification result:")
    print(f"  Accepted: {result['num_accepted_tokens']}/{3}")
    print(f"  Acceptance rate: {result['acceptance_rate']:.1%}")
    print(f"  Time: {result['verification_time_ms']:.1f}ms")
