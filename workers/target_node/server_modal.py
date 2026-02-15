"""
Target Node - VerificationService on Modal
Runs the powerful model on Modal GPUs to verify draft tokens from draft nodes.
"""
import modal
import sys
import os

app = modal.App("treehacks-verification-service")

# Image with all dependencies - use CUDA development image
vllm_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    add_python="3.11"
).pip_install(
    "vllm",
    "transformers",
    "torch",
)

image = (
    vllm_image
    .add_local_dir(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "proto"),
        remote_path="/app/proto",
    )
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "verification_strategies.py"),
        remote_path="/app/verification_strategies.py",
    )
)


# Map Modal region codes to (lat, lng, city, country) for map display
def _region_to_location(region: str) -> dict:
    r = (region or "").lower()
    if r.startswith("us-east") or r.startswith("us-ashburn") or "eastus" in r:
        return {"lat": 38.9072, "lng": -77.0369, "city": "Washington, D.C.", "country": "USA"}
    if r.startswith("us-west-1") or (r == "us-west-1"):
        return {"lat": 37.7749, "lng": -122.4194, "city": "San Francisco", "country": "USA"}
    if r.startswith("us-west") or r.startswith("us-west1") or r.startswith("us-west2"):
        return {"lat": 45.5152, "lng": -122.6784, "city": "Portland", "country": "USA"}
    if "us-central" in r or "centralus" in r or "chicago" in r:
        return {"lat": 41.8781, "lng": -87.6298, "city": "Chicago", "country": "USA"}
    if "us-south" in r or "dallas" in r:
        return {"lat": 32.7767, "lng": -96.7970, "city": "Dallas", "country": "USA"}
    if r.startswith("eu-west-1") or "europe-west1" in r or "ireland" in r:
        return {"lat": 53.3498, "lng": -6.2603, "city": "Dublin", "country": "Ireland"}
    if "eu-central" in r or "europe-west3" in r or "frankfurt" in r:
        return {"lat": 50.1109, "lng": 8.6821, "city": "Frankfurt", "country": "Germany"}
    if "eu-west-2" in r or "europe-west2" in r or "london" in r or r == "uk":
        return {"lat": 51.5074, "lng": -0.1278, "city": "London", "country": "UK"}
    if "eu-west-3" in r or "paris" in r:
        return {"lat": 48.8566, "lng": 2.3522, "city": "Paris", "country": "France"}
    if "eu-north" in r or "stockholm" in r:
        return {"lat": 59.3293, "lng": 18.0686, "city": "Stockholm", "country": "Sweden"}
    if "eu-south" in r or "milan" in r:
        return {"lat": 45.4642, "lng": 9.1900, "city": "Milan", "country": "Italy"}
    if "europe-west4" in r or "netherlands" in r:
        return {"lat": 52.3702, "lng": 4.8952, "city": "Amsterdam", "country": "Netherlands"}
    if "ap-northeast" in r or "asia-northeast" in r or "japan" in r or r.startswith("jp"):
        return {"lat": 35.6762, "lng": 139.6503, "city": "Tokyo", "country": "Japan"}
    if "ap-southeast" in r or "asia-southeast" in r or "singapore" in r:
        return {"lat": 1.3521, "lng": 103.8198, "city": "Singapore", "country": "Singapore"}
    if "ap-south" in r or "asia-south" in r or "mumbai" in r or "india" in r:
        return {"lat": 19.0760, "lng": 72.8777, "city": "Mumbai", "country": "India"}
    if "sydney" in r or "melbourne" in r or "australia" in r or r.startswith("au"):
        return {"lat": -33.8688, "lng": 151.2093, "city": "Sydney", "country": "Australia"}
    if "seoul" in r or "korea" in r:
        return {"lat": 37.5665, "lng": 126.9780, "city": "Seoul", "country": "South Korea"}
    if "ca-" in r or "canada" in r:
        return {"lat": 43.6532, "lng": -79.3832, "city": "Toronto", "country": "Canada"}
    if "sa-" in r or "brazil" in r:
        return {"lat": -23.5505, "lng": -46.6333, "city": "São Paulo", "country": "Brazil"}
    if "me-" in r or "dubai" in r or "uae" in r:
        return {"lat": 25.2048, "lng": 55.2708, "city": "Dubai", "country": "UAE"}
    if "af" in r or "southafrica" in r:
        return {"lat": -26.2041, "lng": 28.0473, "city": "Johannesburg", "country": "South Africa"}
    return {"lat": 37.7749, "lng": -122.4194, "city": "San Francisco", "country": "USA (Modal)"}


@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=10)
class VerificationService:
    model_name: str = "Qwen/Qwen3-32B-Instruct"
    strategy: str = "deterministic"

    @modal.enter()
    def setup(self):
        """Load model and tokenizer on container start."""
        from vllm import LLM
        from transformers import AutoTokenizer

        sys.path.insert(0, "/app/proto")
        sys.path.insert(0, "/app")

        print(f"Initializing verification service with model: {self.model_name}")
        print(f"Using verification strategy: {self.strategy}")

        self.llm = LLM(
            model=self.model_name,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        from verification_strategies import get_strategy
        self.verification_strategy = get_strategy(self.strategy)

        print("Verification service ready!")

    @modal.method()
    def get_location(self) -> dict:
        """Return this worker's location from Modal region for the map."""
        region = os.environ.get("MODAL_REGION", "")
        return _region_to_location(region)

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
        import time
        from vllm import SamplingParams

        start_time = time.time()

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

    @modal.method()
    def batch_verify(self, requests: list[dict]) -> dict:
        """Batch verification for multiple sequences."""
        import time
        start_time = time.time()

        responses = []
        for req in requests:
            response = self.verify_draft(**req)
            responses.append(response)

        total_time = (time.time() - start_time) * 1000

        return {
            "responses": responses,
            "total_batch_time_ms": total_time,
        }



# --- Convenience entrypoint for direct Modal function calls (no gRPC needed) ---

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
