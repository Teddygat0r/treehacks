"""
Draft Node - Running on Modal with GPU
Generates draft tokens and coordinates with Modal verification service.
"""
import modal
import sys
import os

app = modal.App("treehacks-draft-service")

# Image with all dependencies - use vLLM's official image
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
)


# Map Modal region codes to (lat, lng, city, country) for map display
def _region_to_location(region: str) -> dict:
    r = (region or "").lower()
    # US
    if r.startswith("us-east") or r.startswith("us-ashburn") or "eastus" in r or "virginia" in r:
        return {"lat": 38.9072, "lng": -77.0369, "city": "Washington, D.C.", "country": "USA"}
    if r.startswith("us-west-1") or r == "us-west-1" or "westus" in r and "westus2" not in r:
        return {"lat": 37.7749, "lng": -122.4194, "city": "San Francisco", "country": "USA"}
    if r.startswith("us-west") or r.startswith("us-west1") or r.startswith("us-west2"):
        return {"lat": 45.5152, "lng": -122.6784, "city": "Portland", "country": "USA"}
    if "us-central" in r or "centralus" in r or "chicago" in r:
        return {"lat": 41.8781, "lng": -87.6298, "city": "Chicago", "country": "USA"}
    if "us-south" in r or "dallas" in r:
        return {"lat": 32.7767, "lng": -96.7970, "city": "Dallas", "country": "USA"}
    # EU
    if r.startswith("eu-west-1") or "europe-west1" in r or "ireland" in r:
        return {"lat": 53.3498, "lng": -6.2603, "city": "Dublin", "country": "Ireland"}
    if "eu-central" in r or "eu-west" in r and "frankfurt" in r or "europe-west3" in r:
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
    # APAC
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
    # Other
    if "ca-" in r or "canada" in r:
        return {"lat": 43.6532, "lng": -79.3832, "city": "Toronto", "country": "Canada"}
    if "sa-" in r or "brazil" in r:
        return {"lat": -23.5505, "lng": -46.6333, "city": "São Paulo", "country": "Brazil"}
    if "me-" in r or "dubai" in r or "uae" in r:
        return {"lat": 25.2048, "lng": 55.2708, "city": "Dubai", "country": "UAE"}
    if "af" in r or "southafrica" in r:
        return {"lat": -26.2041, "lng": 28.0473, "city": "Johannesburg", "country": "South Africa"}
    # Default (Modal default / unknown)
    return {"lat": 37.7749, "lng": -122.4194, "city": "San Francisco", "country": "USA (Modal)"}


@app.cls(
    image=image,
    gpu="A10G",  # Better GPU with higher compute capability
    timeout=600,
    scaledown_window=300,
)
class DraftService:
    draft_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    num_draft_tokens: int = 5

    @modal.enter()
    def setup(self):
        """Load draft model and tokenizer on container start."""
        from vllm import LLM
        from transformers import AutoTokenizer

        sys.path.insert(0, "/app/proto")

        print(f"Initializing draft service with model: {self.draft_model}")

        self.llm = LLM(
            model=self.draft_model,
            gpu_memory_utilization=0.7,
            max_model_len=4096,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.draft_model)

        # Connect to verification service
        print("Connecting to Modal verification service...")
        self.verification_service = modal.Cls.from_name(
            "treehacks-verification-service",
            "VerificationService"
        )()

        print("Draft service ready!")

    @modal.method()
    def get_location(self) -> dict:
        """Return this worker's location from Modal region for the map."""
        region = os.environ.get("MODAL_REGION", "")
        return _region_to_location(region)

    @modal.method()
    def execute_inference(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int = 16,
        temperature: float = 0.8,
        top_k: int = 50,
        draft_tokens: int = 5,
    ) -> dict:
        """
        Execute inference with speculative decoding.

        Returns a dict with:
        - generated_text: str
        - tokens: list of dicts
        - total_tokens: int
        - draft_tokens_generated: int
        - draft_tokens_accepted: int
        - generation_time_ms: float
        - acceptance_rate: float
        - speculation_rounds: int
        """
        import time
        from vllm import SamplingParams

        start_time = time.time()

        current_text = prompt
        current_token_ids = self.tokenizer.encode(current_text)
        all_tokens = []

        # Statistics
        total_draft_generated = 0
        total_draft_accepted = 0
        speculation_rounds = 0

        draft_tokens_per_round = draft_tokens if draft_tokens > 0 else self.num_draft_tokens

        # Collect all EOS token IDs
        eos_token_ids = set()
        if getattr(self.tokenizer, 'eos_token_id', None) is not None:
            eos_token_ids.add(self.tokenizer.eos_token_id)
        for token in ("<|endoftext|>", "<|im_end|>"):
            if token in self.tokenizer.get_vocab():
                eos_token_ids.add(self.tokenizer.convert_tokens_to_ids(token))

        print(f"\n{'='*80}")
        print(f"Starting inference for request: {request_id}")
        print(f"   Prompt: {prompt!r}")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Draft tokens/round: {draft_tokens_per_round}")
        print(f"{'='*80}\n")

        while len(all_tokens) < max_tokens:
            speculation_rounds += 1

            # Step 1: Generate draft tokens
            num_to_draft = min(draft_tokens_per_round, max_tokens - len(all_tokens))

            sampling_params = SamplingParams(
                temperature=temperature if temperature > 0 else 0.8,
                top_k=top_k if top_k > 0 else -1,
                top_p=0.95,
                max_tokens=num_to_draft,
                logprobs=5,
                seed=42,
            )

            draft_start = time.time()
            outputs = self.llm.generate(
                prompts=[current_text],
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            draft_output = outputs[0].outputs[0]
            draft_token_ids = draft_output.token_ids
            draft_time = (time.time() - draft_start) * 1000

            if not draft_token_ids:
                break

            total_draft_generated += len(draft_token_ids)

            # Extract draft logprobs
            draft_logprobs = []
            if draft_output.logprobs:
                for token_logprobs in draft_output.logprobs:
                    token_id = list(token_logprobs.keys())[0]
                    draft_logprobs.append(token_logprobs[token_id].logprob)

            draft_text = self.tokenizer.decode(draft_token_ids, skip_special_tokens=True)
            print(f"  Round {speculation_rounds}: Drafted {len(draft_token_ids)} tokens in {draft_time:.1f}ms")
            print(f"    Draft: {draft_text!r}")

            # Step 2: Send to Modal verification service
            try:
                verify_response = self.verification_service.verify_draft.remote(
                    request_id=request_id,
                    session_id="session-0",
                    prefix_token_ids=list(current_token_ids),
                    draft_token_ids=list(draft_token_ids),
                    draft_logprobs=list(draft_logprobs),
                    temperature=temperature if temperature > 0 else 0.8,
                    top_k=top_k if top_k > 0 else -1,
                )

                num_accepted = verify_response["num_accepted_tokens"]
                total_draft_accepted += num_accepted

                print(f"    Verified: {num_accepted}/{len(draft_token_ids)} accepted ({verify_response['acceptance_rate']:.1%})")

                # Accept the verified tokens
                accepted_tokens = draft_token_ids[:num_accepted]
                current_token_ids.extend(accepted_tokens)

                # Add corrected token if any
                if verify_response["corrected_token_ids"]:
                    current_token_ids.extend(verify_response["corrected_token_ids"])
                    print(f"    Corrected: +{len(verify_response['corrected_token_ids'])} tokens from target")

                # Add to result with proper token types
                eos_reached = False

                # Add accepted tokens
                for token_id in accepted_tokens:
                    token = {
                        "token_id": token_id,
                        "text": self.tokenizer.decode([token_id]),
                        "logprob": 0.0,
                        "type": "accepted",
                    }
                    all_tokens.append(token)
                    if eos_token_ids and token_id in eos_token_ids:
                        eos_reached = True
                        break

                # Add rejected token if any
                if not eos_reached and num_accepted < len(draft_token_ids):
                    rejected_token_id = draft_token_ids[num_accepted]
                    rejected_token = {
                        "token_id": rejected_token_id,
                        "text": self.tokenizer.decode([rejected_token_id]),
                        "logprob": 0.0,
                        "type": "rejected",
                    }
                    all_tokens.append(rejected_token)

                # Add corrected tokens
                if not eos_reached:
                    for i, token_id in enumerate(verify_response["corrected_token_ids"]):
                        corrected_logprobs = verify_response["corrected_logprobs"]
                        token = {
                            "token_id": token_id,
                            "text": self.tokenizer.decode([token_id]),
                            "logprob": corrected_logprobs[i] if i < len(corrected_logprobs) else 0.0,
                            "type": "corrected",
                        }
                        all_tokens.append(token)
                        if eos_token_ids and token_id in eos_token_ids:
                            eos_reached = True
                            break

                # Check next_token_id from target (when all draft tokens accepted)
                if not eos_reached and eos_token_ids and verify_response["next_token_id"] in eos_token_ids:
                    token = {
                        "token_id": verify_response["next_token_id"],
                        "text": self.tokenizer.decode([verify_response["next_token_id"]]),
                        "logprob": verify_response["next_token_logprob"] or 0.0,
                        "type": "accepted",
                    }
                    all_tokens.append(token)
                    current_token_ids.append(verify_response["next_token_id"])
                    eos_reached = True

                if eos_reached:
                    eos_idx = next((i for i, tid in enumerate(current_token_ids) if tid in eos_token_ids), None)
                    if eos_idx is not None:
                        current_token_ids = current_token_ids[:eos_idx + 1]
                    print(f"    EOS token reached, ending generation")
                    break

                # Update current text
                current_text = self.tokenizer.decode(current_token_ids, skip_special_tokens=True)

            except Exception as e:
                print(f"Error during verification: {e}")
                import traceback
                traceback.print_exc()
                break

        # Generate final response
        elapsed = (time.time() - start_time) * 1000
        final_text = self.tokenizer.decode(current_token_ids, skip_special_tokens=True)
        acceptance_rate = total_draft_accepted / total_draft_generated if total_draft_generated > 0 else 0.0

        print(f"\n{'='*80}")
        print(f"Inference complete!")
        print(f"   Total tokens: {len(all_tokens)}")
        print(f"   Draft generated: {total_draft_generated}")
        print(f"   Draft accepted: {total_draft_accepted} ({acceptance_rate:.1%})")
        print(f"   Speculation rounds: {speculation_rounds}")
        print(f"   Total time: {elapsed:.1f}ms ({len(all_tokens) / (elapsed/1000):.1f} tokens/sec)")
        print(f"   Result: {final_text!r}")
        print(f"{'='*80}\n")

        return {
            "request_id": request_id,
            "generated_text": final_text,
            "tokens": all_tokens,
            "total_tokens": len(all_tokens),
            "draft_tokens_generated": total_draft_generated,
            "draft_tokens_accepted": total_draft_accepted,
            "generation_time_ms": elapsed,
            "acceptance_rate": acceptance_rate,
            "speculation_rounds": speculation_rounds,
        }


@app.local_entrypoint()
def main():
    """Test the draft service with speculative decoding."""
    print("\n" + "="*80)
    print("Draft Service - Speculative Decoding Demo")
    print("="*80 + "\n")

    draft_service = DraftService()

    test_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*80}")
        print(f"Test {i+1}/{len(test_prompts)}")
        print(f"{'='*80}")

        result = draft_service.execute_inference.remote(
            request_id=f"req-{i+1}",
            prompt=prompt,
            max_tokens=16,
            temperature=0.8,
            top_k=50,
            draft_tokens=5,
        )

        print(f"\nFinal Stats:")
        print(f"   Generated: {result['generated_text']!r}")
        print(f"   Acceptance Rate: {result['acceptance_rate']:.1%}")
        print(f"   Speed: {result['total_tokens'] / (result['generation_time_ms']/1000):.1f} tokens/sec")
        print("\n" + "-"*80 + "\n")
