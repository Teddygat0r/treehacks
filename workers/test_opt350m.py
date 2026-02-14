#!/usr/bin/env python3
"""Test with OPT-350m (larger draft model)"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'proto'))
import common_pb2
import speculative_decoding_pb2
from draft_node.client import DraftNodeClient

print("\n" + "="*80)
print("Testing with LARGER draft model: OPT-350m → OPT-1.3b")
print("="*80 + "\n")

client = DraftNodeClient(
    draft_model="facebook/opt-350m",  # Larger draft model
    verification_server="localhost:50051",
    num_draft_tokens=5,
)

request = speculative_decoding_pb2.InferenceJobRequest(
    request_id="opt350m-test",
    prompt="The capital of France is",
    params=common_pb2.InferenceParams(
        max_tokens=10,
        temperature=0.0,  # Greedy for testing
        top_k=-1,
        draft_tokens=5,
    ),
    model_id="facebook/opt-1.3b",
    timestamp=int(time.time() * 1000),
)

response = client.execute_inference(request)

print(f"\n{'='*80}")
print(f"RESULTS")
print(f"{'='*80}")
print(f"🎯 Acceptance: {response.acceptance_rate:.1%} ({response.draft_tokens_accepted}/{response.draft_tokens_generated})")
print(f"⚡ Speed: {response.total_tokens / (response.generation_time_ms/1000):.1f} tokens/sec")
print(f"📝 Output: {response.generated_text}")
print(f"{'='*80}\n")
