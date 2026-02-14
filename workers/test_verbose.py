#!/usr/bin/env python3
"""Single prompt test with verbose output"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'proto'))
import common_pb2
import speculative_decoding_pb2
from draft_node.client import DraftNodeClient

client = DraftNodeClient(
    draft_model="facebook/opt-125m",
    verification_server="localhost:50051",
    num_draft_tokens=5,
)

request = speculative_decoding_pb2.InferenceJobRequest(
    request_id="debug-1",
    prompt="The capital of France is",
    params=common_pb2.InferenceParams(
        max_tokens=10,
        temperature=0.8,
        top_k=50,
        draft_tokens=5,
    ),
    model_id="facebook/opt-1.3b",
    timestamp=int(time.time() * 1000),
)

print("\n" + "="*80)
print("VERBOSE DEBUG TEST - Watch the server logs for token comparisons")
print("="*80 + "\n")

response = client.execute_inference(request)
print(f"\nFinal acceptance: {response.acceptance_rate:.1%}")
