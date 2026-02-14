#!/usr/bin/env python3
"""
Quick test script for the current verification strategy.
Just runs a few prompts and shows acceptance rate.
"""
import sys
import os
import time

# Add proto directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'proto'))

import common_pb2
import speculative_decoding_pb2
from draft_node.client import DraftNodeClient


def main():
    print("\n" + "="*80)
    print("🔬 Quick Verification Strategy Test")
    print("="*80 + "\n")

    # Test prompts
    prompts = [
        "The capital of France is",
        "Hello, my name is",
        "The future of AI is",
    ]

    # Connect to verification service
    print("Connecting to verification service at localhost:50051...")
    try:
        client = DraftNodeClient(
            draft_model="facebook/opt-350m",  # Using larger draft model
            verification_server="localhost:50051",
            num_draft_tokens=5,
        )
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("\nMake sure the target server is running:")
        print("  $ ./start_target_server.sh --strategy <your_strategy>")
        return

    results = []

    for i, prompt in enumerate(prompts):
        request = speculative_decoding_pb2.InferenceJobRequest(
            request_id=f"quick-test-{i}",
            prompt=prompt,
            params=common_pb2.InferenceParams(
                max_tokens=16,
                temperature=0.0,  # Greedy for best acceptance
                top_k=-1,
                draft_tokens=5,
            ),
            model_id="facebook/opt-1.3b",
            timestamp=int(time.time() * 1000),
        )

        print(f"\n[{i+1}/{len(prompts)}] Testing: '{prompt}'")
        response = client.execute_inference(request)

        results.append({
            'prompt': prompt,
            'acceptance_rate': response.acceptance_rate,
            'draft_accepted': response.draft_tokens_accepted,
            'draft_generated': response.draft_tokens_generated,
            'tokens_per_sec': response.total_tokens / (response.generation_time_ms / 1000) if response.generation_time_ms > 0 else 0,
        })

        time.sleep(0.5)

    # Summary
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)

    avg_acceptance = sum(r['acceptance_rate'] for r in results) / len(results)
    avg_speed = sum(r['tokens_per_sec'] for r in results) / len(results)
    total_accepted = sum(r['draft_accepted'] for r in results)
    total_generated = sum(r['draft_generated'] for r in results)

    print(f"\n🎯 Average Acceptance Rate: {avg_acceptance:.1%}")
    print(f"📈 Total Accepted/Generated: {total_accepted}/{total_generated}")
    print(f"⚡ Average Speed: {avg_speed:.1f} tokens/sec")

    print("\nPer-Prompt Breakdown:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. '{r['prompt'][:50]}': {r['acceptance_rate']:.1%} ({r['draft_accepted']}/{r['draft_generated']})")

    # Interpretation
    print("\n" + "="*80)
    print("💡 INTERPRETATION")
    print("="*80)
    if avg_acceptance >= 0.8:
        print("✅ EXCELLENT: >80% acceptance - speculation is very effective!")
    elif avg_acceptance >= 0.6:
        print("✓ GOOD: 60-80% acceptance - decent speedup expected")
    elif avg_acceptance >= 0.4:
        print("⚠ FAIR: 40-60% acceptance - some benefit but could be better")
    else:
        print("❌ POOR: <40% acceptance - speculation may not help much")

    print("\nNOTE: For same-family models (both OPT), we expect 70-90% with deterministic.")
    print("Lower rates suggest the models are diverging due to randomness or bugs.")
    print("\n" + "="*80 + "\n")

    # Cleanup
    del client


if __name__ == '__main__':
    main()
