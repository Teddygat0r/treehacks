#!/usr/bin/env python3
"""
Test different verification strategies to compare acceptance rates.
Run this after starting the target server with different strategies.
"""
import subprocess
import time
import sys
import os

# Add proto directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'proto'))

import common_pb2
import speculative_decoding_pb2
from draft_node.client import DraftNodeClient


def test_strategy(strategy_name, strategy_kwargs=None, test_prompts=None):
    """Test a specific verification strategy"""
    print(f"\n{'='*100}")
    print(f"Testing Strategy: {strategy_name.upper()}")
    if strategy_kwargs:
        print(f"Parameters: {strategy_kwargs}")
    print(f"{'='*100}\n")

    if test_prompts is None:
        test_prompts = [
            "The capital of France is",
            "The president of the United States is",
            "Hello, my name is",
        ]

    # Connect to verification service
    client = DraftNodeClient(
        draft_model="facebook/opt-125m",
        verification_server="localhost:50051",
        num_draft_tokens=5,
    )

    results = []

    for i, prompt in enumerate(test_prompts):
        request = speculative_decoding_pb2.InferenceJobRequest(
            request_id=f"test-{strategy_name}-{i}",
            prompt=prompt,
            params=common_pb2.InferenceParams(
                max_tokens=16,
                temperature=0.8,
                top_k=50,
                draft_tokens=5,
            ),
            model_id="facebook/opt-1.3b",
            timestamp=int(time.time() * 1000),
        )

        response = client.execute_inference(request)

        results.append({
            'prompt': prompt,
            'acceptance_rate': response.acceptance_rate,
            'total_tokens': response.total_tokens,
            'draft_generated': response.draft_tokens_generated,
            'draft_accepted': response.draft_tokens_accepted,
            'speculation_rounds': response.speculation_rounds,
            'time_ms': response.generation_time_ms,
            'tokens_per_sec': response.total_tokens / (response.generation_time_ms / 1000) if response.generation_time_ms > 0 else 0,
        })

        time.sleep(0.5)  # Small delay between requests

    # Print summary
    print(f"\n{'='*100}")
    print(f"SUMMARY - Strategy: {strategy_name.upper()}")
    print(f"{'='*100}")

    avg_acceptance = sum(r['acceptance_rate'] for r in results) / len(results)
    avg_tokens_per_sec = sum(r['tokens_per_sec'] for r in results) / len(results)
    total_accepted = sum(r['draft_accepted'] for r in results)
    total_generated = sum(r['draft_generated'] for r in results)

    print(f"\nOverall Metrics:")
    print(f"  Average Acceptance Rate: {avg_acceptance:.1%}")
    print(f"  Total Accepted/Generated: {total_accepted}/{total_generated}")
    print(f"  Average Speed: {avg_tokens_per_sec:.1f} tokens/sec")

    print(f"\nPer-Prompt Results:")
    for r in results:
        print(f"  '{r['prompt'][:40]}':")
        print(f"    Acceptance: {r['acceptance_rate']:.1%} ({r['draft_accepted']}/{r['draft_generated']})")
        print(f"    Speed: {r['tokens_per_sec']:.1f} tok/s, Rounds: {r['speculation_rounds']}")

    print(f"\n{'='*100}\n")

    # Cleanup
    del client
    time.sleep(1)

    return {
        'strategy': strategy_name,
        'avg_acceptance': avg_acceptance,
        'avg_speed': avg_tokens_per_sec,
        'total_accepted': total_accepted,
        'total_generated': total_generated,
        'results': results,
    }


def main():
    """Run tests for all strategies"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Verification Strategy Comparison Test                    ║
║                                                                              ║
║  This script tests different verification strategies against the same       ║
║  set of prompts to compare acceptance rates and performance.                ║
║                                                                              ║
║  NOTE: You need to restart the target server with each strategy to test it. ║
║        Use the --strategy flag when starting the server.                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    if len(sys.argv) > 1:
        # Test specific strategy from command line
        strategy_name = sys.argv[1]
        print(f"Testing single strategy: {strategy_name}")
        test_strategy(strategy_name)
    else:
        # Instructions for manual testing
        print("""
Manual Testing Instructions:
============================

1. Start target server with a specific strategy:

   For DETERMINISTIC (exact match only):
   $ cd /home/dgorb/Github/treehacks/treehacks/workers
   $ source /home/dgorb/Github/treehacks/.venv/bin/activate
   $ python -c "from target_node.server import serve, VerificationServiceImpl; import grpc; from concurrent import futures; import speculative_decoding_pb2_grpc; server = grpc.server(futures.ThreadPoolExecutor(max_workers=10)); speculative_decoding_pb2_grpc.add_VerificationServiceServicer_to_server(VerificationServiceImpl(strategy='deterministic'), server); server.add_insecure_port('[::]:50051'); server.start(); print('Server ready!'); server.wait_for_termination()"

2. Run this test script:
   $ python test_strategies.py deterministic

3. Stop the server (Ctrl+C) and restart with different strategy:

   For PROBABILISTIC (SLED paper):
   $ python -c "from target_node.server import serve, VerificationServiceImpl; import grpc; from concurrent import futures; import speculative_decoding_pb2_grpc; server = grpc.server(futures.ThreadPoolExecutor(max_workers=10)); speculative_decoding_pb2_grpc.add_VerificationServiceServicer_to_server(VerificationServiceImpl(strategy='probabilistic'), server); server.add_insecure_port('[::]:50051'); server.start(); print('Server ready!'); server.wait_for_termination()"

4. Run test again:
   $ python test_strategies.py probabilistic

5. Repeat for other strategies:
   - threshold (with custom threshold):
     VerificationServiceImpl(strategy='threshold', strategy_kwargs={'threshold': 0.1})
   - greedy (always use target):
     VerificationServiceImpl(strategy='greedy')

Available Strategies:
=====================
- deterministic: Accept only exact token matches (baseline)
- probabilistic: SLED paper algorithm with α = min(1, p_target/p_draft)
- threshold: Accept if p_target(draft_token) > threshold
- greedy: Always use target token (disables speculation)

        """)


if __name__ == '__main__':
    main()
