#!/usr/bin/env python3
"""
Test script for Nexus GPU Network inference API
"""
import requests
import json
import sys

API_URL = "http://localhost:8000"
API_KEY = "nx_demo_test_key_12345"  # Replace with your generated API key

def test_inference():
    """Test the inference endpoint"""
    print("Testing Nexus Inference API...")
    print(f"Endpoint: {API_URL}/api/inference")
    print(f"API Key: {API_KEY[:15]}...\n")

    # Test request
    payload = {
        "prompt": "Explain quantum computing in simple terms",
        "max_tokens": 100,
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    print(f"Request payload:")
    print(json.dumps(payload, indent=2))
    print("\nSending request...\n")

    try:
        response = requests.post(
            f"{API_URL}/api/inference",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            result = response.json()

            print("✅ Success!")
            print(f"\n{'='*60}")
            print("GENERATED TEXT:")
            print(f"{'='*60}")
            print(result.get("generated_text", "No text generated"))
            print(f"{'='*60}\n")

            print("Response Details:")
            print(f"  • Request ID: {result.get('request_id')}")
            print(f"  • Total tokens: {result.get('total_tokens')}")
            print(f"  • Draft tokens generated: {result.get('draft_tokens_generated')}")
            print(f"  • Draft tokens accepted: {result.get('draft_tokens_accepted')}")
            print(f"  • Acceptance rate: {result.get('acceptance_rate', 0)*100:.1f}%")
            print(f"  • Generation time: {result.get('generation_time_ms', 0):.0f}ms")
            print(f"  • Speculation rounds: {result.get('speculation_rounds')}")

        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(response.text)
            return 1

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Could not connect to {API_URL}")
        print("Is the backend running? Start it with:")
        print("  python workers/frontend_bridge/server.py --mock")
        return 1
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>300s)")
        print("The inference is taking too long. This might happen if:")
        print("  • The backend is warming up")
        print("  • The model is loading")
        print("  • Try running in mock mode: --mock flag")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(test_inference())
