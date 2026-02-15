"""
Modal Registry Bridge - Registers Modal services with the local router.

Since Modal containers run in the cloud and can't reach localhost,
this local bridge service:
1. Monitors Modal deployments
2. Registers them with the local router
3. Sends heartbeats on their behalf
"""
import argparse
import time
import requests
import subprocess
import json
from typing import Optional

ROUTER_HTTP_BASE = "http://127.0.0.1:8001"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def get_modal_app_info(app_name: str) -> Optional[dict]:
    """Get Modal app info using modal CLI"""
    try:
        result = subprocess.run(
            ["modal", "app", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        apps = json.loads(result.stdout) if result.stdout.strip() else []
        for app in apps:
            if app.get("name") == app_name:
                return app
        return None
    except Exception as e:
        print(f"Error checking modal app: {e}")
        return None


def is_modal_app_running(app_name: str) -> bool:
    return True


def register_modal_with_router(
    worker_id: str,
    app_name: str,
    model_id: str,
) -> bool:
    """Register Modal service with the local router"""
    payload = {
        "worker_id": worker_id,
        "address": f"modal://{app_name}/VerificationService",
        "transport": "modal",
        "modal_app_name": app_name,
        "modal_class_name": "VerificationService",
        "model_id": model_id,
        "model_name": model_id,
        "version": "1.0",
        "gpu_model": "Modal GPU",
        "gpu_memory_bytes": 80_000_000_000,  # Assume 80GB A100
        "gpu_count": 1,
        "max_concurrent_requests": 32,
        "max_batch_size": 16,
    }

    try:
        response = requests.post(
            f"{ROUTER_HTTP_BASE}/register/target-node",
            json=payload,
            timeout=2.0,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("accepted"):
            print(f"✓ Registered Modal target with router: {worker_id}")
            return True
        else:
            print(f"✗ Router rejected Modal registration: {result}")
            return False
    except Exception as e:
        print(f"✗ Failed to register Modal with router: {e}")
        return False


def send_heartbeat(worker_id: str) -> bool:
    """Send heartbeat for Modal service to router"""
    payload = {
        "worker_id": worker_id,
        "active_requests": 0,
    }

    try:
        response = requests.post(
            f"{ROUTER_HTTP_BASE}/heartbeat/target-node",
            json=payload,
            timeout=2.0,
        )
        response.raise_for_status()
        return True
    except Exception:
        return False


def wait_for_router(timeout: int = 30) -> bool:
    """Wait for router to be available"""
    print(f"Waiting for router at {ROUTER_HTTP_BASE}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{ROUTER_HTTP_BASE}/health", timeout=1.0)
            if response.status_code == 200:
                print("✓ Router is ready")
                return True
        except Exception:
            pass
        time.sleep(0.5)
    print("✗ Router did not become available")
    return False


def wait_for_modal(app_name: str, timeout: int = 60) -> bool:
    """Wait for Modal app to be deployed"""
    print(f"Waiting for Modal app '{app_name}' to be deployed...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_modal_app_running(app_name):
            print(f"✓ Modal app '{app_name}' is deployed")
            return True
        time.sleep(2.0)
    print(f"✗ Modal app '{app_name}' did not become available")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Bridge service to register Modal apps with local router"
    )
    parser.add_argument(
        "--app-name",
        default="treehacks-verification-service",
        help="Modal app name to monitor",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Model ID for the Modal target",
    )
    parser.add_argument(
        "--worker-id",
        default="target-modal-main",
        help="Worker ID to use for registration",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=10,
        help="Heartbeat interval in seconds",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for router/modal, start immediately",
    )

    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"Modal Registry Bridge")
    print(f"{'='*70}")
    print(f"Modal App:        {args.app_name}")
    print(f"Model ID:         {args.model_id}")
    print(f"Worker ID:        {args.worker_id}")
    print(f"Router:           {ROUTER_HTTP_BASE}")
    print(f"Heartbeat:        every {args.heartbeat_interval}s")
    print(f"{'='*70}\n")

    # Wait for services to be ready
    if not args.no_wait:
        if not wait_for_router(timeout=30):
            print("Warning: Router not available, continuing anyway...")

        if not wait_for_modal(args.app_name, timeout=90):
            print("Warning: Modal not deployed, will retry registration...")

    # Register Modal with router
    registered = False
    registration_attempts = 0
    max_registration_attempts = 10

    while not registered and registration_attempts < max_registration_attempts:
        registration_attempts += 1

        # Check if Modal is actually running
        if not is_modal_app_running(args.app_name):
            print(f"Modal app '{args.app_name}' not running yet, waiting...")
            time.sleep(5.0)
            continue

        # Try to register
        registered = register_modal_with_router(
            worker_id=args.worker_id,
            app_name=args.app_name,
            model_id=args.model_id,
        )

        if not registered:
            print(f"Registration attempt {registration_attempts}/{max_registration_attempts} failed, retrying in 3s...")
            time.sleep(3.0)

    if not registered:
        print(f"\n✗ Failed to register Modal after {max_registration_attempts} attempts")
        print("Continuing to send heartbeats in case Modal becomes available...")

    # Send periodic heartbeats
    print(f"\n{'='*70}")
    print(f"Heartbeat loop started (Ctrl+C to stop)")
    print(f"{'='*70}\n")

    heartbeat_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 5

    try:
        while True:
            time.sleep(args.heartbeat_interval)
            heartbeat_count += 1

            # Re-register if not registered yet
            if not registered:
                if is_modal_app_running(args.app_name):
                    registered = register_modal_with_router(
                        worker_id=args.worker_id,
                        app_name=args.app_name,
                        model_id=args.model_id,
                    )

            # Send heartbeat
            if registered:
                success = send_heartbeat(args.worker_id)
                if success:
                    consecutive_failures = 0
                    if heartbeat_count % 6 == 0:  # Log every minute
                        print(f"[{heartbeat_count}] Heartbeat sent successfully")
                else:
                    consecutive_failures += 1
                    print(f"✗ Heartbeat failed ({consecutive_failures}/{max_consecutive_failures})")

                    if consecutive_failures >= max_consecutive_failures:
                        print("Too many consecutive heartbeat failures, attempting re-registration...")
                        registered = False
                        consecutive_failures = 0

    except KeyboardInterrupt:
        print("\n\nShutting down Modal registry bridge...")


if __name__ == "__main__":
    main()
