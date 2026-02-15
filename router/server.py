"""
Router server for registering clients/nodes and resolving node addresses.

This service is the source of truth for:
- Frontend clients (address + liveness)
- Draft nodes (address + capabilities)
- Target nodes (address + model)
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI
from pydantic import BaseModel
import requests


HEARTBEAT_INTERVAL_MS = 10_000
LIVENESS_TTL_MS = 60_000
SERVER_PING_INTERVAL_MS = 10_000
SERVER_PING_TIMEOUT_SEC = 1.0
MAX_PING_FAILURES_BEFORE_STALE = 2
MODAL_PING_ENDPOINT = os.getenv("MODAL_PING_ENDPOINT", "http://127.0.0.1:8090/ping-modal")
MODAL_PING_TIMEOUT_SEC = float(os.getenv("MODAL_PING_TIMEOUT_SEC", "1.5"))
MODAL_PING_WAIT_MS = int(os.getenv("MODAL_PING_WAIT_MS", "8000"))
MODAL_PING_POLL_MS = int(os.getenv("MODAL_PING_POLL_MS", "500"))


logger = logging.getLogger("router.server")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


@dataclass
class ClientEntry:
    client_id: str
    address: str
    metadata: str
    last_seen_ms: int


@dataclass
class DraftNodeEntry:
    draft_node_id: str
    address: str
    model_id: str
    model_name: str
    gpu_model: str
    gpu_memory_bytes: int
    max_draft_tokens: int
    available_capacity: int
    active_requests: int
    last_seen_ms: int


@dataclass
class TargetNodeEntry:
    worker_id: str
    address: str
    transport: str
    is_modal: bool
    modal_app_name: str
    modal_class_name: str
    model_id: str
    model_name: str
    version: str
    gpu_model: str
    gpu_memory_bytes: int
    gpu_count: int
    max_concurrent_requests: int
    max_batch_size: int
    active_requests: int
    last_seen_ms: int


@dataclass
class RequestAssignment:
    request_id: str
    client_id: str
    draft_node_id: str
    target_worker_id: str
    model_id: str
    created_at_ms: int


class ClientRegistration(BaseModel):
    client_id: str
    address: str
    metadata: str = ""


class ClientHeartbeat(BaseModel):
    client_id: str


class DraftNodeRegistration(BaseModel):
    draft_node_id: str
    address: str
    model_id: str = ""
    model_name: str = ""
    gpu_model: str = ""
    gpu_memory_bytes: int = 0
    max_draft_tokens: int = 0


class DraftNodeHeartbeat(BaseModel):
    draft_node_id: str
    available_capacity: int = 0
    active_requests: int = 0


class TargetNodeRegistration(BaseModel):
    worker_id: str
    address: str
    transport: str = ""
    modal_app_name: str = ""
    modal_class_name: str = ""
    model_id: str = ""
    model_name: str = ""
    version: str = ""
    gpu_model: str = ""
    gpu_memory_bytes: int = 0
    gpu_count: int = 1
    max_concurrent_requests: int = 1
    max_batch_size: int = 1


class TargetNodeHeartbeat(BaseModel):
    worker_id: str
    active_requests: int = 0


class ModalOnlineAlert(BaseModel):
    app_name: str = ""
    worker_id: str
    address: str = ""
    event: str = "container_started"
    model_name: str = ""


class DraftRouteRequest(BaseModel):
    request_id: str
    prompt: str = ""
    model_id: str = ""
    priority: int = 0
    client_id: str = ""
    client_address: str = ""


class TargetRouteRequest(BaseModel):
    request_id: str
    draft_node_id: str = ""
    model_id: str = ""


class RouterState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.clients: dict[str, ClientEntry] = {}
        self.draft_nodes: dict[str, DraftNodeEntry] = {}
        self.target_nodes: dict[str, TargetNodeEntry] = {}
        self.assignments: dict[str, RequestAssignment] = {}
        self._ping_failures: dict[str, int] = {}
        self._ping_stop = threading.Event()
        self._ping_thread: threading.Thread | None = None

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _is_online(last_seen_ms: int) -> bool:
        return RouterState._now_ms() - last_seen_ms <= LIVENESS_TTL_MS

    @staticmethod
    def _parse_host_port(address: str) -> tuple[str, int] | None:
        if not address:
            return None

        if "://" in address:
            parsed = urlparse(address)
            if parsed.scheme not in ("http", "https"):
                return None
            if parsed.hostname is None:
                return None
            if parsed.port is not None:
                return parsed.hostname, parsed.port
            return parsed.hostname, 443 if parsed.scheme == "https" else 80

        if address.startswith("[") and "]:" in address:
            host, _, port_str = address[1:].partition("]:")
            try:
                return host, int(port_str)
            except ValueError:
                return None

        if ":" not in address:
            return None

        host, port_str = address.rsplit(":", 1)
        try:
            return host, int(port_str)
        except ValueError:
            return None

    @classmethod
    def _ping_server_address(cls, address: str) -> bool:
        parsed = cls._parse_host_port(address)
        if parsed is None:
            return False
        host, port = parsed
        try:
            with socket.create_connection((host, port), timeout=SERVER_PING_TIMEOUT_SEC):
                return True
        except OSError:
            return False

    def _snapshot_server_addresses(self) -> list[tuple[str, str, str]]:
        with self._lock:
            drafts = [("draft", entry.draft_node_id, entry.address) for entry in self.draft_nodes.values()]
            targets = [
                ("target", entry.worker_id, entry.address)
                for entry in self.target_nodes.values()
                if entry.transport != "modal" and not entry.address.startswith("modal://")
            ]
        return drafts + targets

    def _apply_ping_result(self, server_type: str, server_id: str, ok: bool) -> None:
        key = f"{server_type}:{server_id}"
        with self._lock:
            now = self._now_ms()
            if server_type == "draft":
                entry = self.draft_nodes.get(server_id)
            else:
                entry = self.target_nodes.get(server_id)

            if entry is None:
                self._ping_failures.pop(key, None)
                return

            if ok:
                entry.last_seen_ms = now
                self._ping_failures[key] = 0
                return

            failures = self._ping_failures.get(key, 0) + 1
            self._ping_failures[key] = failures
            if failures >= MAX_PING_FAILURES_BEFORE_STALE:
                entry.last_seen_ms = now - LIVENESS_TTL_MS - 1

    def _ping_registered_servers_once(self) -> None:
        for server_type, server_id, address in self._snapshot_server_addresses():
            ok = self._ping_server_address(address)
            self._apply_ping_result(server_type, server_id, ok)

    def start_server_pinger(self) -> None:
        if self._ping_thread and self._ping_thread.is_alive():
            return
        self._ping_stop.clear()

        def _run() -> None:
            interval_s = SERVER_PING_INTERVAL_MS / 1000.0
            while not self._ping_stop.wait(interval_s):
                self._ping_registered_servers_once()

        self._ping_thread = threading.Thread(
            target=_run,
            name="router-server-pinger",
            daemon=True,
        )
        self._ping_thread.start()

    def stop_server_pinger(self) -> None:
        self._ping_stop.set()
        if self._ping_thread and self._ping_thread.is_alive():
            self._ping_thread.join(timeout=1.0)

    def register_client(self, body: ClientRegistration) -> dict[str, Any]:
        was_existing = False
        with self._lock:
            now = self._now_ms()
            was_existing = body.client_id in self.clients
            self.clients[body.client_id] = ClientEntry(
                client_id=body.client_id,
                address=body.address,
                metadata=body.metadata,
                last_seen_ms=now,
            )
        logger.info(
            "Client %s: id=%s address=%s metadata=%s",
            "re-registered" if was_existing else "connected",
            body.client_id,
            body.address,
            body.metadata or "-",
        )
        return {
            "accepted": True,
            "message": "client registered",
            "assigned_client_id": body.client_id,
        }

    def heartbeat_client(self, body: ClientHeartbeat) -> dict[str, Any]:
        with self._lock:
            entry = self.clients.get(body.client_id)
            if not entry:
                return {
                    "acknowledged": False,
                    "next_heartbeat_interval_ms": HEARTBEAT_INTERVAL_MS,
                    "message": "unknown client_id",
                }
            entry.last_seen_ms = self._now_ms()
        return {
            "acknowledged": True,
            "next_heartbeat_interval_ms": HEARTBEAT_INTERVAL_MS,
            "message": "ok",
        }

    def register_draft_node(self, body: DraftNodeRegistration) -> dict[str, Any]:
        with self._lock:
            now = self._now_ms()
            self.draft_nodes[body.draft_node_id] = DraftNodeEntry(
                draft_node_id=body.draft_node_id,
                address=body.address,
                model_id=body.model_id,
                model_name=body.model_name,
                gpu_model=body.gpu_model,
                gpu_memory_bytes=body.gpu_memory_bytes,
                max_draft_tokens=body.max_draft_tokens,
                available_capacity=0,
                active_requests=0,
                last_seen_ms=now,
            )
            self._ping_failures[f"draft:{body.draft_node_id}"] = 0
        return {
            "accepted": True,
            "message": "draft node registered",
            "assigned_node_id": body.draft_node_id,
        }

    def heartbeat_draft_node(self, body: DraftNodeHeartbeat) -> dict[str, Any]:
        with self._lock:
            entry = self.draft_nodes.get(body.draft_node_id)
            if not entry:
                return {
                    "acknowledged": False,
                    "next_heartbeat_interval_ms": HEARTBEAT_INTERVAL_MS,
                    "message": "unknown draft_node_id",
                }
            entry.available_capacity = body.available_capacity
            entry.active_requests = body.active_requests
            entry.last_seen_ms = self._now_ms()
        return {
            "acknowledged": True,
            "next_heartbeat_interval_ms": HEARTBEAT_INTERVAL_MS,
            "message": "ok",
        }

    def register_target_node(self, body: TargetNodeRegistration) -> dict[str, Any]:
        was_existing = False
        is_modal = False
        transport = "grpc"
        modal_app_name = ""
        modal_class_name = ""
        with self._lock:
            now = self._now_ms()
            was_existing = body.worker_id in self.target_nodes
            transport = (body.transport or "").strip().lower()
            is_modal = (
                transport == "modal"
                or body.address.startswith("modal://")
                or body.worker_id.startswith("target-modal")
            )
            if not transport:
                transport = "modal" if is_modal else "grpc"
            if is_modal and transport != "modal":
                transport = "modal"

            modal_app_name = (body.modal_app_name or "").strip()
            modal_class_name = (body.modal_class_name or "").strip()
            if is_modal and (not modal_app_name or not modal_class_name):
                parsed = body.address.removeprefix("modal://").split("/")
                if len(parsed) >= 2:
                    if not modal_app_name:
                        modal_app_name = parsed[0]
                    if not modal_class_name:
                        modal_class_name = parsed[1]

            self.target_nodes[body.worker_id] = TargetNodeEntry(
                worker_id=body.worker_id,
                address=body.address,
                transport=transport,
                is_modal=is_modal,
                modal_app_name=modal_app_name,
                modal_class_name=modal_class_name,
                model_id=body.model_id,
                model_name=body.model_name,
                version=body.version,
                gpu_model=body.gpu_model,
                gpu_memory_bytes=body.gpu_memory_bytes,
                gpu_count=body.gpu_count,
                max_concurrent_requests=body.max_concurrent_requests,
                max_batch_size=body.max_batch_size,
                active_requests=0,
                last_seen_ms=now,
            )
            self._ping_failures[f"target:{body.worker_id}"] = 0
        if is_modal:
            logger.info(
                "Modal target server %s: worker_id=%s address=%s app=%s class=%s model_id=%s model_name=%s",
                "re-registered" if was_existing else "connected",
                body.worker_id,
                body.address,
                modal_app_name or "-",
                modal_class_name or "-",
                body.model_id or "-",
                body.model_name or "-",
            )
        else:
            logger.info(
                "Target server %s: worker_id=%s address=%s transport=%s model_id=%s model_name=%s",
                "re-registered" if was_existing else "connected",
                body.worker_id,
                body.address,
                transport,
                body.model_id or "-",
                body.model_name or "-",
            )
        return {
            "accepted": True,
            "message": "target node registered",
            "assigned_worker_id": body.worker_id,
        }

    def heartbeat_target_node(self, body: TargetNodeHeartbeat) -> dict[str, Any]:
        with self._lock:
            entry = self.target_nodes.get(body.worker_id)
            if not entry:
                return {
                    "acknowledged": False,
                    "next_heartbeat_interval_ms": HEARTBEAT_INTERVAL_MS,
                    "message": "unknown worker_id",
                }
            entry.active_requests = body.active_requests
            entry.last_seen_ms = self._now_ms()
        return {
            "acknowledged": True,
            "next_heartbeat_interval_ms": HEARTBEAT_INTERVAL_MS,
            "message": "ok",
        }

    def modal_online_alert(self, body: ModalOnlineAlert) -> dict[str, Any]:
        logger.info(
            "Modal lifecycle alert: event=%s app=%s worker_id=%s address=%s model=%s",
            body.event or "container_started",
            body.app_name or "-",
            body.worker_id,
            body.address or "-",
            body.model_name or "-",
        )
        return {
            "accepted": True,
            "message": "modal alert received",
            "worker_id": body.worker_id,
        }

    @staticmethod
    def _target_matches_model(target: TargetNodeEntry, model_id: str) -> bool:
        if not model_id:
            return True
        return target.model_id == model_id

    def _select_target_locked(self, model_id: str) -> TargetNodeEntry | None:
        candidates = [
            node
            for node in self.target_nodes.values()
            if self._is_online(node.last_seen_ms) and self._target_matches_model(node, model_id)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda node: (node.active_requests, node.last_seen_ms))

    def _assign_target_locked(self, body: TargetRouteRequest, selected: TargetNodeEntry) -> dict[str, Any]:
        selected.active_requests += 1

        assignment = self.assignments.get(body.request_id)
        if assignment:
            assignment.target_worker_id = selected.worker_id
        else:
            self.assignments[body.request_id] = RequestAssignment(
                request_id=body.request_id,
                client_id="",
                draft_node_id=body.draft_node_id,
                target_worker_id=selected.worker_id,
                model_id=body.model_id,
                created_at_ms=self._now_ms(),
            )

        return {
            "request_id": body.request_id,
            "worker_id": selected.worker_id,
            "worker_address": selected.address,
            "transport": selected.transport,
            "is_modal": selected.is_modal,
            "modal_app_name": selected.modal_app_name,
            "modal_class_name": selected.modal_class_name,
            "model_id": selected.model_id,
            "model_name": selected.model_name,
            "version": selected.version,
            "status": "success",
            "message": "target node assigned",
        }

    def _trigger_modal_ping(self, body: TargetRouteRequest) -> bool:
        payload = {
            "request_id": body.request_id,
            "draft_node_id": body.draft_node_id,
            "model_id": body.model_id,
        }
        try:
            response = requests.post(
                MODAL_PING_ENDPOINT,
                json=payload,
                timeout=MODAL_PING_TIMEOUT_SEC,
            )
            if response.status_code < 500:
                logger.info(
                    "Triggered modal ping after target miss: endpoint=%s status=%s model_id=%s request_id=%s",
                    MODAL_PING_ENDPOINT,
                    response.status_code,
                    body.model_id or "-",
                    body.request_id,
                )
                return True
            logger.warning(
                "Modal ping endpoint returned server error: endpoint=%s status=%s",
                MODAL_PING_ENDPOINT,
                response.status_code,
            )
            return False
        except Exception as exc:
            logger.warning("Failed to call modal ping endpoint (%s): %s", MODAL_PING_ENDPOINT, exc)
            return False

    def route_draft_node(self, body: DraftRouteRequest) -> dict[str, Any]:
        with self._lock:
            if body.client_id and body.client_id not in self.clients:
                self.clients[body.client_id] = ClientEntry(
                    client_id=body.client_id,
                    address=body.client_address,
                    metadata="auto-registered-by-route-request",
                    last_seen_ms=self._now_ms(),
                )
                logger.info(
                    "Client connected via route request: id=%s address=%s",
                    body.client_id,
                    body.client_address or "-",
                )
            elif body.client_id:
                if body.client_address:
                    self.clients[body.client_id].address = body.client_address
                self.clients[body.client_id].last_seen_ms = self._now_ms()

            candidates = [
                node for node in self.draft_nodes.values() if self._is_online(node.last_seen_ms)
            ]
            if not candidates:
                return {
                    "request_id": body.request_id,
                    "assigned_draft_node_id": "",
                    "assigned_draft_node_address": "",
                    "status": "failed",
                    "message": "no online draft nodes registered",
                    "estimated_queue_time_ms": 0,
                }

            matching = [node for node in candidates if node.model_id == body.model_id]
            selected_pool = matching or candidates
            selected = min(
                selected_pool,
                key=lambda node: (node.active_requests, -node.available_capacity, node.last_seen_ms),
            )
            selected.active_requests += 1

            self.assignments[body.request_id] = RequestAssignment(
                request_id=body.request_id,
                client_id=body.client_id,
                draft_node_id=selected.draft_node_id,
                target_worker_id="",
                model_id=body.model_id,
                created_at_ms=self._now_ms(),
            )

        return {
            "request_id": body.request_id,
            "assigned_draft_node_id": selected.draft_node_id,
            "assigned_draft_node_address": selected.address,
            "status": "success",
            "message": "draft node assigned",
            "estimated_queue_time_ms": 0,
        }

    def route_target_node(self, body: TargetRouteRequest) -> dict[str, Any]:
        selected: TargetNodeEntry | None = None
        online_target_count = 0
        online_model_ids: list[str] = []
        with self._lock:
            selected = self._select_target_locked(body.model_id)
            if selected is not None:
                return self._assign_target_locked(body, selected)

            online_targets = [node for node in self.target_nodes.values() if self._is_online(node.last_seen_ms)]
            online_target_count = len(online_targets)
            online_model_ids = sorted({node.model_id for node in online_targets if node.model_id})

        if online_target_count > 0:
            logger.info(
                "No matching online target for model_id=%s (request_id=%s); online_targets=%s models=%s. "
                "Skipping modal wake ping.",
                body.model_id or "-",
                body.request_id,
                online_target_count,
                online_model_ids or ["-"],
            )
            return {
                "request_id": body.request_id,
                "worker_id": "",
                "worker_address": "",
                "transport": "",
                "is_modal": False,
                "modal_app_name": "",
                "modal_class_name": "",
                "model_id": body.model_id,
                "model_name": "",
                "version": "",
                "status": "failed",
                "message": "no online target nodes available for requested model_id",
            }

        # No online target node for the requested model. Trigger wakeup ping.
        ping_called = self._trigger_modal_ping(body)

        # If ping was accepted, briefly wait for a modal target to come online.
        if ping_called and MODAL_PING_WAIT_MS > 0:
            deadline = time.time() + (MODAL_PING_WAIT_MS / 1000.0)
            poll_s = max(0.1, MODAL_PING_POLL_MS / 1000.0)
            while time.time() < deadline:
                time.sleep(poll_s)
                with self._lock:
                    selected = self._select_target_locked(body.model_id)
                    if selected is not None:
                        return self._assign_target_locked(body, selected)

        return {
            "request_id": body.request_id,
            "worker_id": "",
            "worker_address": "",
            "transport": "",
            "is_modal": False,
            "modal_app_name": "",
            "modal_class_name": "",
            "model_id": body.model_id,
            "model_name": "",
            "version": "",
            "status": "failed",
            "message": "no online target nodes available for requested model",
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            clients = [
                {**asdict(entry), "online": self._is_online(entry.last_seen_ms)}
                for entry in self.clients.values()
            ]
            draft_nodes = [
                {**asdict(entry), "online": self._is_online(entry.last_seen_ms)}
                for entry in self.draft_nodes.values()
            ]
            target_nodes = [
                {**asdict(entry), "online": self._is_online(entry.last_seen_ms)}
                for entry in self.target_nodes.values()
            ]
            assignments = [asdict(entry) for entry in self.assignments.values()]

        return {
            "clients": clients,
            "draft_nodes": draft_nodes,
            "target_nodes": target_nodes,
            "assignments": assignments,
        }


router_state = RouterState()
app = FastAPI(title="SpecNet Router")


@app.on_event("startup")
def startup() -> None:
    logger.info("Router starting; server pinger enabled")
    router_state.start_server_pinger()


@app.on_event("shutdown")
def shutdown() -> None:
    logger.info("Router shutting down; stopping server pinger")
    router_state.stop_server_pinger()


@app.post("/register/client")
def register_client(body: ClientRegistration) -> dict[str, Any]:
    return router_state.register_client(body)


@app.post("/heartbeat/client")
def heartbeat_client(body: ClientHeartbeat) -> dict[str, Any]:
    return router_state.heartbeat_client(body)


@app.post("/register/draft-node")
def register_draft_node(body: DraftNodeRegistration) -> dict[str, Any]:
    return router_state.register_draft_node(body)


@app.post("/heartbeat/draft-node")
def heartbeat_draft_node(body: DraftNodeHeartbeat) -> dict[str, Any]:
    return router_state.heartbeat_draft_node(body)


@app.post("/register/target-node")
def register_target_node(body: TargetNodeRegistration) -> dict[str, Any]:
    return router_state.register_target_node(body)


@app.post("/heartbeat/target-node")
def heartbeat_target_node(body: TargetNodeHeartbeat) -> dict[str, Any]:
    return router_state.heartbeat_target_node(body)


@app.post("/alerts/modal-online")
def modal_online_alert(body: ModalOnlineAlert) -> dict[str, Any]:
    return router_state.modal_online_alert(body)


@app.post("/route/draft-node")
def route_draft_node(body: DraftRouteRequest) -> dict[str, Any]:
    return router_state.route_draft_node(body)


@app.post("/route/target-node")
def route_target_node(body: TargetRouteRequest) -> dict[str, Any]:
    return router_state.route_target_node(body)


@app.get("/state")
def state() -> dict[str, Any]:
    return router_state.state()


@app.get("/stats")
def stats() -> dict[str, Any]:
    state_view = router_state.state()
    online_drafts = sum(1 for node in state_view["draft_nodes"] if node["online"])
    online_targets = sum(1 for node in state_view["target_nodes"] if node["online"])
    return {
        "active_draft_nodes": online_drafts,
        "active_target_nodes": online_targets,
        "total_clients": sum(1 for client in state_view["clients"] if client["online"]),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description="SpecNet Router Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
