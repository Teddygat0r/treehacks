"""
Router server for registering clients/nodes and resolving node addresses.

This service is the source of truth for:
- Frontend clients (address + liveness)
- Draft nodes (address + capabilities)
- Target nodes (address + model)
"""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


HEARTBEAT_INTERVAL_MS = 10_000
LIVENESS_TTL_MS = 60_000


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

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _is_online(last_seen_ms: int) -> bool:
        return RouterState._now_ms() - last_seen_ms <= LIVENESS_TTL_MS

    def register_client(self, body: ClientRegistration) -> dict[str, Any]:
        with self._lock:
            now = self._now_ms()
            self.clients[body.client_id] = ClientEntry(
                client_id=body.client_id,
                address=body.address,
                metadata=body.metadata,
                last_seen_ms=now,
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
        with self._lock:
            now = self._now_ms()
            self.target_nodes[body.worker_id] = TargetNodeEntry(
                worker_id=body.worker_id,
                address=body.address,
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

    def route_draft_node(self, body: DraftRouteRequest) -> dict[str, Any]:
        with self._lock:
            if body.client_id and body.client_id not in self.clients:
                self.clients[body.client_id] = ClientEntry(
                    client_id=body.client_id,
                    address=body.client_address,
                    metadata="auto-registered-by-route-request",
                    last_seen_ms=self._now_ms(),
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
        with self._lock:
            candidates = [
                node for node in self.target_nodes.values() if self._is_online(node.last_seen_ms)
            ]
            if body.model_id:
                candidates = [node for node in candidates if node.model_id == body.model_id]
            if not candidates:
                return {
                    "request_id": body.request_id,
                    "worker_id": "",
                    "worker_address": "",
                    "model_id": body.model_id,
                    "model_name": "",
                    "version": "",
                    "status": "failed",
                    "message": "no online target nodes available for requested model",
                }

            selected = min(
                candidates,
                key=lambda node: (node.active_requests, node.last_seen_ms),
            )
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
            "model_id": selected.model_id,
            "model_name": selected.model_name,
            "version": selected.version,
            "status": "success",
            "message": "target node assigned",
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
