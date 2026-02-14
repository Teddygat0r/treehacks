Router server for node/client discovery.

This service is the source of truth for:
- Draft node addresses
- Target node addresses
- Frontend client addresses

## Run

```bash
python router/server.py --port 8001
```

## Core endpoints

- `POST /register/client`
- `POST /heartbeat/client`
- `POST /register/draft-node`
- `POST /heartbeat/draft-node`
- `POST /register/target-node`
- `POST /heartbeat/target-node`
- `POST /route/draft-node` (frontend asks for draft-node IP)
- `POST /route/target-node` (draft asks for target-node IP by model)
- `GET /state`
- `GET /stats`
