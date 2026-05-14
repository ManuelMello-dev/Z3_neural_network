# Z³ Runtime Infrastructure Wiring

The Railway project now has the right service topology for the next stage of the Z³ runtime. The neural process should remain the live PyTorch membrane, while supporting services take over durable memory, coordination, and structured audit history.

| Service | Runtime role | Primary data |
|---|---|---|
| Volume | Checkpoint and cache substrate | `z3_neural_dynamics.pt`, world-model JSON, resonant-memory JSON, language corpus cache. |
| Qdrant | Long-term vector memory | World-model latent vectors, resonant-memory ring vectors, language observation embeddings, future conversation embeddings. |
| Redis | Fast runtime coordination | Heartbeat/tick status, recent event queue, lightweight locks, stream cursor hints. |
| Postgres | Durable structured ledger | Observations, ingestion manifests, runtime ticks, state-save manifests, endpoint/audit metadata. |

The implementation should be optional and graceful. If a service variable is missing, the runtime must keep operating from local files and in-memory data. When variables are present, `/infra` should report connection status and `/infra/sync` should push a small snapshot to each available backend.

The adapter layer will use common Railway environment variable conventions where possible:

| Adapter | Environment variables checked |
|---|---|
| Volume | `Z3_STATE_DIR`, `RAILWAY_VOLUME_MOUNT_PATH` |
| Qdrant | `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` |
| Redis | `REDIS_URL`, `REDIS_PRIVATE_URL` |
| Postgres | `DATABASE_URL`, `POSTGRES_URL`, `PGHOST`/`PGPORT`/`PGUSER`/`PGPASSWORD`/`PGDATABASE` |

The first wiring pass should not require background migrations or destructive schema operations. It should create small tables/collections only when the target is available, and it should report unavailable services instead of failing deployment.
