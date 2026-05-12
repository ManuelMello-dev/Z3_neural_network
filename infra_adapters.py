"""Optional Railway infrastructure adapters for the Z³ runtime.

The runtime must remain usable without external services. This module therefore
uses lazy optional imports and reports unavailable services instead of raising
hard deployment failures. When Railway service variables are present, it can
write observation vectors to Qdrant, runtime coordination events to Redis, and a
structured observation ledger to Postgres.
"""
from __future__ import annotations

import json
import os
import time
import uuid
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def _env_any(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


class InfrastructureHub:
    """Graceful adapter hub for Railway volume, Qdrant, Redis, and Postgres."""

    def __init__(self) -> None:
        self.qdrant_url = (_env_any("QDRANT_URL", "QDRANT_PUBLIC_URL", "QDRANT_PRIVATE_URL") or "").rstrip("/")
        self.qdrant_api_key = _env_any("QDRANT_API_KEY")
        self.qdrant_collection = os.environ.get("QDRANT_COLLECTION", "z3_observations")
        self.redis_url = _env_any("REDIS_URL", "REDIS_PRIVATE_URL")
        self.postgres_url = _env_any("DATABASE_URL", "POSTGRES_URL", "POSTGRES_PRIVATE_URL")
        self._qdrant_ready = False
        self._postgres_ready = False

    def status(self, state_manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return configuration and lightweight connectivity status."""
        return {
            "volume": {
                "configured": bool(_env_any("Z3_STATE_DIR", "RAILWAY_VOLUME_MOUNT_PATH")),
                "state_manifest": state_manifest or {},
            },
            "qdrant": self._qdrant_status(),
            "redis": self._redis_status(),
            "postgres": self._postgres_status(),
        }

    def record_observation(
        self,
        *,
        observation: Dict[str, Any],
        domain: str,
        vector: List[float],
        world_model: Dict[str, Any],
        memory: Dict[str, Any],
        z3_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Best-effort write of one observation to all configured backends."""
        observation_id = str(observation.get("entity_id") or observation.get("id") or uuid.uuid4())
        payload = {
            "id": observation_id,
            "domain": domain,
            "timestamp": time.time(),
            "observation": observation,
            "world_model": world_model,
            "memory": memory,
            "z3_metrics": z3_metrics or {},
        }
        return {
            "id": observation_id,
            "qdrant": self._qdrant_upsert(observation_id, vector, payload),
            "redis": self._redis_publish(payload),
            "postgres": self._postgres_insert_observation(observation_id, domain, vector, payload),
        }

    def sync_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Best-effort write of a runtime snapshot/manifest."""
        payload = {"timestamp": time.time(), "snapshot": snapshot}
        return {
            "redis": self._redis_set("z3:runtime:snapshot", payload),
            "postgres": self._postgres_insert_manifest(payload),
        }

    def _qdrant_status(self) -> Dict[str, Any]:
        if not self.qdrant_url:
            return {"configured": False, "ok": False, "reason": "QDRANT_URL not set"}
        result = self._qdrant_request("GET", "/collections")
        return {"configured": True, "ok": result.get("ok", False), "collection": self.qdrant_collection, "detail": result}

    def _qdrant_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.qdrant_api_key:
            headers["api-key"] = self.qdrant_api_key
        return headers

    def _qdrant_request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.qdrant_url:
            return {"ok": False, "reason": "not configured"}
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.qdrant_url}{path}",
            data=data,
            headers=self._qdrant_headers(),
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:  # noqa: S310 - user-provided service URL.
                body = response.read().decode("utf-8")
            return {"ok": True, "status": getattr(response, "status", None), "body": json.loads(body) if body else {}}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _ensure_qdrant_collection(self, vector_size: int) -> Dict[str, Any]:
        if self._qdrant_ready:
            return {"ok": True, "cached": True}
        create = self._qdrant_request(
            "PUT",
            f"/collections/{self.qdrant_collection}",
            {"vectors": {"size": int(vector_size), "distance": "Cosine"}},
        )
        # Qdrant returns success if created; if it already exists with same schema,
        # upsert may still work, so we do not fail hard here.
        self._qdrant_ready = True
        return create

    def _qdrant_upsert(self, observation_id: str, vector: List[float], payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.qdrant_url:
            return {"ok": False, "skipped": True, "reason": "QDRANT_URL not set"}
        if not vector:
            return {"ok": False, "skipped": True, "reason": "empty vector"}
        self._ensure_qdrant_collection(len(vector))
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, observation_id))
        return self._qdrant_request(
            "PUT",
            f"/collections/{self.qdrant_collection}/points?wait=true",
            {"points": [{"id": point_id, "vector": [float(v) for v in vector], "payload": payload}]},
        )

    def _redis_status(self) -> Dict[str, Any]:
        if not self.redis_url:
            return {"configured": False, "ok": False, "reason": "REDIS_URL not set"}
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(self.redis_url, socket_connect_timeout=3, socket_timeout=3)
            pong = client.ping()
            return {"configured": True, "ok": bool(pong)}
        except Exception as exc:
            return {"configured": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _redis_publish(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._redis_set("z3:observations:latest", payload, stream_key="z3:observations")

    def _redis_set(self, key: str, payload: Dict[str, Any], stream_key: Optional[str] = None) -> Dict[str, Any]:
        if not self.redis_url:
            return {"ok": False, "skipped": True, "reason": "REDIS_URL not set"}
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(self.redis_url, socket_connect_timeout=3, socket_timeout=3)
            encoded = json.dumps(payload, default=str)
            client.set(key, encoded)
            stream_id = None
            if stream_key:
                stream_id = client.xadd(stream_key, {"payload": encoded}, maxlen=1000, approximate=True)
                if isinstance(stream_id, bytes):
                    stream_id = stream_id.decode("utf-8")
            return {"ok": True, "key": key, "stream_id": stream_id}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _postgres_status(self) -> Dict[str, Any]:
        if not self.postgres_url:
            return {"configured": False, "ok": False, "reason": "DATABASE_URL/POSTGRES_URL not set"}
        try:
            import psycopg  # type: ignore

            with psycopg.connect(self.postgres_url, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute("select 1")
                    cur.fetchone()
            return {"configured": True, "ok": True}
        except Exception as exc:
            return {"configured": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _ensure_postgres_schema(self, conn: Any) -> None:
        if self._postgres_ready:
            return
        with conn.cursor() as cur:
            cur.execute(
                """
                create table if not exists z3_observations (
                    id text primary key,
                    domain text not null,
                    vector jsonb not null,
                    payload jsonb not null,
                    created_at timestamptz default now()
                )
                """
            )
            cur.execute(
                """
                create table if not exists z3_runtime_manifests (
                    id bigserial primary key,
                    payload jsonb not null,
                    created_at timestamptz default now()
                )
                """
            )
        self._postgres_ready = True

    def _postgres_insert_observation(self, observation_id: str, domain: str, vector: List[float], payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.postgres_url:
            return {"ok": False, "skipped": True, "reason": "DATABASE_URL/POSTGRES_URL not set"}
        try:
            import psycopg  # type: ignore
            from psycopg.types.json import Jsonb  # type: ignore

            with psycopg.connect(self.postgres_url, connect_timeout=5) as conn:
                self._ensure_postgres_schema(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        insert into z3_observations (id, domain, vector, payload)
                        values (%s, %s, %s, %s)
                        on conflict (id) do update set
                            domain = excluded.domain,
                            vector = excluded.vector,
                            payload = excluded.payload
                        """,
                        (observation_id, domain, Jsonb([float(v) for v in vector]), Jsonb(payload)),
                    )
                conn.commit()
            return {"ok": True, "id": observation_id}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _postgres_insert_manifest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.postgres_url:
            return {"ok": False, "skipped": True, "reason": "DATABASE_URL/POSTGRES_URL not set"}
        try:
            import psycopg  # type: ignore
            from psycopg.types.json import Jsonb  # type: ignore

            with psycopg.connect(self.postgres_url, connect_timeout=5) as conn:
                self._ensure_postgres_schema(conn)
                with conn.cursor() as cur:
                    cur.execute("insert into z3_runtime_manifests (payload) values (%s) returning id", (Jsonb(payload),))
                    row = cur.fetchone()
                conn.commit()
            return {"ok": True, "id": row[0] if row else None}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
