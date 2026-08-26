# Security Policy

## Runtime configuration

Set `Z3_API_TOKEN` to a long random secret in production. When `RAILWAY_ENVIRONMENT_NAME` is present, the API automatically requires that token for all non-public routes. You can also force this behavior locally with `Z3_REQUIRE_AUTH=true`.

Clients may send the token as `Authorization: Bearer <token>` or `X-API-Key: <token>`. The WebSocket endpoint accepts the same headers and also supports a `token` query parameter for browser clients that cannot set upgrade headers.

Set `Z3_CORS_ORIGINS` to a comma-separated allowlist of trusted origins. The default is `http://localhost:8000`; do not use a wildcard origin for a deployed service. Mutating requests are rate-limited by client address using `Z3_RATE_LIMIT_REQUESTS` within `Z3_RATE_LIMIT_WINDOW_SECONDS`.

## Deployment constraints

Run one Uvicorn worker per runtime instance. The neural state and autonomous loop are process-local, and multiple workers would maintain independent state. If horizontal scaling is required, move scheduling and state coordination behind a distributed lock before increasing worker or replica counts.

Do not expose state mutation, training, corpus ingestion, infrastructure synchronization, or conscience-outcome endpoints without authentication. Store `Z3_API_TOKEN`, database URLs, Redis URLs, Qdrant credentials, and any deployment secrets in the hosting provider’s secret manager rather than in source control.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Contact the repository maintainer privately through GitHub with a reproduction, affected endpoint or module, impact, and suggested mitigation.
