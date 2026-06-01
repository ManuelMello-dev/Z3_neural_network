# Z3 Neural Network

`Z3_neural_network` is a standalone neural-core implementation of the **Z³ / Z-prime** architecture. It provides a trainable PyTorch runtime where **Z³** is modeled as a persistent global observer state and **Z-prime agents** are modeled as differentiated local hypothesis states that emit evidence, novelty, coherence, and trust-weighted proposals.

The repository is intentionally focused on the core neural substrate. It is not a full application, dashboard, API server, or Cognitive Mesh clone. Its purpose is to provide a clean, testable, importable implementation that can later be consumed by a larger cognitive system.

## Core Idea

The runtime maintains a persistent global latent state, `z3_state`, and a set of local recurrent states, `zprime_state`. Each forward step encodes the current context input, projects a context-aware target from Z³ into local agent space, updates local agents through attraction, learned transition, pairwise repulsion, and noise, then emits evidence from each agent.

Novelty is computed as context-relative prediction error between emitted evidence and expected evidence. Coherence is computed as local alignment with the context-aware boot target. The model uses a differentiable gate during training and optional hard gating during runtime. Trusted local proposals are integrated back into Z³.

> **Living coherence** means the agents remain aligned enough to integrate, but differentiated enough to discover, test, and reconcile alternative hypotheses.

## Main Components

| Component | Role |
|---|---|
| `Z3Config` | Dataclass containing model dimensions, update constants, thresholds, and loss weights. |
| `Z3NeuralDynamics` | Main PyTorch module implementing Z³ / Z-prime recurrent dynamics. |
| `context_encoder` | Encodes external/context input into the runtime context space. |
| `boot_projection` | Projects global Z³ plus context into the local agent target manifold. |
| `agent_transition` | Learns local agent transition dynamics. |
| `evidence_projection` | Converts local agent states into evidence vectors. |
| `expected_evidence` | Predicts expected evidence from Z³ and context. |
| `gamma` | Converts trusted local evidence into global Z³ proposal deltas. |
| `prediction_head` | Predicts or reconstructs the target embedding for self-supervised grounding. |
| `phi_balance` loss | Penalizes excessive concentration of learned per-agent φ gain so one Z-prime agent cannot become the default winner without paying an anti-monopoly cost. |

## Installation

Install an environment-appropriate PyTorch build first. For CPU-only environments, this is typically enough:

```bash
python -m pip install -r requirements.txt
```

If you need a CUDA-enabled PyTorch build, follow the official PyTorch installation selector for your hardware and then install the remaining dependencies.

## Quick Usage

```python
import torch
from Z3_neural_dynamics import Z3NeuralDynamics

model = Z3NeuralDynamics()
x = torch.randn(4, model.config.input_dim)
output = model.forward(x, update_state=False, add_noise=False)
projection = model.public_projection(output)

print(projection["z_cubed_state"])
```

## φ Anti-Monopoly Constraint

The learned per-agent φ values remain positive attention/credibility gains, but the loss now includes an explicit concentration guard. The model normalizes φ across agents, computes a Herfindahl-style concentration score `sum(phi_weight ** 2)`, and penalizes only the portion above `phi_concentration_ratio_max / agent_count`. With the default `phi_concentration_ratio_max = 2.0`, φ can differentiate naturally, but its effective support is discouraged from collapsing below roughly half of the active agent pool.

The regularizer appears in runtime metrics as `phi_balance`, `phi_concentration`, and `phi_effective_agents`. Its strength is controlled by `beta_phi_balance`, which defaults to `0.05`.

## Boundary Calibration Upgrades

The neural core now includes three additional guardrails for the activation boundary. First, `boot_projection` is regularized with `boot_context` and `boot_variance` losses so the local target manifold remains context-sensitive instead of collapsing into a trivial Z³-only projection. The context-sensitivity check compares `B(Z³, context)` against `B(Z³, 0)`, while the variance floor discourages batch-level target collapse.

Second, the gate thresholds are context-adaptive. A bounded `threshold_adapter` maps encoded context into per-sample adjustments for `theta_novelty`, `theta_coherence`, `tau_novelty`, and `tau_coherence`. The deltas are clamped through `threshold_delta_max`, `tau_delta_max`, and `tau_min`, and `gate_rate_band` penalizes threshold settings that make the gate nearly always closed or nearly always open.

Third, the runtime maintains `rare_expert_credit`, an exponential moving average of each agent's gated coherent novelty. During trust normalization, a small `rare_expert_trust_bonus` gives historically useful but infrequent agents a residual path to contribute, reducing the risk that frequent medium-value gate passage crowds out rarer high-value insight.

## Training Smoke Test

```python
import torch
from Z3_neural_dynamics import Z3NeuralDynamics, generate_regime_sequence

model = Z3NeuralDynamics()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
x, y = generate_regime_sequence(32, model.config.input_dim, batch_size=4)
metrics = model.train_step(optimizer, x[:8], target=y[:8])
print(metrics)
```

For sequence training with bounded graph length, use `train_sequence_window()`:

```python
stream = torch.randn(2, 16, model.config.input_dim)
metrics = model.train_sequence_window(optimizer, stream, truncation_steps=4)
print(metrics["window_loss"])
```

## Validation

The repository includes a lightweight smoke test script that skips gracefully when PyTorch is not installed:

```bash
python test_z3_neural_dynamics.py
python -m py_compile Z3_neural_dynamics.py test_z3_neural_dynamics.py
```

The smoke test checks import behavior, configuration presets, shape consistency, bounded gates, normalized trust weights, zero-trust fallback, public projection compatibility, train-step mutation, sequence-window training, and checkpoint round-tripping when PyTorch is available.

## Repository Boundary

This repository should remain the canonical neural dynamics core. A larger system can import it and feed the resulting public projection into its own interface, adjudication layer, memory system, or dashboard. The neural core should produce compact runtime signals such as `phi`, `sigma`, `drift_vector`, coherence, useful novelty, and agent diversity rather than owning the entire application stack.

## Zip Inspection Note

The uploaded `Z3_neural_network-main.zip` contained only `LICENSE` and the placeholder `README.md`; it did not include additional non-core assets useful for this repository. No extra files from the zip were merged beyond preserving the existing license and replacing the placeholder README with this full documentation.

## Railway Deployment

Railway needs a long-running web process. This repository therefore includes a minimal FastAPI membrane in `main.py` while keeping `Z3_neural_dynamics.py` as the neural-core library.

The explicit Railway start command is:

```bash
python main.py
```

The repository also includes `railway.json`, which sets this command automatically for Railpack:

```json
{
  "deploy": {
    "startCommand": "python main.py",
    "healthcheckPath": "/health"
  }
}
```

The service exposes these endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | Browser dashboard interface. |
| `/api` | `GET` | Basic machine-readable service metadata. |
| `/health` | `GET` | Railway health check and dependency status. |
| `/config` | `GET` | Lazy-loads the neural model and returns config/metric names. |
| `/step` | `POST` | Runs one Z³ runtime step using a supplied input vector. |
| `/train-step` | `POST` | Runs one lightweight online train step using a supplied input vector and optional target. |

Example runtime step request:

```bash
curl -X POST "$RAILWAY_PUBLIC_DOMAIN/step" \
  -H "Content-Type: application/json" \
  -d '{"x":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
```

`/step` requires a vector length matching `input_dim`, which defaults to `16`.

## Browser Interface

The service includes a browser-based control panel at:

```text
/interface
```

The interface exposes every current endpoint from one page. It can call API metadata, health, configuration, runtime stepping, online train-step mutation, and the FastAPI OpenAPI documentation. The root endpoint `/` now serves the dashboard directly for a better mobile landing experience, while JSON metadata lives at `/api`.

| Interface Control | Backing Endpoint | Behavior |
|---|---|---|
| Dashboard Landing | `GET /` | Opens the browser interface directly. |
| API Metadata | `GET /api` | Displays service metadata and navigation links. |
| Health Check | `GET /health` | Displays dependency status, PyTorch availability, and model-load state. |
| Runtime Config | `GET /config` | Lazy-loads the model and displays configuration plus metric names. |
| Runtime Step | `POST /step` | Sends an input vector through one Z³ runtime step and displays projection, metrics, and prediction. |
| Train Step | `POST /train-step` | Runs one lightweight online training step and displays updated metrics/projection. |
| API Docs | `GET /docs` and `GET /openapi.json` | Opens the generated FastAPI schema and interactive documentation. |

In Railway, open the deployed public URL to use the control panel immediately. `/interface` remains available as an explicit alias for the same dashboard.

## Migrated Cognitive Mesh Runtime Components

This repository now includes the useful standalone pieces migrated from Cognitive Mesh without importing the whole application stack. The goal is to keep `Z3_neural_network` clean while giving it real stream learning, resonant memory, and Railway-safe persistence.

| Component | File | Purpose |
|---|---|---|
| Online world model | `world_model.py` | Converts arbitrary observation dictionaries into compact learned latent states, prediction/reconstruction losses, novelty, and nearest-memory distance. |
| Resonant memory | `resonant_memory.py` | Stores observations as phase-related memory rings with salience, anchors, phase alignment, top matches, and reconstruction confidence. |
| State persistence | `state_store.py` | Saves and loads neural checkpoints plus world-model and memory JSON state. Uses `Z3_STATE_DIR`, `RAILWAY_VOLUME_MOUNT_PATH`, `/data`, or local `data/`. |
| Integrated observe flow | `POST /observe` | Runs observation → world model → resonant memory → Z³ runtime vector → neural step or train step. |

## Expanded API Surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | `GET` | Browser dashboard interface. |
| `/api` | `GET` | Machine-readable service metadata and route map. |
| `/health` | `GET` | Health, dependency, model-load, state-load, and persistence-store status. |
| `/config` | `GET` | Z³ neural config, metric names, world-model summary, and memory summary. |
| `/step` | `POST` | Manual Z³ runtime step from a numeric vector. |
| `/train-step` | `POST` | Manual Z³ online train step from a numeric vector and optional target. |
| `/world-model` | `GET` | Current online world-model state. |
| `/world-model/observe` | `POST` | Observe one structured event with the world model only. |
| `/memory` | `GET` | Current resonant-memory snapshot. |
| `/memory/observe` | `POST` | Observe one structured event with resonant memory only. |
| `/observe` | `POST` | Integrated observe flow across world model, resonant memory, and Z³. |
| `/state` | `GET` | Persistence manifest showing checkpoint/state files. |
| `/state/save` | `POST` | Save neural, world-model, and resonant-memory state. |
| `/state/load` | `POST` | Reload neural, world-model, and resonant-memory state. |

## Integrated Observation Example

```bash
curl -X POST "$RAILWAY_PUBLIC_DOMAIN/observe" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "conversation",
    "train": false,
    "persist": true,
    "observation": {
      "content": "first live observation",
      "tone": 0.25,
      "salience": 0.7
    }
  }'
```

The response includes the generated 16-dimensional Z³ input vector, world-model losses and novelty, resonant-memory confidence and matches, and the resulting Z³ projection.

## Railway Persistence

For persistence across Railway restarts, attach a Railway volume and set:

```text
Z3_STATE_DIR=/data
```

If no volume is attached, the app still saves state to a local `data/` directory, but that storage may not survive redeploys. The `/state`, `/state/save`, and `/state/load` endpoints expose the current persistence status.

## Autonomous Runtime Loop

The runtime now includes an always-on autonomous loop that can be controlled through the dashboard or API. Each tick creates a system heartbeat observation, feeds it through the online world model and resonant memory, composes a 16-dimensional Z³ input vector, runs one online neural training step, and autosaves state on a configurable interval.

| Endpoint | Method | Purpose |
|---|---|---|
| `/runtime` | `GET` | Show loop status, tick count, last tick, errors, and recent history. |
| `/runtime/start` | `POST` | Start the background loop with `interval_seconds` and `autosave_every_ticks`. |
| `/runtime/stop` | `POST` | Stop the background loop safely. |
| `/runtime/tick` | `POST` | Run one autonomous learning tick synchronously. |

Example:

```bash
curl -X POST "$RAILWAY_PUBLIC_DOMAIN/runtime/start" \
  -H "Content-Type: application/json" \
  -d '{"interval_seconds":30,"autosave_every_ticks":5}'
```

The autonomous loop can also be tuned with environment variables:

| Variable | Default | Meaning |
|---|---:|---|
| `Z3_RUNTIME_INTERVAL` | `30` | Default seconds between background ticks. |
| `Z3_AUTOSAVE_EVERY_TICKS` | `5` | Default number of ticks between state saves. |
| `Z3_RUNTIME_LR` | `0.001` | Learning rate used by autonomous heartbeat training. |

This is the first step from a manual endpoint demo toward a continuously pulsing, memory-bearing runtime. The loop does not replace external observations; it keeps the system alive between observations by training on its own heartbeat state and preserving continuity through autosave.

## Language Training Stream

The runtime now has a real corpus ingestion stream. `language_stream.py` can read operator-supplied text from `LANGUAGE_TRAINING_CORPUS_PATH`, `LANGUAGE_TRAINING_TEXT`, or the persisted language cache, and it can also initialize a Hugging Face streaming dataset. By default, the remote corpus source is `manu/project_gutenberg` with split `en`.

When `/language/ingest`, `/chat`, or scheduled runtime language ingestion is called with `train=true`, the runtime no longer only trains on an observation latent. It routes text through `corpus_neural_ingestor.Z3CorpusNeuralIngestor`, the canonical ingestion abstraction that buffers corpus text, preserves provenance, applies rollback on failed batches, and delegates real sequence learning to `z3_language_training.train_z3_on_language_window()`. That function converts raw text into temporal language windows and trains `Z3NeuralDynamics.train_sequence_window()` directly.

The corpus ingestor is attached to the live runtime model rather than a duplicate model, so language sequence training updates the same Z³ state used by the API membrane. Runtime status exposes `language_ingestor` diagnostics when the ingestor has been initialized, and state saves include `z3_corpus_ingestor.pt` alongside the neural, world-model, and resonant-memory artifacts.

For production safety, the ingestor uses schema-versioned checkpoints, writes through a temporary file, keeps a `.previous` checkpoint, and quarantines corrupt checkpoint files before falling back to the previous copy. It also enforces maximum text size, minimum word count, optional recent-text deduplication, dropped-reason counters, training duration metrics, and a circuit breaker that pauses corpus training after repeated failures while keeping the rest of the runtime online.

| Endpoint | Method | Purpose |
|---|---|---|
| `/language` | `GET` | Show language corpus/cache/remote-stream status. |
| `/language/load` | `POST` | Load local corpus text or initialize the configured remote corpus stream. |
| `/language/fetch` | `POST` | Fetch converted language observations without ingesting them. |
| `/language/ingest` | `POST` | Feed a language batch through world model, resonant memory, and Z³; with `train=true`, also run direct corpus sequence training. |
| `/chat` | `POST` | Send one live chatbox message into Z³ as `language:chat`. |

Example:

```bash
curl -X POST "$RAILWAY_PUBLIC_DOMAIN/language/ingest" \
  -H "Content-Type: application/json" \
  -d '{"batch_size":5,"train":true,"persist":true}'
```

| Variable | Default | Meaning |
|---|---|---|
| `LANGUAGE_TRAINING_CORPUS_PATH` | empty | Path to a real text corpus file. |
| `LANGUAGE_TRAINING_TEXT` | empty | Inline language text for quick live tests. |
| `LANGUAGE_TRAINING_CACHE` | `$Z3_STATE_DIR/language_corpus.txt` | Persisted language corpus cache. |
| `LANGUAGE_TRAINING_DATASET` / `Z3_CORPUS_DATASET` | `manu/project_gutenberg` | Hugging Face dataset used when no local corpus is configured. |
| `LANGUAGE_TRAINING_DATASET_SPLIT` / `Z3_CORPUS_DATASET_SPLIT` | `en` | Dataset split for remote corpus streaming. |
| `LANGUAGE_TRAINING_TEXT_FIELD` / `Z3_CORPUS_TEXT_FIELD` | `text` | Text field to read from remote dataset rows. |
| `DISABLE_REMOTE_CORPUS_STREAM` | empty | Set to `1` to disable remote dataset streaming. |
| `LANGUAGE_TRAINING_BATCH_SIZE` | `25` | Default fetch/ingest batch size. |
| `LANGUAGE_TRAINING_MIN_WORDS` | `24` | Minimum words required before a corpus segment is emitted. |
| `Z3_CORPUS_MIN_WORDS` | `LANGUAGE_TRAINING_MIN_WORDS` | Minimum words accepted by the canonical neural ingestor before buffering. |
| `Z3_CORPUS_BUFFER_TEXTS` | `256` | Maximum buffered corpus texts retained before training. |
| `Z3_CORPUS_MAX_TRAIN_BATCHES_PER_FLUSH` | `1` | Upper bound on training batches processed during one flush. |
| `Z3_CORPUS_CHECKPOINT_PATH` | `$Z3_STATE_DIR/z3_corpus_ingestor.pt` | Corpus ingestor checkpoint containing ingestion counters, buffer/provenance metadata, and optimizer state. |
| `Z3_CORPUS_CHECKPOINT_EVERY_STEPS` | `25` | Save ingestor checkpoint every N successful language sequence updates; `0` disables automatic checkpointing. |
| `Z3_CORPUS_MAX_TEXT_BYTES` | `262144` | Reject individual corpus segments larger than this byte limit. |
| `Z3_CORPUS_DEDUPLICATE_TEXTS` | `true` | Reject recently seen duplicate text segments by content hash. |
| `Z3_CORPUS_RECENT_HASHES_LIMIT` | `4096` | Number of recent text hashes retained for deduplication. |
| `Z3_CORPUS_FAILURE_THRESHOLD` | `3` | Consecutive failed language training batches before opening the circuit breaker. |
| `Z3_CORPUS_COOLDOWN_SECONDS` | `300` | Seconds to pause language training after the circuit breaker opens. |
| `Z3_CORPUS_SLOW_TRAIN_SECONDS` | `30` | Training duration threshold that marks a batch as slow in diagnostics. |
| `Z3_CORPUS_BACKLOG_WARNING_RATIO` | `0.80` | Buffer fill ratio that marks the ingestor as backlogged. |
| `Z3_RUNTIME_LANGUAGE_ENABLED` | `true` | Enables periodic language ingestion during autonomous ticks. |
| `Z3_RUNTIME_LANGUAGE_EVERY_TICKS` | `10` | Runs language ingestion every N autonomous ticks. |
| `Z3_RUNTIME_LANGUAGE_BATCH_SIZE` | `5` | Number of language segments ingested per scheduled batch. |
| `Z3_RUNTIME_LANGUAGE_TRAIN` | `false` | If true, scheduled language ingestion trains directly on text sequence windows. |
| `Z3_RUNTIME_LANGUAGE_LR` | `0.001` | Learning rate used when scheduled language training is enabled. |
| `Z3_LANGUAGE_WINDOW_SIZE` | `24` | Token window size for direct corpus sequence training. |
| `Z3_LANGUAGE_STRIDE` | `12` | Token stride for direct corpus sequence training. |
| `Z3_LANGUAGE_TRUNCATION_STEPS` | `16` | Truncated BPTT length for direct language sequence training. |

## Railway Infrastructure Wiring

The runtime now supports optional infrastructure services around the neural core. These adapters are intentionally graceful: if a service is not configured, the app still boots and continues using local file and in-memory state. When the service variables are present, the runtime can write vectors, coordination state, and structured ledgers to the corresponding Railway service.

| Railway service | Runtime role | Environment variables |
|---|---|---|
| Volume | Durable neural checkpoint, world-model state, resonant-memory state, and language corpus cache. | `Z3_STATE_DIR=/data` with a Railway volume mounted at `/data`. |
| Qdrant | Long-term vector memory for integrated observations and language-derived latent vectors. | `QDRANT_URL`, optional `QDRANT_API_KEY`, optional `QDRANT_COLLECTION=z3_observations`. |
| Redis | Fast runtime coordination, latest observation cache, and lightweight observation stream. | `REDIS_URL` or `REDIS_PRIVATE_URL`. |
| Postgres | Durable structured observation ledger and runtime manifest history. | `DATABASE_URL`, `POSTGRES_URL`, or `POSTGRES_PRIVATE_URL`. |

The new infrastructure endpoints are:

| Endpoint | Method | Purpose |
|---|---|---|
| `/infra` | `GET` | Reports volume, Qdrant, Redis, and Postgres configuration/connectivity. |
| `/infra/sync` | `POST` | Pushes a lightweight runtime snapshot to configured Redis/Postgres backends. |

The integrated observation pathway now attempts to write each observation to the configured infrastructure backends after it passes through the world model, resonant memory, and Z³. This means `/observe`, manual language ingestion, chatbox interaction, and scheduled language ingestion can become durable across Qdrant, Redis, and Postgres when those Railway variables are connected.

