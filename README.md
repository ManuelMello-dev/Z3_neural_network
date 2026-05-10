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
