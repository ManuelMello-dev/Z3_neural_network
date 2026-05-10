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
| `/` | `GET` | Basic service metadata. |
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
