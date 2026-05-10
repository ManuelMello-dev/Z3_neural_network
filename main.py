"""Railway-compatible API entry point for the Z³ neural runtime.

The neural network remains a standalone core, while this file provides a thin
runtime membrane: dashboard, health checks, neural stepping, online training,
world-model observation, resonant memory, integrated observe→Z³ flow, and state
persistence across Railway restarts when a volume is attached.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from resonant_memory import ResonantMemoryGeometry
from runtime_loop import AutonomousRuntimeLoop, RuntimeLoopConfig
from state_store import StateStore
from world_model import OnlineWorldModel

try:  # Keep service importable enough to report dependency status.
    import torch
    from Z3_neural_dynamics import Z3NeuralDynamics
except ModuleNotFoundError as exc:  # pragma: no cover - deployment diagnostic path.
    torch = None  # type: ignore[assignment]
    Z3NeuralDynamics = None  # type: ignore[assignment]
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = None


app = FastAPI(
    title="Z³ Neural Network Runtime",
    description="Runtime membrane for the standalone Z³ / Z-prime neural dynamics core.",
    version="0.2.0",
)

_MODEL = None
_WORLD_MODEL = OnlineWorldModel(feature_dim=64, latent_dim=8)
_MEMORY = ResonantMemoryGeometry(max_rings=256, resonance_horizon=72)
_STATE_STORE = StateStore()
_STATE_LOADED = False
_RUNTIME_LOCK = threading.Lock()
_OPTIMIZER = None
_RUNTIME_LOOP = None


class StepRequest(BaseModel):
    """Request payload for one runtime step."""

    x: List[float] = Field(..., description="Input/context vector matching the configured input dimension.")
    hard_gate: bool = Field(True, description="Use hard inference gates instead of soft training gates.")
    update_state: bool = Field(True, description="Commit the resulting recurrent Z³/Z-prime state.")
    persist: bool = Field(False, description="Save runtime state after this operation.")


class TrainStepRequest(BaseModel):
    """Request payload for a single lightweight online train step."""

    x: List[float] = Field(..., description="Input/context vector matching the configured input dimension.")
    target: Optional[List[float]] = Field(None, description="Optional target vector. If omitted, x is used as reconstruction target.")
    learning_rate: float = Field(1e-3, gt=0.0, le=1.0, description="AdamW learning rate for this train step.")
    persist: bool = Field(True, description="Save runtime state after this operation.")


class ObservationRequest(BaseModel):
    """Observation payload for world-model and memory migration endpoints."""

    observation: Dict[str, Any] = Field(..., description="Arbitrary structured observation.")
    domain: str = Field("general", description="Observation domain, such as conversation, market, memory, or sensor.")
    phi_hint: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional coherence/access hint for resonant memory.")
    sigma_hint: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional noise hint for resonant memory.")
    persist: bool = Field(True, description="Save runtime state after this operation.")


class IntegratedObserveRequest(ObservationRequest):
    """Observation payload that also feeds the generated latent vector into Z³."""

    train: bool = Field(False, description="Run one training step instead of only a runtime step.")
    hard_gate: bool = Field(True, description="Use hard inference gates for runtime step.")
    learning_rate: float = Field(1e-3, gt=0.0, le=1.0, description="Learning rate when train=true.")


class RuntimeStartRequest(BaseModel):
    """Request payload for starting the autonomous runtime loop."""

    interval_seconds: float = Field(30.0, ge=1.0, le=3600.0, description="Seconds between autonomous ticks.")
    autosave_every_ticks: int = Field(5, ge=1, le=1000, description="Save state every N ticks.")


def _render_interface() -> str:
    interface_path = os.path.join(os.path.dirname(__file__), "interface.html")
    with open(interface_path, "r", encoding="utf-8") as handle:
        return handle.read()


def _status_payload() -> Dict[str, Any]:
    return {
        "service": "Z³ Neural Network Runtime",
        "status": "online",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "interface": "/interface",
        "api": "/api",
        "world_model": "/world-model",
        "memory": "/memory",
        "state": "/state",
        "runtime": "/runtime",
    }


def _require_torch() -> None:
    if IMPORT_ERROR or Z3NeuralDynamics is None or torch is None:
        raise HTTPException(status_code=503, detail=f"Neural runtime dependency unavailable: {IMPORT_ERROR}")


def ensure_runtime_loaded() -> Any:
    """Lazily instantiate and restore runtime components once."""
    global _MODEL, _STATE_LOADED
    _require_torch()
    with _RUNTIME_LOCK:
        if _MODEL is None:
            _MODEL = Z3NeuralDynamics()
        if not _STATE_LOADED:
            loaded = _STATE_STORE.load_all(model=_MODEL, model_cls=Z3NeuralDynamics, world_model=_WORLD_MODEL, memory=_MEMORY)
            _MODEL = loaded.get("model") or _MODEL
            _STATE_LOADED = True
        return _MODEL


def get_model() -> Any:
    """Return the neural runtime after lazy restore."""
    return ensure_runtime_loaded()


def _get_optimizer(model: Any, learning_rate: float) -> Any:
    global _OPTIMIZER
    if _OPTIMIZER is None or abs(float(_OPTIMIZER.param_groups[0]["lr"]) - float(learning_rate)) > 1e-12:
        _OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    return _OPTIMIZER


def _tensor_from_vector(values: List[float], expected_dim: int) -> Any:
    if len(values) != expected_dim:
        raise HTTPException(status_code=422, detail=f"Expected vector length {expected_dim}, got {len(values)}")
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def _metrics(output: Dict[str, Any], model: Any) -> Dict[str, Any]:
    projection = model.public_projection(output)
    metrics = projection["z_cubed_state"]["neural_metrics"]
    return {
        "projection": projection,
        "metrics": metrics,
        "prediction": output["prediction"].detach().cpu().squeeze(0).tolist(),
    }


def _current_phi_sigma(model: Any) -> tuple[float, float]:
    metrics = model.metrics_to_dict(model.last_metrics)
    phi = float(metrics.get("mean_coherence", 0.5) or 0.5)
    sigma = min(1.0, max(0.0, float(model.config.noise_scale) + float(metrics.get("gate_entropy", 0.5) or 0.5)))
    return max(0.0, min(1.0, phi)), sigma


def _compose_z3_input(world_output: Dict[str, Any], memory_output: Dict[str, Any], expected_dim: int) -> List[float]:
    latent = [float(v) for v in world_output.get("latent_state", [])]
    features = latent + [
        float(world_output.get("total_loss", 0.0)),
        float(world_output.get("prediction_loss", 0.0)),
        float(world_output.get("reconstruction_loss", 0.0)),
        float(world_output.get("memory_loss", 0.0)),
        float(world_output.get("coherence_alignment_loss", 0.0)),
        float(world_output.get("novelty", 0.0)),
        float(world_output.get("nearest_memory_distance", 0.0)),
        float(memory_output.get("reconstruction_confidence", 0.0)),
        float(memory_output.get("salience", 0.0)),
    ]
    if len(features) < expected_dim:
        features.extend([0.0] * (expected_dim - len(features)))
    return features[:expected_dim]


def _persist_if_requested(persist: bool) -> Optional[Dict[str, Any]]:
    if not persist:
        return None
    model = get_model()
    return _STATE_STORE.save_all(model=model, world_model=_WORLD_MODEL, memory=_MEMORY)


def _runtime_save() -> Dict[str, Any]:
    model = get_model()
    return _STATE_STORE.save_all(model=model, world_model=_WORLD_MODEL, memory=_MEMORY)


def _runtime_tick() -> Dict[str, Any]:
    """Run one autonomous heartbeat observation and learning update."""
    model = get_model()
    phi, sigma = _current_phi_sigma(model)
    observation = {
        "timestamp": time.time(),
        "source": "autonomous_runtime_loop",
        "domain": "runtime",
        "world_iteration": _WORLD_MODEL.iteration,
        "memory_rings": len(_MEMORY.rings),
        "phi": phi,
        "sigma": sigma,
        "tick_kind": "heartbeat_learning",
    }
    world_output = _WORLD_MODEL.observe(observation, domain="runtime")
    memory_output = _MEMORY.observe(
        observation,
        domain="runtime",
        phi_hint=phi,
        sigma_hint=sigma,
        constitutional_context={
            "phi": phi,
            "sigma": sigma,
            "coherence": phi,
            "drift": float(model.metrics_to_dict(model.last_metrics).get("z3_delta_norm", 0.0) or 0.0),
            "regime": "autonomous_runtime",
        },
    )
    z3_vector = _compose_z3_input(world_output.to_dict(), memory_output, model.config.input_dim)
    x = _tensor_from_vector(z3_vector, model.config.input_dim)
    optimizer = _get_optimizer(model, float(os.environ.get("Z3_RUNTIME_LR", "0.001")))
    train_metrics = model.train_step(optimizer, x, target=x, update_recurrent_state=True)
    with torch.no_grad():
        projection_output = model.forward(x, hard_gate=True, update_state=False, add_noise=False)
    return {
        "observation": observation,
        "input_vector": z3_vector,
        "world_model": world_output.to_dict(),
        "memory": memory_output,
        "z3": {
            "metrics": train_metrics,
            "projection": model.public_projection(projection_output),
        },
    }


def get_runtime_loop() -> AutonomousRuntimeLoop:
    global _RUNTIME_LOOP
    if _RUNTIME_LOOP is None:
        _RUNTIME_LOOP = AutonomousRuntimeLoop(
            tick_callback=_runtime_tick,
            save_callback=_runtime_save,
            config=RuntimeLoopConfig(
                interval_seconds=float(os.environ.get("Z3_RUNTIME_INTERVAL", "30")),
                autosave_every_ticks=int(os.environ.get("Z3_AUTOSAVE_EVERY_TICKS", "5")),
            ),
        )
    return _RUNTIME_LOOP


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Serve the browser dashboard as the default mobile-friendly landing page."""
    return _render_interface()


@app.get("/api")
def api_metadata() -> Dict[str, Any]:
    """Return machine-readable service metadata."""
    return _status_payload()


@app.get("/interface", response_class=HTMLResponse)
def interface() -> str:
    """Serve the browser-based control panel exposing every API endpoint."""
    return _render_interface()


@app.get("/health")
def health() -> Dict[str, Any]:
    model_loaded = _MODEL is not None
    return {
        "status": "ok" if IMPORT_ERROR is None else "degraded",
        "torch_available": torch is not None,
        "model_loaded": model_loaded,
        "state_loaded": _STATE_LOADED,
        "import_error": IMPORT_ERROR,
        "state_store": _STATE_STORE.manifest(),
        "runtime": get_runtime_loop().status(),
    }


@app.get("/config")
def config() -> Dict[str, Any]:
    model = get_model()
    return {
        "config": model.config.__dict__,
        "metric_names": list(model.metric_names()),
        "world_model": _WORLD_MODEL.get_state(),
        "memory": _MEMORY.get_snapshot(recent_ring_count=3)["metrics"],
    }


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    model = get_model()
    x = _tensor_from_vector(request.x, model.config.input_dim)
    with torch.no_grad():
        output = model.forward(
            x,
            hard_gate=request.hard_gate,
            update_state=request.update_state,
            add_noise=False,
        )
    response = _metrics(output, model)
    manifest = _persist_if_requested(request.persist)
    if manifest:
        response["state_manifest"] = manifest
    return response


@app.post("/train-step")
def train_step(request: TrainStepRequest) -> Dict[str, Any]:
    model = get_model()
    x = _tensor_from_vector(request.x, model.config.input_dim)
    target = _tensor_from_vector(request.target, model.config.input_dim) if request.target is not None else None
    optimizer = _get_optimizer(model, request.learning_rate)
    metrics = model.train_step(optimizer, x, target=target, update_recurrent_state=True)
    response = {"metrics": metrics, "projection": model.public_projection(model.step_runtime(x, hard_gate=True))}
    manifest = _persist_if_requested(request.persist)
    if manifest:
        response["state_manifest"] = manifest
    return response


@app.get("/world-model")
def world_model_state() -> Dict[str, Any]:
    get_model()
    return _WORLD_MODEL.get_state()


@app.post("/world-model/observe")
def world_model_observe(request: ObservationRequest) -> Dict[str, Any]:
    get_model()
    output = _WORLD_MODEL.observe(request.observation, domain=request.domain)
    response = {"world_model": output.to_dict(), "state": _WORLD_MODEL.get_state()}
    manifest = _persist_if_requested(request.persist)
    if manifest:
        response["state_manifest"] = manifest
    return response


@app.get("/memory")
def memory_state() -> Dict[str, Any]:
    get_model()
    return _MEMORY.get_snapshot()


@app.post("/memory/observe")
def memory_observe(request: ObservationRequest) -> Dict[str, Any]:
    model = get_model()
    phi, sigma = _current_phi_sigma(model)
    phi_hint = request.phi_hint if request.phi_hint is not None else phi
    sigma_hint = request.sigma_hint if request.sigma_hint is not None else sigma
    output = _MEMORY.observe(request.observation, domain=request.domain, phi_hint=phi_hint, sigma_hint=sigma_hint)
    response = {"memory": output, "snapshot": _MEMORY.get_snapshot(recent_ring_count=5)}
    manifest = _persist_if_requested(request.persist)
    if manifest:
        response["state_manifest"] = manifest
    return response


@app.post("/observe")
def integrated_observe(request: IntegratedObserveRequest) -> Dict[str, Any]:
    model = get_model()
    phi, sigma = _current_phi_sigma(model)
    phi_hint = request.phi_hint if request.phi_hint is not None else phi
    sigma_hint = request.sigma_hint if request.sigma_hint is not None else sigma

    world_output = _WORLD_MODEL.observe(request.observation, domain=request.domain)
    constitutional_context = {
        "phi": phi_hint,
        "sigma": sigma_hint,
        "coherence": phi_hint,
        "drift": float(model.metrics_to_dict(model.last_metrics).get("z3_delta_norm", 0.0) or 0.0),
        "regime": "z3_integrated_observe",
    }
    memory_output = _MEMORY.observe(
        request.observation,
        domain=request.domain,
        phi_hint=phi_hint,
        sigma_hint=sigma_hint,
        constitutional_context=constitutional_context,
    )
    z3_vector = _compose_z3_input(world_output.to_dict(), memory_output, model.config.input_dim)
    x = _tensor_from_vector(z3_vector, model.config.input_dim)

    if request.train:
        optimizer = _get_optimizer(model, request.learning_rate)
        z3_metrics = model.train_step(optimizer, x, target=x, update_recurrent_state=True)
        z3_response: Dict[str, Any] = {"metrics": z3_metrics, "projection": model.public_projection(model.step_runtime(x, hard_gate=True))}
    else:
        with torch.no_grad():
            neural_output = model.forward(x, hard_gate=request.hard_gate, update_state=True, add_noise=False)
        z3_response = _metrics(neural_output, model)

    response = {
        "input_vector": z3_vector,
        "world_model": world_output.to_dict(),
        "memory": memory_output,
        "z3": z3_response,
    }
    manifest = _persist_if_requested(request.persist)
    if manifest:
        response["state_manifest"] = manifest
    return response


@app.get("/state")
def state_manifest() -> Dict[str, Any]:
    return _STATE_STORE.manifest()


@app.post("/state/save")
def state_save() -> Dict[str, Any]:
    model = get_model()
    return _STATE_STORE.save_all(model=model, world_model=_WORLD_MODEL, memory=_MEMORY)


@app.post("/state/load")
def state_load() -> Dict[str, Any]:
    global _MODEL, _STATE_LOADED
    _require_torch()
    loaded = _STATE_STORE.load_all(model=_MODEL, model_cls=Z3NeuralDynamics, world_model=_WORLD_MODEL, memory=_MEMORY)
    _MODEL = loaded.pop("model")
    _STATE_LOADED = True
    return loaded


@app.get("/runtime")
def runtime_status() -> Dict[str, Any]:
    return get_runtime_loop().status()


@app.post("/runtime/start")
def runtime_start(request: RuntimeStartRequest) -> Dict[str, Any]:
    get_model()
    return get_runtime_loop().start(
        interval_seconds=request.interval_seconds,
        autosave_every_ticks=request.autosave_every_ticks,
    )


@app.post("/runtime/stop")
def runtime_stop() -> Dict[str, Any]:
    return get_runtime_loop().stop()


@app.post("/runtime/tick")
def runtime_tick() -> Dict[str, Any]:
    get_model()
    return get_runtime_loop().tick_once()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
