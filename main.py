"""Railway-compatible API entry point for the Z³ neural runtime.

The neural network remains a standalone library, but Railway needs a long-running
process with a web listener. This file provides that thin runtime membrane.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

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
    description="Minimal API membrane for the standalone Z³ / Z-prime neural dynamics core.",
    version="0.1.0",
)

_MODEL = None
_MODEL_LOCK = threading.Lock()


class StepRequest(BaseModel):
    """Request payload for one runtime step."""

    x: List[float] = Field(..., description="Input/context vector matching the configured input dimension.")
    hard_gate: bool = Field(True, description="Use hard inference gates instead of soft training gates.")
    update_state: bool = Field(True, description="Commit the resulting recurrent Z³/Z-prime state.")


class TrainStepRequest(BaseModel):
    """Request payload for a single lightweight online train step."""

    x: List[float] = Field(..., description="Input/context vector matching the configured input dimension.")
    target: Optional[List[float]] = Field(None, description="Optional target vector. If omitted, x is used as the reconstruction target.")
    learning_rate: float = Field(1e-3, gt=0.0, le=1.0, description="AdamW learning rate for this one train step.")


def get_model() -> Any:
    """Lazily instantiate the neural runtime after dependencies are available."""
    global _MODEL
    if IMPORT_ERROR or Z3NeuralDynamics is None:
        raise HTTPException(status_code=503, detail=f"Neural runtime dependency unavailable: {IMPORT_ERROR}")
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = Z3NeuralDynamics()
        return _MODEL


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


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "Z³ Neural Network Runtime",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "interface": "/interface",
    }


@app.get("/interface", response_class=HTMLResponse)
def interface() -> str:
    """Serve the browser-based control panel exposing every API endpoint."""
    interface_path = os.path.join(os.path.dirname(__file__), "interface.html")
    with open(interface_path, "r", encoding="utf-8") as handle:
        return handle.read()


@app.get("/health")
def health() -> Dict[str, Any]:
    model_loaded = _MODEL is not None
    return {
        "status": "ok" if IMPORT_ERROR is None else "degraded",
        "torch_available": torch is not None,
        "model_loaded": model_loaded,
        "import_error": IMPORT_ERROR,
    }


@app.get("/config")
def config() -> Dict[str, Any]:
    model = get_model()
    return {
        "config": model.config.__dict__,
        "metric_names": list(model.metric_names()),
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
    return _metrics(output, model)


@app.post("/train-step")
def train_step(request: TrainStepRequest) -> Dict[str, Any]:
    model = get_model()
    x = _tensor_from_vector(request.x, model.config.input_dim)
    target = _tensor_from_vector(request.target, model.config.input_dim) if request.target is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=request.learning_rate)
    metrics = model.train_step(optimizer, x, target=target, update_recurrent_state=True)
    return {"metrics": metrics, "projection": model.public_projection(model.step_runtime(x, hard_gate=True))}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
