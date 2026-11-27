"""Railway-compatible API entry point for the Z³ neural runtime.

The neural network remains a standalone core, while this file provides a thin
runtime membrane: dashboard, health checks, neural stepping, online training,
world-model observation, resonant memory, integrated observe→Z³ flow, and state
persistence across Railway restarts when a volume is attached.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import secrets
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import replace
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("z3.runtime")

from language_stream import LanguageStream
from curriculum_stream import CurriculumStream
from corpus_neural_ingestor import Z3CorpusIngestionConfig, Z3CorpusNeuralIngestor
from z3_language_training import train_z3_on_language_window
from z3_audio_synth import get_synthesizer, get_ws_manager
from z3_language_decoder import (
    get_decoder, get_tokenizer, generate as z3_generate,
    train_language_step, save_decoder, load_decoder,
)
try:
    from z3_articulatory_synth import get_articulatory_voice, TrajectoryControlLayer
except ImportError as exc:  # Keep the API importable for dependency diagnostics.
    get_articulatory_voice = None  # type: ignore[assignment]
    TrajectoryControlLayer = None  # type: ignore[assignment]
    ARTICULATORY_IMPORT_ERROR = str(exc)
else:
    ARTICULATORY_IMPORT_ERROR = None
try:
    from z3_acoustic_encoder import get_acoustic_encoder
except ImportError as exc:  # Keep the API importable for dependency diagnostics.
    get_acoustic_encoder = None  # type: ignore[assignment]
    ACOUSTIC_IMPORT_ERROR = str(exc)
else:
    ACOUSTIC_IMPORT_ERROR = None
from infra_adapters import InfrastructureHub
from resonant_memory import ResonantMemoryGeometry
from response_adapter import build_z3_expression
from conscience import ConsciencePipeline
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


@contextlib.asynccontextmanager
async def _lifespan(app_: Any):
    """FastAPI lifespan: start audio synth on boot, save + stop on shutdown."""
    # --- startup ---
    global _MAIN_LOOP
    _MAIN_LOOP = asyncio.get_running_loop()
    try:
        agent_count = 8
        if _MODEL is not None:
            agent_count = getattr(getattr(_MODEL, "config", None), "agent_count", 8)
        synth = get_synthesizer(agent_count)
        synth.ws_manager = get_ws_manager()
        synth.load_weights()
        synth.start()
        print(f"[Z³ Audio] Synthesizer started with {agent_count} agents.")
    except Exception as exc:
        print(f"[Z³ Audio] Synthesizer startup skipped: {exc}")
    # Start language decoder
    try:
        cfg = _MODEL.config if _MODEL is not None else None
        dec = get_decoder(
            input_dim=getattr(cfg, 'input_dim', 16),
            evidence_dim=getattr(cfg, 'evidence_dim', 24),
            state_dim=getattr(cfg, 'state_dim', 64),
            context_dim=getattr(cfg, 'context_dim', 48),
        )
        load_decoder(dec)
        print("[Z³ Decoder] Language decoder ready.")
    except Exception as exc:
        print(f"[Z³ Decoder] Startup skipped: {exc}")
    # Start articulatory voice
    if get_articulatory_voice is not None:
        try:
            cfg = _MODEL.config if _MODEL is not None else None
            get_articulatory_voice(
                z3_prediction_dim=getattr(cfg, 'evidence_dim', 24)
                                  + getattr(cfg, 'state_dim', 64)
                                  + getattr(cfg, 'context_dim', 48),
            )
            print("[Z³ Voice] Articulatory voice module ready.")
        except Exception as exc:
            print(f"[Z³ Voice] Startup skipped: {exc}")
    else:
        print(f"[Z³ Voice] Startup skipped: {ARTICULATORY_IMPORT_ERROR}")
    yield
    # --- shutdown ---
    try:
        synth = get_synthesizer()
        synth.stop()
        synth.save_weights()
    except Exception as exc:
        print(f"[Z³ Audio] Shutdown save skipped: {exc}")
    try:
        save_decoder(get_decoder())
    except Exception as exc:
        print(f"[Z³ Decoder] Shutdown save skipped: {exc}")


app = FastAPI(
    title="Z³ Neural Network Runtime",
    description="Runtime membrane for the standalone Z³ / Z-prime neural dynamics core.",
    version="0.2.0",
    lifespan=_lifespan,
)

_PUBLIC_PATHS = {"/", "/interface", "/api", "/health", "/docs", "/openapi.json", "/redoc"}
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_API_TOKEN = os.environ.get("Z3_API_TOKEN", "")
_AUTH_REQUIRED = os.environ.get("Z3_REQUIRE_AUTH", "").lower() in {"1", "true", "yes", "on"} or bool(os.environ.get("RAILWAY_ENVIRONMENT_NAME"))
_RATE_LIMIT_WINDOW = max(1, int(os.environ.get("Z3_RATE_LIMIT_WINDOW_SECONDS", "60")))
_RATE_LIMIT_REQUESTS = max(1, int(os.environ.get("Z3_RATE_LIMIT_REQUESTS", "120")))
_RATE_LIMIT_STATE: dict[str, deque[float]] = defaultdict(deque)
_RATE_LIMIT_LOCK = threading.Lock()


class RuntimeSecurityMiddleware(BaseHTTPMiddleware):
    """Protect non-public routes and bound request volume in production."""

    @staticmethod
    def _provided_token(request: Any) -> str:
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return request.headers.get("x-api-key", "").strip()

    @staticmethod
    def _client_key(request: Any) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        return forwarded.split(",", 1)[0].strip() if forwarded else (request.client.host if request.client else "unknown")

    async def dispatch(self, request: Any, call_next: Any) -> Any:
        path = request.url.path
        if _AUTH_REQUIRED and request.method != "OPTIONS" and path not in _PUBLIC_PATHS:
            if not _API_TOKEN or not secrets.compare_digest(self._provided_token(request), _API_TOKEN):
                return JSONResponse({"detail": "Authentication required"}, status_code=401)
        if request.method in _MUTATING_METHODS:
            key = self._client_key(request)
            now = time.monotonic()
            with _RATE_LIMIT_LOCK:
                timestamps = _RATE_LIMIT_STATE[key]
                while timestamps and now - timestamps[0] >= _RATE_LIMIT_WINDOW:
                    timestamps.popleft()
                if len(timestamps) >= _RATE_LIMIT_REQUESTS:
                    return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429, headers={"Retry-After": str(_RATE_LIMIT_WINDOW)})
                timestamps.append(now)
        return await call_next(request)


app.add_middleware(RuntimeSecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.environ.get("Z3_CORS_ORIGINS", "http://localhost:8000").split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

_MODEL = None
_WORLD_MODEL = OnlineWorldModel(feature_dim=64, latent_dim=8)
_MEMORY = ResonantMemoryGeometry(max_rings=256, resonance_horizon=72)
_CONSCIENCE = ConsciencePipeline()
_LANGUAGE_STREAM = LanguageStream()
_CURRICULUM_STREAM = CurriculumStream()
_INFRA = InfrastructureHub()
_STATE_STORE = StateStore()
_STATE_LOADED = False
# Re-entrant: integrated_observe is reached both directly and from inside other
# locked paths (runtime tick -> language/curriculum ingest -> observe).
_RUNTIME_LOCK = threading.RLock()
# Event loop that owns WebSocket connections; captured in lifespan so sync
# endpoints (threadpool) and the runtime thread can schedule broadcasts.
_MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None
_OPTIMIZER = None
_OPTIMIZER_MODEL_ID: Optional[int] = None
_LANGUAGE_INGESTOR = None
# What the last boot / state load actually restored (surfaced on /runtime and /state).
_BOOT_RESTORE: Dict[str, Any] = {}
_RUNTIME_LOOP = None
_RUNTIME_TICK_SEQUENCE = 0
_RUNTIME_LANGUAGE_CONFIG: Dict[str, Any] = {
    "enabled": os.environ.get("Z3_RUNTIME_LANGUAGE_ENABLED", "true").lower() in ("1", "true", "yes", "on"),
    "every_ticks": int(os.environ.get("Z3_RUNTIME_LANGUAGE_EVERY_TICKS", "10")),
    "batch_size": int(os.environ.get("Z3_RUNTIME_LANGUAGE_BATCH_SIZE", "5")),
    "train": os.environ.get("Z3_RUNTIME_LANGUAGE_TRAIN", "false").lower() in ("1", "true", "yes", "on"),
    "learning_rate": float(os.environ.get("Z3_RUNTIME_LANGUAGE_LR", "0.001")),
    "window_size": int(os.environ.get("Z3_LANGUAGE_WINDOW_SIZE", "24")),
    "stride": int(os.environ.get("Z3_LANGUAGE_STRIDE", "12")),
    "truncation_steps": int(os.environ.get("Z3_LANGUAGE_TRUNCATION_STEPS", "16")),
}


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
    language_enabled: bool = Field(True, description="Periodically ingest language batches during autonomous ticks.")
    language_every_ticks: int = Field(10, ge=1, le=10000, description="Run language ingestion every N autonomous ticks when enabled.")
    language_batch_size: int = Field(5, ge=1, le=250, description="Language segments to ingest per scheduled batch.")
    language_train: bool = Field(False, description="Train Z³ on each scheduled language segment instead of runtime stepping only.")
    language_learning_rate: float = Field(1e-3, gt=0.0, le=1.0, description="Learning rate for scheduled language training when language_train=true.")


class LanguageBatchRequest(BaseModel):
    """Request payload for language fetch and ingest operations."""

    batch_size: int = Field(5, ge=1, le=250, description="Number of language segments to fetch or ingest.")
    train: bool = Field(False, description="Train Z³ on each ingested language segment instead of runtime stepping only.")
    persist: bool = Field(True, description="Save runtime state after ingestion.")
    learning_rate: float = Field(1e-3, gt=0.0, le=1.0, description="Learning rate when train=true.")


class CurriculumBatchRequest(BaseModel):
    """Request payload for layered curriculum fetch and ingest operations."""

    batch_size: int = Field(6, ge=1, le=250, description="Number of curriculum observations to fetch or ingest.")
    source: Optional[str] = Field(None, description="Optional curriculum source kind: language, dialogue, contradiction, event, anomaly, or identity.")
    train: bool = Field(False, description="Run one online Z³ train step for each ingested curriculum observation.")
    persist: bool = Field(True, description="Save runtime state after ingestion.")
    learning_rate: float = Field(1e-3, gt=0.0, le=1.0, description="Learning rate when train=true.")


class ChatRequest(BaseModel):
    """Request payload for chatbox language interaction and testing."""

    message: str = Field(..., min_length=1, description="Language message to observe through Z³.")
    train: bool = Field(False, description="Train Z³ on the message instead of runtime stepping only.")
    persist: bool = Field(True, description="Save runtime state after chat observation.")
    learning_rate: float = Field(1e-3, gt=0.0, le=1.0, description="Learning rate when train=true.")


class ConscienceEvaluateRequest(BaseModel):
    """Request payload for direct production conscience evaluation."""

    proposal: Dict[str, Any] | str = Field(..., description="Structured proposal or text/action to evaluate.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Optional world-model, memory, Z³, stakeholder, and constraint context.")
    persist: bool = Field(False, description="Save runtime state after evaluation.")


class ConscienceOutcomeRequest(BaseModel):
    """Outcome feedback for conscience memory calibration."""

    outcome_value: float = Field(..., ge=-1.0, le=1.0, description="Observed outcome valence for the last conscience result.")
    salience: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional outcome salience override.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional outcome confidence override.")
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Who/what observed or labeled this outcome.")
    notes: str = Field("", description="Auditable notes about the outcome.")
    persist: bool = Field(True, description="Save runtime state after learning the outcome.")


def _render_interface() -> str:
    """Load the dashboard from a checkout or an installed distribution."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "interface.html"),
        os.path.join(sys.prefix, "share", "z3-neural-network", "interface.html"),
    ]
    for interface_path in candidates:
        if os.path.isfile(interface_path):
            with open(interface_path, "r", encoding="utf-8") as handle:
                return handle.read()
    raise FileNotFoundError("interface.html was not found in the source or installed asset locations")


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
        "language": "/language",
        "curriculum": "/curriculum",
        "chat": "/chat",
        "infra": "/infra",
        "conscience": "/conscience",
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
            loaded = _STATE_STORE.load_all(model=_MODEL, model_cls=Z3NeuralDynamics, world_model=_WORLD_MODEL, memory=_MEMORY, corpus_ingestor=_LANGUAGE_INGESTOR, conscience=_CONSCIENCE)
            _MODEL = loaded.get("model") or _MODEL
            # Mark loaded before building dependants: _get_language_ingestor()
            # re-enters get_model() on this thread (RLock) and must not reload.
            _STATE_LOADED = True
            _restore_language_ingestor_at_boot(loaded)
        return _MODEL


def _restore_language_ingestor_at_boot(loaded: Dict[str, Any]) -> None:
    """Build the canonical ingestor against the restored model and restore it.

    Previously the ingestor was created lazily on first language use, so at
    boot nothing was restored (load_all received None) and, when it was
    finally created, its optimizer restore failed on a param-group mismatch
    and the valid checkpoint was quarantined as corrupt.
    """
    global _BOOT_RESTORE
    report: Dict[str, Any] = {
        "at": time.time(),
        "loaded": dict(loaded.get("loaded", {})),
        "recovered": dict(loaded.get("recovered", {})),
        "errors": list(loaded.get("errors", [])),
    }
    config = _language_ingestor_config()
    if not config.enabled:
        report["corpus_ingestor"] = {"restored": False, "reason": "disabled"}
    else:
        try:
            ingestor = _get_language_ingestor()
            snap = ingestor.snapshot()
            restored = bool(snap.get("checkpoint_exists")) and not str(snap.get("last_error") or "").startswith(
                "Z3 corpus checkpoint restore failed"
            )
            report["corpus_ingestor"] = {
                "restored": restored,
                "trained_steps": snap.get("trained_steps"),
                "buffer_size": snap.get("buffer_size"),
                "optimizer_restore": snap.get("optimizer_restore"),
                "health_state": snap.get("health_state"),
                "checkpoint_path": ingestor.config.checkpoint_path,
                "last_error": snap.get("last_error") or None,
            }
        except Exception as exc:
            logger.exception("Corpus ingestor boot restore failed")
            report["corpus_ingestor"] = {"restored": False, "reason": f"{type(exc).__name__}: {exc}"}
    report["loaded"]["corpus_ingestor"] = bool(report["corpus_ingestor"].get("restored"))
    _BOOT_RESTORE = report


def get_model() -> Any:
    """Return the neural runtime after lazy restore."""
    return ensure_runtime_loaded()


def _get_optimizer(model: Any, learning_rate: float) -> Any:
    """Return the single persistent optimizer bound to the live model.

    The optimizer is rebuilt only when the model object changes (boot, load,
    reset). A learning-rate change is applied in place to every param group so
    Adam moments and any extra groups (e.g. the language decoder group added by
    /train-language) survive.
    """
    global _OPTIMIZER, _OPTIMIZER_MODEL_ID
    if _OPTIMIZER is None or _OPTIMIZER_MODEL_ID != id(model):
        _OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
        _OPTIMIZER_MODEL_ID = id(model)
        # Register the decoder group at creation so the optimizer has the same
        # shape on every boot; persisted Adam moments then restore fully.
        _register_decoder_params(_OPTIMIZER, model, float(learning_rate))
    else:
        for group in _OPTIMIZER.param_groups:
            group["lr"] = float(learning_rate)
    return _OPTIMIZER


def _register_decoder_params(optimizer: Any, model: Any, learning_rate: float) -> bool:
    """Add language-decoder parameters to the shared optimizer exactly once."""
    try:
        cfg = model.config
        dec = get_decoder(
            input_dim=cfg.input_dim,
            evidence_dim=cfg.evidence_dim,
            state_dim=cfg.state_dim,
            context_dim=cfg.context_dim,
        )
    except Exception as exc:
        logger.debug("Decoder unavailable for optimizer registration: %s", exc)
        return False
    existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
    new_params = [p for p in dec.parameters() if id(p) not in existing]
    if new_params:
        optimizer.add_param_group({"params": new_params, "lr": float(learning_rate)})
    return True


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


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    except Exception:
        return default



def _current_phi_sigma(model: Any) -> tuple[float, float]:
    metrics = model.metrics_to_dict(model.last_metrics)
    phi = _finite_float(metrics.get("mean_coherence", 0.5), 0.5)
    gate_entropy = _finite_float(metrics.get("gate_entropy", 0.5), 0.5)
    phi_floor = _finite_float(getattr(model.config, "phi_floor", 0.05), 0.05)
    sigma = min(1.0, max(0.0, _finite_float(model.config.noise_scale, 0.01) + gate_entropy))
    return max(phi_floor, min(1.0, phi)), sigma


def _tensor_is_finite(value: Any) -> bool:
    return bool(torch.isfinite(value).all().item())


def _model_state_is_finite(model: Any) -> bool:
    """Check model parameter/buffer finiteness under the RUNTIME_LOCK.

    The autonomous loop runs train_step on a background thread while the
    asyncio event loop may simultaneously call _ensure_finite_neural_runtime
    from a different context. Iterating model.parameters() / model.buffers()
    while another thread is mutating them via optimizer.step() causes the
    'free(): corrupted unsorted chunks' / segfault seen in production.
    Acquiring _RUNTIME_LOCK before the scan serialises access.
    """
    if torch is None:
        return False
    with _RUNTIME_LOCK:
        with torch.no_grad():
            for parameter in model.parameters():
                if not _tensor_is_finite(parameter):
                    return False
            for buffer in model.buffers():
                if buffer.numel() > 0 and not _tensor_is_finite(buffer):
                    return False
    return True


def _reset_corrupted_neural_runtime(reason: str) -> Dict[str, Any]:
    """Replace a non-finite neural core with a fresh model and persist the clean state."""
    global _MODEL, _OPTIMIZER, _STATE_LOADED
    _require_torch()
    with _RUNTIME_LOCK:
        _MODEL = Z3NeuralDynamics()
        _OPTIMIZER = None
        _STATE_LOADED = True
        manifest = _STATE_STORE.save_all(model=_MODEL, world_model=_WORLD_MODEL, memory=_MEMORY, conscience=_CONSCIENCE)
    return {"reset": True, "reason": reason, "state_manifest": manifest}


def _ensure_finite_neural_runtime(model: Any) -> Optional[Dict[str, Any]]:
    if not _model_state_is_finite(model):
        return _reset_corrupted_neural_runtime("non_finite_neural_state_detected")
    return None


def _broadcast_from_thread(payload: Dict[str, Any]) -> bool:
    """Schedule a WebSocket JSON broadcast on the main event loop from any thread."""
    loop = _MAIN_LOOP
    if loop is None or loop.is_closed():
        return False
    try:
        asyncio.run_coroutine_threadsafe(get_ws_manager().send_json(payload), loop)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("WebSocket broadcast scheduling failed: %s", exc)
        return False


def _push_voice_state(model: Any, neural_output: Dict[str, Any], memory_output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drive the audio synth and articulatory voice from a committed Z³ output."""
    status: Dict[str, Any] = {"synth": False, "articulatory": False, "broadcast": False}
    try:
        get_synthesizer(model.config.agent_count).push_z3_state(neural_output)
        status["synth"] = True
    except Exception as exc:
        logger.debug("Synth push failed: %s", exc)
    if get_articulatory_voice is None:
        return status
    try:
        cfg = model.config
        voice = get_articulatory_voice(z3_prediction_dim=cfg.evidence_dim + cfg.state_dim + cfg.context_dim)
        pred_input = torch.cat(
            [neural_output["integrated_evidence"], neural_output["z3_after"], neural_output["context"]],
            dim=-1,
        )
        mem_ctx = TrajectoryControlLayer.encode_memory_context(memory_output or {}, device=pred_input.device)
        with torch.no_grad():
            broadcast = voice.articulatory_state_broadcast(pred_input, mem_ctx)
        status["articulatory"] = True
        status["broadcast"] = _broadcast_from_thread(broadcast)
    except Exception as exc:
        logger.warning("Articulatory push failed: %s", exc)
    return status


_COMPOSE_OVERFLOW_WARNED = False


def _compose_z3_input(world_output: Dict[str, Any], memory_output: Dict[str, Any], expected_dim: int) -> List[float]:
    """Compose the Z³ input vector from world-model and resonant-memory outputs.

    Layout (default input_dim=16, latent_dim=8):
      0-7  world latent state
      8    memory salience (replaces world total_loss, which is a fixed linear
           combination of slots 9-12 and carried no independent signal)
      9-15 prediction, reconstruction, memory, coherence-alignment losses,
           novelty, nearest-memory distance, memory reconstruction confidence

    Previously 17 features were truncated to 16, silently dropping salience.
    If the vector is still too long (larger latent_dim), latent dimensions are
    trimmed so every scalar signal reaches Z³; nothing is dropped silently.
    """
    global _COMPOSE_OVERFLOW_WARNED
    latent = [float(v) for v in world_output.get("latent_state", [])]
    scalars = [
        float(memory_output.get("salience", 0.0)),
        float(world_output.get("prediction_loss", 0.0)),
        float(world_output.get("reconstruction_loss", 0.0)),
        float(world_output.get("memory_loss", 0.0)),
        float(world_output.get("coherence_alignment_loss", 0.0)),
        float(world_output.get("novelty", 0.0)),
        float(world_output.get("nearest_memory_distance", 0.0)),
        float(memory_output.get("reconstruction_confidence", 0.0)),
    ]
    latent_budget = max(int(expected_dim) - len(scalars), 0)
    if len(latent) > latent_budget:
        if not _COMPOSE_OVERFLOW_WARNED:
            logger.warning(
                "Z3 input overflow: latent_dim=%d + %d scalars > input_dim=%d; trimming latent tail",
                len(latent), len(scalars), expected_dim,
            )
            _COMPOSE_OVERFLOW_WARNED = True
        latent = latent[:latent_budget]
    features = latent + scalars
    if len(features) < expected_dim:
        features.extend([0.0] * (expected_dim - len(features)))
    return features[:expected_dim]


def _conscience_context(
    *,
    domain: str,
    world_output: Optional[Dict[str, Any]] = None,
    memory_output: Optional[Dict[str, Any]] = None,
    z3_metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    context = {
        "domain": domain,
        "world_model": world_output or {},
        "memory": memory_output or {},
        "z3_metrics": z3_metrics or {},
        "source": "z3_runtime_membrane",
    }
    if extra:
        context.update(extra)
    return context


def _evaluate_conscience(
    proposal: Dict[str, Any] | str,
    *,
    domain: str,
    world_output: Optional[Dict[str, Any]] = None,
    memory_output: Optional[Dict[str, Any]] = None,
    z3_metrics: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged_context = _conscience_context(
        domain=domain,
        world_output=world_output,
        memory_output=memory_output,
        z3_metrics=z3_metrics,
        extra=context,
    )
    return _CONSCIENCE.evaluate(proposal, context=merged_context).to_dict()


def _persist_if_requested(persist: bool) -> Optional[Dict[str, Any]]:
    if not persist:
        return None
    model = get_model()
    return _STATE_STORE.save_all(model=model, world_model=_WORLD_MODEL, memory=_MEMORY, corpus_ingestor=_LANGUAGE_INGESTOR, conscience=_CONSCIENCE)


def _runtime_save() -> Dict[str, Any]:
    model = get_model()
    return _STATE_STORE.save_all(model=model, world_model=_WORLD_MODEL, memory=_MEMORY, corpus_ingestor=_LANGUAGE_INGESTOR, conscience=_CONSCIENCE)


def _language_ingestor_config(*, batch_size: Optional[int] = None, learning_rate: Optional[float] = None) -> Z3CorpusIngestionConfig:
    """Build the canonical ingestor config from env plus live runtime overrides."""
    base = Z3CorpusIngestionConfig.from_env()
    checkpoint_path = base.checkpoint_path
    if not os.environ.get("Z3_CORPUS_CHECKPOINT_PATH"):
        checkpoint_path = str(_STATE_STORE.corpus_ingestor_path)
    return replace(
        base,
        checkpoint_path=checkpoint_path,
        batch_size=max(1, int(batch_size if batch_size is not None else _RUNTIME_LANGUAGE_CONFIG.get("batch_size", base.batch_size))),
        learning_rate=float(learning_rate if learning_rate is not None else _RUNTIME_LANGUAGE_CONFIG.get("learning_rate", base.learning_rate)),
        window_size=max(1, int(_RUNTIME_LANGUAGE_CONFIG.get("window_size", base.window_size))),
        stride=max(1, int(_RUNTIME_LANGUAGE_CONFIG.get("stride", base.stride))),
        truncation_steps=max(1, int(_RUNTIME_LANGUAGE_CONFIG.get("truncation_steps", base.truncation_steps))),
    )


def _get_language_ingestor(*, batch_size: Optional[int] = None, learning_rate: Optional[float] = None) -> Z3CorpusNeuralIngestor:
    """Return the canonical language-training ingestor attached to the live model."""
    global _LANGUAGE_INGESTOR
    model = get_model()
    config = _language_ingestor_config(batch_size=batch_size, learning_rate=learning_rate)
    optimizer = _get_optimizer(model, config.learning_rate)
    if _LANGUAGE_INGESTOR is None:
        _LANGUAGE_INGESTOR = Z3CorpusNeuralIngestor(
            config,
            model=model,
            optimizer=optimizer,
            trainer=train_z3_on_language_window,
            own_model=False,
        )
    else:
        _LANGUAGE_INGESTOR.config = config
        _LANGUAGE_INGESTOR.attach_runtime(model=model, optimizer=optimizer)
        # Keep the monkeypatch-friendly module symbol as the active trainer.
        _LANGUAGE_INGESTOR._trainer = train_z3_on_language_window
    return _LANGUAGE_INGESTOR


def _runtime_language_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    world_losses = [float(item.get("world_model", {}).get("total_loss", 0.0) or 0.0) for item in results]
    confidences = [float(item.get("memory", {}).get("reconstruction_confidence", 0.0) or 0.0) for item in results]
    return {
        "events_ingested": len(results),
        "mean_world_loss": round(sum(world_losses) / max(len(world_losses), 1), 6),
        "mean_memory_confidence": round(sum(confidences) / max(len(confidences), 1), 6),
    }


def _runtime_curriculum_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    base = _runtime_language_summary(results)
    kinds: Dict[str, int] = {}
    mean_salience = []
    for item in results:
        observation = item.get("observation", {})
        kind = str(observation.get("curriculum_kind") or "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1
        memory = item.get("memory", {})
        mean_salience.append(float(memory.get("salience", 0.0) or 0.0))
    base["curriculum_kinds"] = kinds
    base["mean_salience"] = round(sum(mean_salience) / max(len(mean_salience), 1), 6)
    return base


def _train_z3_on_language_observations(observations: List[Dict[str, Any]], *, learning_rate: float) -> Dict[str, Any]:
    """Train Z³ directly on corpus text sequence windows through the canonical ingestor."""
    texts = [str(obs.get("text") or obs.get("content") or "").strip() for obs in observations]
    texts = [text for text in texts if text]
    if not texts:
        return {"trained": False, "reason": "no_text", "texts": 0, "mode": "canonical_corpus_neural_ingestor"}
    metadata = [{key: value for key, value in obs.items() if key not in ("text", "content")} for obs in observations]
    with _RUNTIME_LOCK:
        ingestor = _get_language_ingestor(batch_size=len(texts), learning_rate=learning_rate)
        report = ingestor.train_texts_now(texts, metadata=metadata)
    report["ingestor_health"] = ingestor.health_state()
    return report


def _ingest_language_batch_for_runtime(*, batch_size: int, train: bool, learning_rate: float) -> Dict[str, Any]:
    batch = _LANGUAGE_STREAM.fetch_batch(batch_size=batch_size)
    observations = list(batch.get("observations", []))
    language_training: Optional[Dict[str, Any]] = None
    if train and observations:
        language_training = _train_z3_on_language_observations(observations, learning_rate=learning_rate)
    results: List[Dict[str, Any]] = []
    for observation in observations:
        integrated = IntegratedObserveRequest(
            observation=observation,
            domain=batch.get("domain", LanguageStream.DEFAULT_DOMAIN),
            train=False,
            persist=False,
            learning_rate=learning_rate,
        )
        results.append(integrated_observe(integrated))
    return {
        "dataset": batch.get("dataset"),
        "domain": batch.get("domain"),
        "count": len(results),
        "offset": batch.get("offset"),
        "summary": _runtime_language_summary(results),
        "language_training": language_training,
        "sample_entity_ids": [
            item.get("world_model", {}).get("observation", {}).get("entity_id")
            for item in results[:5]
        ],
    }


def _ingest_curriculum_batch_for_runtime(*, batch_size: int, source: Optional[str], train: bool, learning_rate: float) -> Dict[str, Any]:
    batch = _CURRICULUM_STREAM.fetch_batch(batch_size=batch_size, source=source)
    observations = list(batch.get("observations", []))
    results: List[Dict[str, Any]] = []
    for observation in observations:
        domain = f"curriculum:{observation.get('curriculum_kind', 'mixed')}"
        integrated = IntegratedObserveRequest(
            observation=observation,
            domain=domain,
            train=train,
            persist=False,
            learning_rate=learning_rate,
        )
        item = integrated_observe(integrated)
        item["observation"] = observation
        results.append(item)
    return {
        "dataset": batch.get("dataset"),
        "domain": batch.get("domain"),
        "count": len(results),
        "offset": batch.get("offset"),
        "summary": _runtime_curriculum_summary(results),
        "sample_entity_ids": [item.get("observation", {}).get("entity_id") for item in results[:5]],
    }


def _runtime_tick() -> Dict[str, Any]:
    """Run one autonomous heartbeat observation and optional Language learning update."""
    global _RUNTIME_TICK_SEQUENCE
    _RUNTIME_TICK_SEQUENCE += 1
    runtime_tick_id = _RUNTIME_TICK_SEQUENCE
    model = get_model()
    reset_info = _ensure_finite_neural_runtime(model)
    if reset_info:
        model = get_model()
    phi, sigma = _current_phi_sigma(model)
    prior_metrics = model.metrics_to_dict(model.last_metrics)
    runtime_phase = runtime_tick_id / 10.0
    observation = {
        "timestamp": time.time(),
        "source": "autonomous_runtime_loop",
        "domain": "runtime",
        "world_iteration": _WORLD_MODEL.iteration,
        "memory_rings": len(_MEMORY.rings),
        "phi": phi,
        "sigma": sigma,
        "tick_kind": "heartbeat_learning",
        "runtime_tick_id": runtime_tick_id,
        "runtime_phase_sin": math.sin(runtime_phase),
        "runtime_phase_cos": math.cos(runtime_phase),
        "recent_drift": _finite_float(prior_metrics.get("z3_delta_norm", 0.0), 0.0),
        "recent_gate": _finite_float(prior_metrics.get("mean_gate", 0.0), 0.0),
        "recent_novelty": _finite_float(prior_metrics.get("mean_novelty", 0.0), 0.0),
        "recent_useful_novelty": _finite_float(prior_metrics.get("useful_novelty", 0.0), 0.0),
        "language_ingestion_enabled": bool(_RUNTIME_LANGUAGE_CONFIG.get("enabled", False)),
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
            "drift": _finite_float(prior_metrics.get("z3_delta_norm", 0.0), 0.0),
            "gate": _finite_float(prior_metrics.get("mean_gate", 0.0), 0.0),
            "novelty": _finite_float(prior_metrics.get("mean_novelty", 0.0), 0.0),
            "useful_novelty": _finite_float(prior_metrics.get("useful_novelty", 0.0), 0.0),
            "regime": "autonomous_runtime",
        },
    )
    world_dict = world_output.to_dict()
    conscience_result = _evaluate_conscience(
        observation,
        domain="runtime",
        world_output=world_dict,
        memory_output=memory_output,
        z3_metrics=prior_metrics,
        context={"kind": "autonomous_runtime_tick"},
    )
    z3_vector = _compose_z3_input(world_dict, memory_output, model.config.input_dim)
    x = _tensor_from_vector(z3_vector, model.config.input_dim)
    optimizer = _get_optimizer(model, float(os.environ.get("Z3_RUNTIME_LR", "0.001")))
    try:
        with _RUNTIME_LOCK:
            train_metrics = model.train_step(optimizer, x, target=x, update_recurrent_state=True)
            with torch.no_grad():
                projection_output = model.forward(x, hard_gate=False, update_state=False, add_noise=False)
    except Exception as exc:
        error_text = str(exc).lower()
        if "nan" not in error_text and "non-finite" not in error_text and "inf" not in error_text:
            raise
        reset_info = _reset_corrupted_neural_runtime(f"runtime_tick_recovered_from_{type(exc).__name__}")
        model = get_model()
        optimizer = _get_optimizer(model, float(os.environ.get("Z3_RUNTIME_LR", "0.001")))
        with _RUNTIME_LOCK:
            train_metrics = model.train_step(optimizer, x, target=x, update_recurrent_state=True)
            with torch.no_grad():
                projection_output = model.forward(x, hard_gate=False, update_state=False, add_noise=False)
        train_metrics["self_healing_reset"] = reset_info

    # Feed live phase state into the audio synthesizer (autonomous tick)
    try:
        get_synthesizer(model.config.agent_count).push_z3_state(projection_output)
    except Exception:
        pass

    language_result: Optional[Dict[str, Any]] = None
    language_enabled = bool(_RUNTIME_LANGUAGE_CONFIG.get("enabled", False))
    language_every = max(1, int(_RUNTIME_LANGUAGE_CONFIG.get("every_ticks", 10)))
    if language_enabled and runtime_tick_id % language_every == 0:
        language_result = _ingest_language_batch_for_runtime(
            batch_size=max(1, int(_RUNTIME_LANGUAGE_CONFIG.get("batch_size", 5))),
            train=bool(_RUNTIME_LANGUAGE_CONFIG.get("train", False)),
            learning_rate=float(_RUNTIME_LANGUAGE_CONFIG.get("learning_rate", 0.001)),
        )

    return {
        "observation": observation,
        "input_vector": z3_vector,
        "world_model": world_dict,
        "memory": memory_output,
        "conscience": conscience_result,
        "z3": {
            "metrics": train_metrics,
            "projection": model.public_projection(projection_output),
        },
        "language_ingestion": language_result,
        "language_ingestor": _LANGUAGE_INGESTOR.snapshot() if _LANGUAGE_INGESTOR is not None else None,
        "language_config": dict(_RUNTIME_LANGUAGE_CONFIG),
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


def runtime_status_payload() -> Dict[str, Any]:
    payload = get_runtime_loop().status()
    payload["language_schedule"] = dict(_RUNTIME_LANGUAGE_CONFIG)
    payload["language_stream"] = _LANGUAGE_STREAM.status()
    if _LANGUAGE_INGESTOR is not None:
        payload["language_ingestor"] = _LANGUAGE_INGESTOR.snapshot()
    payload["boot_restore"] = dict(_BOOT_RESTORE)
    return payload


def apply_runtime_language_config(request: RuntimeStartRequest) -> None:
    _RUNTIME_LANGUAGE_CONFIG.update(
        {
            "enabled": bool(request.language_enabled),
            "every_ticks": max(1, int(request.language_every_ticks)),
            "batch_size": max(1, int(request.language_batch_size)),
            "train": bool(request.language_train),
            "learning_rate": float(request.language_learning_rate),
        }
    )


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
    ingestor_snapshot = _LANGUAGE_INGESTOR.snapshot() if _LANGUAGE_INGESTOR is not None else None
    ingestor_health = (ingestor_snapshot or {}).get("health_state", "not_initialized")
    degraded_ingestor_states = {"circuit_open", "degraded", "backlogged", "unavailable"}
    status = "ok" if IMPORT_ERROR is None else "degraded"
    if ingestor_health in degraded_ingestor_states:
        status = "degraded"
    return {
        "status": status,
        "security": {
            "auth_required": _AUTH_REQUIRED,
            "auth_configured": bool(_API_TOKEN),
            "cors_origins": [origin.strip() for origin in os.environ.get("Z3_CORS_ORIGINS", "http://localhost:8000").split(",") if origin.strip()],
            "rate_limit_requests": _RATE_LIMIT_REQUESTS,
            "rate_limit_window_seconds": _RATE_LIMIT_WINDOW,
        },
        "torch_available": torch is not None,
        "optional_modules": {
            "acoustic_encoder": get_acoustic_encoder is not None,
            "articulatory_voice": get_articulatory_voice is not None,
            "articulatory_import_error": ARTICULATORY_IMPORT_ERROR,
            "acoustic_import_error": ACOUSTIC_IMPORT_ERROR,
        },
        "model_loaded": model_loaded,
        "state_loaded": _STATE_LOADED,
        "import_error": IMPORT_ERROR,
        "state_store": _STATE_STORE.manifest(),
        "language_ingestor_health": ingestor_health,
        "language_ingestor": ingestor_snapshot,
        "runtime": runtime_status_payload(),
    }


@app.get("/config")
def config() -> Dict[str, Any]:
    model = get_model()
    return {
        "config": model.config.__dict__,
        "metric_names": list(model.metric_names()),
        "world_model": _WORLD_MODEL.get_state(),
        "memory": _MEMORY.get_snapshot(recent_ring_count=3)["metrics"],
        "conscience": _CONSCIENCE.get_state(),
        "curriculum": _CURRICULUM_STREAM.status(),
    }


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    model = get_model()
    x = _tensor_from_vector(request.x, model.config.input_dim)
    with _RUNTIME_LOCK, torch.no_grad():
        output = model.forward(
            x,
            hard_gate=request.hard_gate,
            update_state=request.update_state,
            add_noise=False,
        )
    response = _metrics(output, model)
    # Feed live phase state into the audio synthesizer
    try:
        get_synthesizer(model.config.agent_count).push_z3_state(output)
    except Exception:
        pass
    manifest = _persist_if_requested(request.persist)
    if manifest:
        response["state_manifest"] = manifest
    return response


@app.post("/train-step")
def train_step(request: TrainStepRequest) -> Dict[str, Any]:
    model = get_model()
    x = _tensor_from_vector(request.x, model.config.input_dim)
    target = _tensor_from_vector(request.target, model.config.input_dim) if request.target is not None else None
    with _RUNTIME_LOCK:
        optimizer = _get_optimizer(model, request.learning_rate)
        metrics = model.train_step(optimizer, x, target=target, update_recurrent_state=True)
        # Read the projection without advancing the recurrent state a second time.
        with torch.no_grad():
            projection_output = model.forward(x, hard_gate=False, update_state=False, add_noise=False)
    response = {"metrics": metrics, "projection": model.public_projection(projection_output)}
    manifest = _persist_if_requested(request.persist)
    if manifest:
        response["state_manifest"] = manifest
    return response


class GenerateRequest(BaseModel):
    """Request payload for Z³ end-to-end language generation."""
    prompt: str = Field("", description="Text prompt to condition Z³ on before generating.")
    max_new_tokens: int = Field(120, ge=1, le=512, description="Maximum characters to generate.")
    temperature: float = Field(0.85, ge=0.01, le=2.0, description="Sampling temperature.")
    top_k: int = Field(40, ge=1, le=200, description="Top-k sampling cutoff.")
    top_p: float = Field(0.92, ge=0.1, le=1.0, description="Nucleus sampling cutoff.")
    repetition_penalty: float = Field(1.15, ge=1.0, le=3.0, description="Repetition penalty.")


class TrainLanguageRequest(BaseModel):
    """Request payload for one language decoder training step."""
    text: str = Field(..., min_length=4, description="Text to train the language decoder on.")
    learning_rate: float = Field(1e-3, gt=0.0, le=1.0, description="Learning rate.")
    max_seq_len: int = Field(256, ge=8, le=1024, description="Maximum sequence length.")
    persist: bool = Field(True, description="Save decoder weights after training.")


@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    """Observe a chatbox message and express a state-grounded Z³ response.

    If the language decoder has been trained, the response will be generated
    end-to-end from Z³'s internal state. Otherwise falls back to the
    deterministic response_adapter template.
    """
    model = get_model()
    observation = LanguageStream.text_to_observation(request.message, source="chatbox")
    language_training = _train_z3_on_language_observations([observation], learning_rate=request.learning_rate) if request.train else None
    integrated = IntegratedObserveRequest(
        observation=observation,
        domain="language:chat",
        train=False,
        persist=request.persist,
        learning_rate=request.learning_rate,
    )
    result = integrated_observe(integrated)

    # Try end-to-end generation from Z³'s own language decoder
    generated_response = None
    decoder_used = False
    try:
        cfg = model.config
        dec = get_decoder(
            input_dim=cfg.input_dim,
            evidence_dim=cfg.evidence_dim,
            state_dim=cfg.state_dim,
            context_dim=cfg.context_dim,
        )
        with _RUNTIME_LOCK:
            generated_response = z3_generate(
                model, dec,
                prompt=request.message,
                max_new_tokens=120,
                temperature=0.85,
                top_k=40,
                top_p=0.92,
            )
        decoder_used = bool(generated_response and generated_response.strip())
    except Exception:
        pass

    # Fallback to deterministic template if decoder not trained yet
    expression = build_z3_expression(
        message=request.message,
        observation=observation,
        integrated_result=result,
        language_training=language_training,
    )
    expression_payload = expression.to_dict()

    response_text = generated_response.strip() if decoder_used else expression.response

    return {
        "response": response_text,
        "response_source": "z3_language_decoder" if decoder_used else "response_adapter_template",
        "domain": "language:chat",
        "language_ingested": True,
        "language_training": language_training,
        "expression": expression_payload,
        "observation": observation,
        "result": result,
    }


@app.post("/generate")
def generate_text(request: GenerateRequest) -> Dict[str, Any]:
    """Generate natural English text directly from Z³'s internal state.

    This is the end-to-end language generation endpoint. Z³ uses its own
    trained language decoder to produce text without any external LLM.
    The quality improves as the decoder is trained on more corpus text.
    """
    model = get_model()
    cfg = model.config
    try:
        dec = get_decoder(
            input_dim=cfg.input_dim,
            evidence_dim=cfg.evidence_dim,
            state_dim=cfg.state_dim,
            context_dim=cfg.context_dim,
        )
        with _RUNTIME_LOCK:
            text = z3_generate(
                model, dec,
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
            )
        return {
            "generated": text,
            "prompt": request.prompt,
            "tokens": len(text),
            "decoder": "z3_language_decoder",
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Language decoder unavailable: {exc}")


@app.post("/train-language")
def train_language(request: TrainLanguageRequest) -> Dict[str, Any]:
    """Train Z³'s end-to-end language decoder on one text sample.

    Each call runs one next-token prediction training step through both
    Z³'s neural core and the language decoder head jointly. Call this
    repeatedly with corpus text to teach Z³ to generate natural English.
    """
    model = get_model()
    cfg = model.config
    try:
        dec = get_decoder(
            input_dim=cfg.input_dim,
            evidence_dim=cfg.evidence_dim,
            state_dim=cfg.state_dim,
            context_dim=cfg.context_dim,
        )
        optimizer = _get_optimizer(model, request.learning_rate)
        _register_decoder_params(optimizer, model, request.learning_rate)
        with _RUNTIME_LOCK:
            metrics = train_language_step(
                model, dec, optimizer,
                request.text,
                max_seq_len=request.max_seq_len,
            )
        if request.persist:
            save_decoder(dec)
        return {"language_decoder_training": metrics}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Language training failed: {exc}")


@app.get("/language")
def language_status() -> Dict[str, Any]:
    """Return language stream status without loading corpus content into the response."""
    return _LANGUAGE_STREAM.status()


@app.post("/language/load")
def language_load() -> Dict[str, Any]:
    """Load configured real language corpus text."""
    return _LANGUAGE_STREAM.ensure_loaded()


@app.post("/language/fetch")
def language_fetch(request: LanguageBatchRequest) -> Dict[str, Any]:
    """Fetch converted language observations without ingesting them."""
    return _LANGUAGE_STREAM.fetch_batch(batch_size=request.batch_size)


@app.post("/language/ingest")
def language_ingest(request: LanguageBatchRequest) -> Dict[str, Any]:
    """Fetch language observations and feed them through the integrated Z³ observe path."""
    summary = _ingest_language_batch_for_runtime(
        batch_size=request.batch_size,
        train=request.train,
        learning_rate=request.learning_rate,
    )
    manifest = _persist_if_requested(request.persist)
    return {**summary, "state_manifest": manifest}


@app.get("/curriculum")
def curriculum_status() -> Dict[str, Any]:
    """Return layered curriculum stream status without loading full data into the response."""
    return _CURRICULUM_STREAM.status()


@app.post("/curriculum/load")
def curriculum_load() -> Dict[str, Any]:
    """Load configured layered curriculum data or initialize remote streams."""
    return _CURRICULUM_STREAM.ensure_loaded()


@app.post("/curriculum/fetch")
def curriculum_fetch(request: CurriculumBatchRequest) -> Dict[str, Any]:
    """Fetch converted curriculum observations without ingesting them."""
    return _CURRICULUM_STREAM.fetch_batch(batch_size=request.batch_size, source=request.source)


@app.post("/curriculum/ingest")
def curriculum_ingest(request: CurriculumBatchRequest) -> Dict[str, Any]:
    """Feed layered curriculum observations through the integrated Z³ observe path."""
    summary = _ingest_curriculum_batch_for_runtime(
        batch_size=request.batch_size,
        source=request.source,
        train=request.train,
        learning_rate=request.learning_rate,
    )
    manifest = _persist_if_requested(request.persist)
    return {**summary, "state_manifest": manifest}


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
    # World model, resonant memory, conscience and Z³ are all mutated here; hold
    # the runtime lock so the autonomous loop cannot train concurrently.
    with _RUNTIME_LOCK:
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
        world_dict = world_output.to_dict()
        prior_z3_metrics = model.metrics_to_dict(model.last_metrics)
        conscience_result = _evaluate_conscience(
            request.observation,
            domain=request.domain,
            world_output=world_dict,
            memory_output=memory_output,
            z3_metrics=prior_z3_metrics,
            context={"kind": "integrated_observe"},
        )
        z3_vector = _compose_z3_input(world_dict, memory_output, model.config.input_dim)
        x = _tensor_from_vector(z3_vector, model.config.input_dim)

        neural_output: Optional[Dict[str, Any]] = None
        if conscience_result.get("decision") == "reject":
            z3_response: Dict[str, Any] = {
                "blocked_by_conscience": True,
                "metrics": prior_z3_metrics,
                "projection": None,
                "reason": conscience_result.get("rationale"),
            }
        elif request.train:
            optimizer = _get_optimizer(model, request.learning_rate)
            # train_step commits the recurrent state exactly once; the projection
            # is read without a second state advance (was step_runtime -> 2x).
            z3_metrics = model.train_step(optimizer, x, target=x, update_recurrent_state=True)
            with torch.no_grad():
                neural_output = model.forward(x, hard_gate=False, update_state=False, add_noise=False)
            z3_response = {"metrics": z3_metrics, "projection": model.public_projection(neural_output)}
        else:
            with torch.no_grad():
                neural_output = model.forward(x, hard_gate=request.hard_gate, update_state=True, add_noise=False)
            z3_response = _metrics(neural_output, model)

        if neural_output is not None:
            z3_response["voice"] = _push_voice_state(model, neural_output, memory_output)

    response = {
        "input_vector": z3_vector,
        "world_model": world_dict,
        "memory": memory_output,
        "conscience": conscience_result,
        "z3": z3_response,
    }
    response["infra"] = _INFRA.record_observation(
        observation=request.observation,
        domain=request.domain,
        vector=z3_vector,
        world_model=world_dict,
        memory=memory_output,
        z3_metrics=z3_response.get("metrics", {}),
    )
    manifest = _persist_if_requested(request.persist)
    if manifest:
        response["state_manifest"] = manifest
    return response


@app.get("/conscience")
def conscience_state() -> Dict[str, Any]:
    """Return production conscience policy identity, memory status, and last decision."""
    get_model()
    return _CONSCIENCE.get_state()


@app.get("/conscience/policy")
def conscience_policy() -> Dict[str, Any]:
    """Return the active externalized conscience policy for auditability."""
    get_model()
    return _CONSCIENCE.policy.to_dict()


@app.post("/conscience/evaluate")
def conscience_evaluate(request: ConscienceEvaluateRequest) -> Dict[str, Any]:
    """Evaluate a proposal through the production conscience membrane."""
    get_model()
    result = _CONSCIENCE.evaluate(request.proposal, context=request.context).to_dict()
    manifest = _persist_if_requested(request.persist)
    if manifest:
        result["state_manifest"] = manifest
    return result


@app.post("/conscience/outcome")
def conscience_outcome(request: ConscienceOutcomeRequest) -> Dict[str, Any]:
    """Record observed outcome feedback for the last conscience decision."""
    if _CONSCIENCE.last_result is None:
        raise HTTPException(status_code=409, detail="No conscience decision is available to calibrate.")
    trace = _CONSCIENCE.learn_from_outcome(
        _CONSCIENCE.last_result,
        outcome_value=request.outcome_value,
        salience=request.salience,
        confidence=request.confidence,
        provenance=request.provenance,
        notes=request.notes,
    )
    response: Dict[str, Any] = {"trace": trace.to_dict(), "conscience": _CONSCIENCE.get_state()}
    manifest = _persist_if_requested(request.persist)
    if manifest:
        response["state_manifest"] = manifest
    return response


@app.get("/infra")
def infra_status() -> Dict[str, Any]:
    """Return optional Railway infrastructure wiring status."""
    return _INFRA.status(state_manifest=_STATE_STORE.manifest())


@app.post("/infra/sync")
def infra_sync() -> Dict[str, Any]:
    """Push a lightweight runtime snapshot to configured infrastructure backends."""
    snapshot = {
        "runtime": runtime_status_payload(),
        "state": _STATE_STORE.manifest(),
        "world_model": _WORLD_MODEL.get_state(),
        "memory": _MEMORY.get_snapshot(recent_ring_count=5).get("metrics", {}),
        "conscience": _CONSCIENCE.get_state(),
        "language": _LANGUAGE_STREAM.status(),
        "language_ingestor": _LANGUAGE_INGESTOR.snapshot() if _LANGUAGE_INGESTOR is not None else None,
        "curriculum": _CURRICULUM_STREAM.status(),
    }
    return _INFRA.sync_snapshot(snapshot)


@app.get("/state")
def state_manifest() -> Dict[str, Any]:
    manifest = _STATE_STORE.manifest()
    manifest["boot_restore"] = dict(_BOOT_RESTORE)
    return manifest


@app.post("/state/save")
def state_save() -> Dict[str, Any]:
    model = get_model()
    with _RUNTIME_LOCK:
        return _STATE_STORE.save_all(model=model, world_model=_WORLD_MODEL, memory=_MEMORY, corpus_ingestor=_LANGUAGE_INGESTOR, conscience=_CONSCIENCE)


@app.post("/state/load")
def state_load() -> Dict[str, Any]:
    global _MODEL, _STATE_LOADED
    _require_torch()
    with _RUNTIME_LOCK:
        loaded = _STATE_STORE.load_all(model=_MODEL, model_cls=Z3NeuralDynamics, world_model=_WORLD_MODEL, memory=_MEMORY, corpus_ingestor=_LANGUAGE_INGESTOR, conscience=_CONSCIENCE)
        _MODEL = loaded.pop("model")
        _STATE_LOADED = True
    return loaded


@app.get("/runtime")
def runtime_status() -> Dict[str, Any]:
    return runtime_status_payload()


@app.post("/runtime/start")
def runtime_start(request: RuntimeStartRequest) -> Dict[str, Any]:
    get_model()
    apply_runtime_language_config(request)
    get_runtime_loop().start(
        interval_seconds=request.interval_seconds,
        autosave_every_ticks=request.autosave_every_ticks,
    )
    return runtime_status_payload()


@app.post("/runtime/stop")
def runtime_stop() -> Dict[str, Any]:
    get_runtime_loop().stop()
    return runtime_status_payload()


@app.post("/runtime/tick")
def runtime_tick() -> Dict[str, Any]:
    get_model()
    get_runtime_loop().tick_once()
    return runtime_status_payload()


# Startup/shutdown are now handled by the lifespan context manager above.
# The on_event handlers have been removed to eliminate the DeprecationWarning
# and to ensure a single, authoritative lifecycle path.


@app.websocket("/ws/neural_stream")
async def neural_stream_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint: streams PCM audio bytes and JSON metrics to the dashboard.

    Accepts JSON messages from the client:
      {"type": "user_input", "text": "..."}   — text perturbation strike
      {"type": "mic_pitch", "hz": 440.0}       — microphone pitch input
      {"type": "trigger_save"}                  — manual W matrix save
    """
    if _AUTH_REQUIRED:
        authorization = websocket.headers.get("authorization", "")
        header_token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else websocket.headers.get("x-api-key", "").strip()
        provided_token = header_token or websocket.query_params.get("token", "").strip()
        if not _API_TOKEN or not secrets.compare_digest(provided_token, _API_TOKEN):
            await websocket.close(code=1008, reason="Authentication required")
            return
    manager = get_ws_manager()
    synth = get_synthesizer()
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except Exception:
                continue
            msg_type = payload.get("type", "")
            if msg_type == "user_input":
                synth.push_perturbation(payload.get("text", ""))
            elif msg_type == "mic_pitch":
                synth.push_mic_pitch(float(payload.get("hz", 0.0)))
            elif msg_type == "mic_audio":
                try:
                    # Expecting base64 encoded raw 16-bit PCM bytes (100ms window)
                    import base64
                    import numpy as np
                    audio_bytes = base64.b64decode(payload.get("audio", ""))
                    if audio_bytes:
                        waveform = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).to(_MODEL.device)
                        
                        # Encode audio to Z³ input vector
                        encoder = get_acoustic_encoder(input_dim=_MODEL.config.input_dim)
                        z3_input = encoder(waveform_tensor)
                        
                        # Feed into Z³ for entrainment
                        with _RUNTIME_LOCK:
                            neural_output = _MODEL.forward(z3_input, update_state=True, add_noise=False)
                        
                        # Update audio synth and articulatory voice
                        get_synthesizer().push_z3_state(neural_output)
                        
                        # Push articulatory state broadcast over WebSocket
                        cfg = _MODEL.config
                        voice = get_articulatory_voice(
                            z3_prediction_dim=cfg.evidence_dim + cfg.state_dim + cfg.context_dim
                        )
                        pred_input = torch.cat([
                            neural_output["integrated_evidence"],
                            neural_output["z3_after"],
                            neural_output["context"],
                        ], dim=-1)
                        
                        # Memory context from recent observe
                        mem_ctx = TrajectoryControlLayer.encode_memory_context(
                            _MEMORY.get_snapshot().get("metrics", {}),
                            device=pred_input.device,
                        )
                        
                        broadcast = voice.articulatory_state_broadcast(pred_input, mem_ctx)
                        await manager.send_json(broadcast)
                except Exception as exc:
                    print(f"[Z³ Entrainment] Error: {exc}")
            elif msg_type == "trigger_save":
                success = synth.save_weights()
                await websocket.send_text(json.dumps({
                    "type": "save_status",
                    "status": "success" if success else "failed",
                }))
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)


@app.get("/api/audio-stream")
async def audio_stream() -> StreamingResponse:
    """Stream the Z³ neural core's emergent phase waves as raw 16-bit PCM audio.

    Connect with an HTML5 <audio> tag or the Web Audio API client in the
    dashboard. The stream reflects the live coherence and phase alignment of
    the Z-prime agent ensemble in real time.

    Media type: audio/x-raw; codec=pcm; bit=16; rate=44100; channels=1
    """
    try:
        agent_count = 8
        model = _MODEL
        if model is not None:
            agent_count = getattr(getattr(model, "config", None), "agent_count", 8)
        synth = get_synthesizer(agent_count)
        if not synth._running:
            synth.start()
        return StreamingResponse(
            synth.stream_generator(),
            media_type="audio/x-raw;codec=pcm;bit=16;rate=44100;channels=1",
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Audio synthesizer unavailable: {exc}")


@app.get("/api/articulatory-state")
def articulatory_state_endpoint() -> dict:
    """Return the current vocal tract geometry from Z³'s live state.

    Shows the 16 articulatory parameters (F0, formants, nasality, etc.)
    and the 4-step coarticulatory trajectory predicted by the
    TrajectoryControlLayer conditioned on the resonant memory rings.
    """
    try:
        model = get_model()
        cfg = model.config
        voice = get_articulatory_voice(
            z3_prediction_dim=cfg.evidence_dim + cfg.state_dim + cfg.context_dim
        )
        with _RUNTIME_LOCK:
            with torch.no_grad():
                x = torch.zeros(1, cfg.input_dim)
                out = model.forward(x, update_state=False, add_noise=False)
            pred_input = torch.cat([
                out["integrated_evidence"],
                out["z3_after"],
                out["context"],
            ], dim=-1)
        mem_ctx = TrajectoryControlLayer.encode_memory_context(
            _MEMORY.get_snapshot() if hasattr(_MEMORY, 'get_snapshot') else {},
        )
        broadcast = voice.articulatory_state_broadcast(pred_input, mem_ctx)
        return broadcast
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Articulatory state unavailable: {exc}")


@app.get("/api/audio-status")
def audio_status() -> dict:
    """Return the current status of the Z³ audio synthesizer."""
    try:
        agent_count = 8
        if _MODEL is not None:
            agent_count = getattr(getattr(_MODEL, "config", None), "agent_count", 8)
        synth = get_synthesizer(agent_count)
        return {
            "running": synth._running,
            "agent_count": synth.agent_count,
            "sample_rate": synth.sample_rate,
            "chunk_samples": synth.chunk_samples,
            "amplitude_scale": synth.amplitude_scale,
            "state_version": synth._state_version,
            "queue_size": synth._audio_queue.qsize() if synth._audio_queue else 0,
            "intrinsic_freqs_hz": synth.intrinsic_freqs.tolist(),
        }
    except Exception as exc:
        return {"running": False, "error": str(exc)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
