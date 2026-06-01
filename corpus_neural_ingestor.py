"""Canonical neural corpus ingestion for the Z³ runtime.

This module centralizes the path from raw corpus text into real Z³ neural
language-window training. It can either own a standalone neural model or attach
to the live runtime model used by ``main.py``. The ingestor keeps deployment
safety concerns local: environment parsing, bounded buffering, rollback after
failed batches, diagnostics, and optional optimizer/ingestion checkpoint state.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - depends on deployment image.
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from z3_language_training import train_z3_on_language_window
    from Z3_neural_dynamics import Z3Config, Z3NeuralDynamics
except Exception as exc:  # pragma: no cover - import diagnostics for runtime status.
    train_z3_on_language_window = None  # type: ignore[assignment]
    Z3Config = None  # type: ignore[assignment]
    Z3NeuralDynamics = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

TrainerFn = Callable[..., Dict[str, float]]
OptimizerFactory = Callable[[Any, float], Any]
ModelProvider = Callable[[], Any]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except Exception:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, *, minimum: Optional[float] = None) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            numeric = float(raw)
            value = numeric if math.isfinite(numeric) else default
        except Exception:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _default_state_dir() -> Path:
    return Path(os.environ.get("Z3_STATE_DIR", os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/data" if Path("/data").exists() else "data")))


@dataclass(frozen=True)
class Z3CorpusIngestionConfig:
    """Runtime configuration for neural corpus ingestion."""

    enabled: bool = True
    batch_size: int = 8
    min_words: int = 24
    max_buffer_texts: int = 256
    max_train_batches_per_flush: int = 1
    window_size: int = 24
    stride: int = 12
    truncation_steps: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_path: str = ""
    checkpoint_every_steps: int = 25
    mode: str = "balanced"
    device: str = "cpu"
    add_noise: bool = True
    commit_recurrent_state: bool = True

    @classmethod
    def from_env(cls) -> "Z3CorpusIngestionConfig":
        """Build config from deployment environment variables with safe fallbacks."""
        disabled = _env_bool("DISABLE_Z3_CORPUS_INGESTION", False)
        state_dir = _default_state_dir()
        checkpoint_path = os.getenv("Z3_CORPUS_CHECKPOINT_PATH", str(state_dir / "z3_corpus_ingestor.pt"))
        return cls(
            enabled=not disabled,
            batch_size=_env_int("Z3_CORPUS_TRAIN_BATCH_SIZE", _env_int("LANGUAGE_TRAINING_BATCH_SIZE", 8, minimum=1), minimum=1),
            min_words=_env_int("Z3_CORPUS_MIN_WORDS", _env_int("LANGUAGE_TRAINING_MIN_WORDS", 24, minimum=1), minimum=1),
            max_buffer_texts=_env_int("Z3_CORPUS_BUFFER_TEXTS", 256, minimum=1),
            max_train_batches_per_flush=_env_int("Z3_CORPUS_MAX_TRAIN_BATCHES_PER_FLUSH", 1, minimum=1),
            window_size=_env_int("Z3_CORPUS_WINDOW_SIZE", _env_int("Z3_LANGUAGE_WINDOW_SIZE", 24, minimum=1), minimum=1),
            stride=_env_int("Z3_CORPUS_STRIDE", _env_int("Z3_LANGUAGE_STRIDE", 12, minimum=1), minimum=1),
            truncation_steps=_env_int("Z3_CORPUS_TRUNCATION_STEPS", _env_int("Z3_LANGUAGE_TRUNCATION_STEPS", 16, minimum=1), minimum=1),
            learning_rate=_env_float("Z3_CORPUS_LEARNING_RATE", _env_float("Z3_RUNTIME_LANGUAGE_LR", 0.001, minimum=1e-12), minimum=1e-12),
            weight_decay=_env_float("Z3_CORPUS_WEIGHT_DECAY", 0.0001, minimum=0.0),
            checkpoint_path=checkpoint_path,
            checkpoint_every_steps=_env_int("Z3_CORPUS_CHECKPOINT_EVERY_STEPS", 25, minimum=0),
            mode=(os.getenv("Z3_CORPUS_NEURAL_MODE", "balanced").strip().lower() or "balanced"),
            device=os.getenv("Z3_CORPUS_DEVICE", "cpu").strip() or "cpu",
            add_noise=_env_bool("Z3_CORPUS_ADD_NOISE", True),
            commit_recurrent_state=_env_bool("Z3_CORPUS_COMMIT_RECURRENT_STATE", True),
        )


class Z3CorpusNeuralIngestor:
    """Buffer corpus text and apply real Z³ neural language-window updates."""

    def __init__(
        self,
        config: Optional[Z3CorpusIngestionConfig] = None,
        *,
        model: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        model_provider: Optional[ModelProvider] = None,
        optimizer_factory: Optional[OptimizerFactory] = None,
        trainer: Optional[TrainerFn] = None,
        own_model: bool = False,
    ) -> None:
        self.config = config or Z3CorpusIngestionConfig.from_env()
        self._lock = threading.RLock()
        self._buffer: Deque[Tuple[str, Dict[str, Any]]] = deque(maxlen=max(1, self.config.max_buffer_texts))
        self._last_metrics: Dict[str, float] = {}
        self._last_error: str = ""
        self._trained_steps = 0
        self._texts_seen = 0
        self._dropped_texts = 0
        self._failed_train_attempts = 0
        self._last_train_time = 0.0
        self._last_checkpoint_time = 0.0
        self._last_batch_size = 0
        self._last_provenance: List[Dict[str, Any]] = []
        self._model_provider = model_provider
        self._optimizer_factory = optimizer_factory
        self._trainer = trainer or train_z3_on_language_window
        self._own_model = own_model
        self.model = model
        self.optimizer = optimizer

        if self.config.enabled and self._own_model and self.model is None:
            self._initialize_standalone_runtime()
        if self.config.enabled:
            self.load_checkpoint(load_model=self._own_model)

    @property
    def available(self) -> bool:
        """Return true only when a real trainable neural runtime exists."""
        return self.config.enabled and self._trainer is not None and self._resolve_model() is not None and self._resolve_optimizer() is not None

    def attach_runtime(self, *, model: Any, optimizer: Any) -> None:
        """Attach the ingestor to an externally managed runtime model and optimizer."""
        with self._lock:
            self.model = model
            self.optimizer = optimizer
            self._own_model = False

    def _resolve_model(self) -> Any:
        if self._model_provider is not None:
            try:
                model = self._model_provider()
                if model is not None:
                    self.model = model
            except Exception as exc:
                self._last_error = f"Z3 corpus model provider failed: {exc}"
                return None
        return self.model

    def _resolve_optimizer(self) -> Any:
        model = self.model
        if self._optimizer_factory is not None and model is not None:
            try:
                optimizer = self._optimizer_factory(model, self.config.learning_rate)
                if optimizer is not None:
                    self.optimizer = optimizer
            except Exception as exc:
                self._last_error = f"Z3 corpus optimizer provider failed: {exc}"
                return None
        return self.optimizer

    def _initialize_standalone_runtime(self) -> None:
        if torch is None:
            self._last_error = f"PyTorch unavailable: {_TORCH_IMPORT_ERROR}"
            return
        if _IMPORT_ERROR is not None or Z3NeuralDynamics is None or Z3Config is None:
            self._last_error = f"Z3 neural imports unavailable: {_IMPORT_ERROR}"
            return
        try:
            if self.config.mode == "internal_coherence":
                neural_config = Z3Config.internal_coherence()
            elif self.config.mode == "predictive_runtime":
                neural_config = Z3Config.predictive_runtime()
            else:
                neural_config = Z3Config.balanced()
            self.model = Z3NeuralDynamics(neural_config)
            if hasattr(self.model, "to"):
                self.model.to(self.config.device)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard.
            self.model = None
            self.optimizer = None
            self._last_error = f"Z3 corpus neural initialization failed: {exc}"

    def observe_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Accept one corpus text sample and train if enough buffered text exists."""
        if not self.config.enabled:
            return self.snapshot()
        text = (text or "").strip()
        if len(text.split()) < self.config.min_words:
            with self._lock:
                self._dropped_texts += 1
            return self.snapshot()
        provenance = dict(metadata or {})
        provenance.setdefault("received_at", time.time())
        provenance.setdefault("word_count", len(text.split()))
        with self._lock:
            self._buffer.append((text, provenance))
            self._texts_seen += 1
        return self.flush(force=False)

    def observe_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Accept a language observation dict produced by ``LanguageStream``."""
        text = str(observation.get("text") or observation.get("content") or "")
        metadata = {key: value for key, value in observation.items() if key not in ("text", "content")}
        return self.observe_text(text, metadata=metadata)

    def observe_texts(self, texts: Iterable[str]) -> Dict[str, Any]:
        """Accept multiple corpus samples and train if the configured batch is ready."""
        for text in texts:
            self.observe_text(text)
        return self.flush(force=False)

    def observe_observations(self, observations: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Accept multiple language observations and train if the batch is ready."""
        for observation in observations:
            self.observe_observation(observation)
        return self.flush(force=False)

    def train_texts_now(self, texts: Sequence[str], metadata: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Train immediately on the provided texts without requiring the buffer threshold."""
        if not self.config.enabled:
            return self._training_report(False, reason="disabled", texts=0)
        cleaned = [str(text or "").strip() for text in texts if str(text or "").strip()]
        if not cleaned:
            return self._training_report(False, reason="no_text", texts=0)
        provenance = [dict(item) for item in (metadata or [])]
        while len(provenance) < len(cleaned):
            provenance.append({})
        return self._train_batch(cleaned, provenance)

    def flush(self, *, force: bool = False) -> Dict[str, Any]:
        """Run one or more training updates from the buffered corpus text."""
        if not self.available:
            return self.snapshot()
        trained_now = 0
        while True:
            with self._lock:
                ready = len(self._buffer) >= self.config.batch_size or (force and bool(self._buffer))
                if not ready or trained_now >= self.config.max_train_batches_per_flush:
                    break
                batch_items: List[Tuple[str, Dict[str, Any]]] = []
                while self._buffer and len(batch_items) < self.config.batch_size:
                    batch_items.append(self._buffer.popleft())
            texts = [item[0] for item in batch_items]
            provenance = [item[1] for item in batch_items]
            report = self._train_batch(texts, provenance)
            if not report.get("trained"):
                with self._lock:
                    for item in reversed(batch_items):
                        self._buffer.appendleft(item)
                break
            trained_now += 1
        return self.snapshot()

    def _train_batch(self, texts: Sequence[str], provenance: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        model = self._resolve_model()
        optimizer = self._resolve_optimizer()
        if model is None or optimizer is None or self._trainer is None:
            return self._training_report(False, reason="unavailable", texts=len(texts))
        try:
            metrics = self._trainer(
                model,
                optimizer,
                list(texts),
                truncation_steps=max(1, int(self.config.truncation_steps)),
                window_size=max(1, int(self.config.window_size)),
                stride=max(1, int(self.config.stride)),
                commit_recurrent_state=bool(self.config.commit_recurrent_state),
                add_noise=bool(self.config.add_noise),
            )
        except Exception as exc:
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            with self._lock:
                self._failed_train_attempts += 1
                self._last_error = f"Z3 corpus training failed: {exc}"
            return self._training_report(
                False,
                reason="language_sequence_training_failed",
                texts=len(texts),
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )

        now = time.time()
        numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        with self._lock:
            self._last_metrics = numeric_metrics
            self._trained_steps += 1
            self._last_train_time = now
            self._last_error = ""
            self._last_batch_size = len(texts)
            self._last_provenance = [dict(item) for item in provenance[-5:]]
            trained_steps = self._trained_steps
        if self.config.checkpoint_every_steps > 0 and trained_steps % self.config.checkpoint_every_steps == 0:
            self.save_checkpoint()
        return self._training_report(True, texts=len(texts), metrics=numeric_metrics)

    def _training_report(self, trained: bool, *, texts: int, reason: str = "", metrics: Optional[Dict[str, float]] = None, **extra: Any) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "trained": bool(trained),
            "texts": int(texts),
            "mode": "canonical_corpus_neural_ingestor",
        }
        if reason:
            report["reason"] = reason
        if metrics is not None:
            report["metrics"] = metrics
        report.update(extra)
        return report

    def save_checkpoint(self) -> bool:
        """Persist ingestor metadata plus optional standalone model and optimizer state."""
        if not self.config.enabled or torch is None:
            return False
        path = Path(self.config.checkpoint_path)
        try:
            with self._lock:
                payload: Dict[str, Any] = {
                    "config": asdict(self.config),
                    "ingestor": {
                        "trained_steps": self._trained_steps,
                        "texts_seen": self._texts_seen,
                        "dropped_texts": self._dropped_texts,
                        "failed_train_attempts": self._failed_train_attempts,
                        "last_metrics": dict(self._last_metrics),
                        "last_error": self._last_error,
                        "last_train_time": self._last_train_time,
                        "last_checkpoint_time": time.time(),
                        "last_batch_size": self._last_batch_size,
                        "last_provenance": list(self._last_provenance),
                        "buffer": list(self._buffer),
                    },
                    "own_model": self._own_model,
                }
                model = self.model
                optimizer = self.optimizer
            if self._own_model and model is not None:
                payload["model_state_dict"] = model.state_dict()
                payload["model_config"] = asdict(model.config)
                payload["z3_state"] = model.z3_state.detach().cpu()
                payload["zprime_state"] = model.zprime_state.detach().cpu()
                payload["last_model_metrics"] = model.last_metrics.detach().cpu()
            if optimizer is not None:
                payload["optimizer_state_dict"] = optimizer.state_dict()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(path)
            with self._lock:
                self._last_checkpoint_time = time.time()
            return True
        except Exception as exc:  # pragma: no cover
            with self._lock:
                self._last_error = f"Z3 corpus checkpoint failed: {exc}"
            return False

    def load_checkpoint(self, *, load_model: bool = False) -> bool:
        """Restore persisted ingestor state and, when requested, standalone model state."""
        if torch is None:
            return False
        path = Path(self.config.checkpoint_path)
        if not path.exists():
            return False
        try:
            payload = torch.load(path, map_location=self.config.device)
            ingestor = dict(payload.get("ingestor") or {})
            with self._lock:
                self._trained_steps = int(ingestor.get("trained_steps", self._trained_steps) or 0)
                self._texts_seen = int(ingestor.get("texts_seen", self._texts_seen) or 0)
                self._dropped_texts = int(ingestor.get("dropped_texts", self._dropped_texts) or 0)
                self._failed_train_attempts = int(ingestor.get("failed_train_attempts", self._failed_train_attempts) or 0)
                self._last_metrics = {str(k): float(v) for k, v in dict(ingestor.get("last_metrics") or {}).items() if isinstance(v, (int, float))}
                self._last_error = str(ingestor.get("last_error") or "")
                self._last_train_time = float(ingestor.get("last_train_time", 0.0) or 0.0)
                self._last_checkpoint_time = float(ingestor.get("last_checkpoint_time", 0.0) or 0.0)
                self._last_batch_size = int(ingestor.get("last_batch_size", 0) or 0)
                self._last_provenance = [dict(item) for item in ingestor.get("last_provenance", []) if isinstance(item, dict)]
                for item in ingestor.get("buffer", []):
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        self._buffer.append((str(item[0]), dict(item[1]) if isinstance(item[1], dict) else {}))
            if load_model and self._own_model and self.model is not None and payload.get("model_state_dict") is not None:
                self.model.load_state_dict(payload["model_state_dict"], strict=False)
                if payload.get("z3_state") is not None:
                    self.model.z3_state = payload["z3_state"].to(next(self.model.parameters()).device)
                if payload.get("zprime_state") is not None:
                    self.model.zprime_state = payload["zprime_state"].to(next(self.model.parameters()).device)
                if payload.get("last_model_metrics") is not None:
                    self.model.last_metrics = payload["last_model_metrics"].to(next(self.model.parameters()).device)
            if self.optimizer is not None and payload.get("optimizer_state_dict") is not None:
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            return True
        except Exception as exc:  # pragma: no cover
            with self._lock:
                self._last_error = f"Z3 corpus checkpoint restore failed: {exc}"
            return False

    def snapshot(self) -> Dict[str, Any]:
        """Expose ingestion state for runtime diagnostics."""
        model = self._resolve_model()
        optimizer = self._resolve_optimizer() if model is not None else None
        with self._lock:
            return {
                "enabled": self.config.enabled,
                "available": self.config.enabled and self._trainer is not None and model is not None and optimizer is not None,
                "own_model": self._own_model,
                "buffer_size": len(self._buffer),
                "texts_seen": self._texts_seen,
                "dropped_texts": self._dropped_texts,
                "trained_steps": self._trained_steps,
                "failed_train_attempts": self._failed_train_attempts,
                "last_train_time": self._last_train_time,
                "last_checkpoint_time": self._last_checkpoint_time,
                "last_batch_size": self._last_batch_size,
                "last_error": self._last_error,
                "checkpoint_path": self.config.checkpoint_path,
                "checkpoint_exists": Path(self.config.checkpoint_path).exists() if self.config.checkpoint_path else False,
                "batch_size": self.config.batch_size,
                "window_size": self.config.window_size,
                "stride": self.config.stride,
                "truncation_steps": self.config.truncation_steps,
                "learning_rate": self.config.learning_rate,
                "mode": self.config.mode,
                "device": self.config.device,
                "last_metrics": dict(self._last_metrics),
                "last_provenance": list(self._last_provenance),
            }


__all__ = ["Z3CorpusIngestionConfig", "Z3CorpusNeuralIngestor"]
