"""Canonical neural corpus ingestion for the Z³ runtime.

This module centralizes the path from raw corpus text into real Z³ neural
language-window training. It can either own a standalone neural model or attach
to the live runtime model used by ``main.py``. The ingestor keeps deployment
safety concerns local: environment parsing, bounded buffering, rollback after
failed batches, diagnostics, checkpoint recovery, input pressure control,
deduplication, and training circuit breaking.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
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
CHECKPOINT_VERSION = 2


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


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _text_digest(text: str) -> str:
    return hashlib.blake2b((text or "").encode("utf-8", errors="ignore"), digest_size=16).hexdigest()


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
    max_text_bytes: int = 262_144
    deduplicate_texts: bool = True
    recent_hashes_limit: int = 4096
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_cooldown_seconds: float = 300.0
    slow_train_seconds: float = 30.0
    backlog_warning_ratio: float = 0.80

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
            max_text_bytes=_env_int("Z3_CORPUS_MAX_TEXT_BYTES", 262_144, minimum=256),
            deduplicate_texts=_env_bool("Z3_CORPUS_DEDUPLICATE_TEXTS", True),
            recent_hashes_limit=_env_int("Z3_CORPUS_RECENT_HASHES_LIMIT", 4096, minimum=1),
            circuit_breaker_failure_threshold=_env_int("Z3_CORPUS_FAILURE_THRESHOLD", 3, minimum=1),
            circuit_breaker_cooldown_seconds=_env_float("Z3_CORPUS_COOLDOWN_SECONDS", 300.0, minimum=0.0),
            slow_train_seconds=_env_float("Z3_CORPUS_SLOW_TRAIN_SECONDS", 30.0, minimum=0.0),
            backlog_warning_ratio=_env_float("Z3_CORPUS_BACKLOG_WARNING_RATIO", 0.80, minimum=0.0),
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
        self._last_warning: str = ""
        self._trained_steps = 0
        self._texts_seen = 0
        self._accepted_texts = 0
        self._dropped_texts = 0
        self._duplicate_texts = 0
        self._oversized_texts = 0
        self._dropped_reasons: Dict[str, int] = {}
        self._failed_train_attempts = 0
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._circuit_opened_count = 0
        self._last_train_time = 0.0
        self._last_train_duration = 0.0
        self._max_train_duration = 0.0
        self._last_checkpoint_time = 0.0
        self._last_checkpoint_version = 0
        self._last_checkpoint_recovered_from = ""
        self._last_batch_size = 0
        self._last_provenance: List[Dict[str, Any]] = []
        self._recent_hashes: Deque[str] = deque(maxlen=max(1, self.config.recent_hashes_limit))
        self._recent_hash_set: set[str] = set()
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

    def pause_training(self, seconds: Optional[float] = None, *, reason: str = "operator_pause") -> Dict[str, Any]:
        """Open the training circuit breaker for a bounded cooldown interval."""
        cooldown = self.config.circuit_breaker_cooldown_seconds if seconds is None else max(0.0, float(seconds))
        with self._lock:
            self._circuit_open_until = time.time() + cooldown
            self._circuit_opened_count += 1
            self._last_warning = reason
        return self.snapshot()

    def resume_training(self) -> Dict[str, Any]:
        """Close the training circuit breaker and reset consecutive failure pressure."""
        with self._lock:
            self._circuit_open_until = 0.0
            self._consecutive_failures = 0
            self._last_warning = ""
        return self.snapshot()

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
        accepted = self._prepare_text(text, metadata)
        if accepted is None:
            return self.snapshot()
        with self._lock:
            if len(self._buffer) >= self.config.max_buffer_texts:
                self._record_drop("buffer_full")
                return self.snapshot()
            self._buffer.append(accepted)
            self._accepted_texts += 1
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
        """Train immediately on accepted texts without requiring the buffer threshold."""
        if not self.config.enabled:
            return self._training_report(False, reason="disabled", texts=0)
        metadata_items = list(metadata or [])
        accepted: List[Tuple[str, Dict[str, Any]]] = []
        for index, text in enumerate(texts):
            item_metadata = metadata_items[index] if index < len(metadata_items) else {}
            prepared = self._prepare_text(str(text or ""), item_metadata)
            if prepared is not None:
                accepted.append(prepared)
        if not accepted:
            return self._training_report(False, reason="all_texts_rejected", texts=0, health_state=self.health_state())
        return self._train_batch([item[0] for item in accepted], [item[1] for item in accepted])

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
                        if len(self._buffer) < self.config.max_buffer_texts:
                            self._buffer.appendleft(item)
                        else:
                            self._record_drop("rollback_buffer_full")
                break
            trained_now += 1
        return self.snapshot()

    def _prepare_text(self, text: str, metadata: Optional[Dict[str, Any]]) -> Optional[Tuple[str, Dict[str, Any]]]:
        text = (text or "").strip()
        with self._lock:
            self._texts_seen += 1
        if not text:
            self._record_drop("empty_text")
            return None
        byte_count = len(text.encode("utf-8", errors="ignore"))
        if byte_count > self.config.max_text_bytes:
            with self._lock:
                self._oversized_texts += 1
            self._record_drop("oversized_text")
            return None
        word_count = len(text.split())
        if word_count < self.config.min_words:
            self._record_drop("too_few_words")
            return None
        digest = _text_digest(text)
        if self.config.deduplicate_texts:
            with self._lock:
                if digest in self._recent_hash_set:
                    self._duplicate_texts += 1
                    self._record_drop("duplicate_text")
                    return None
                self._remember_hash(digest)
        provenance = self._normalize_provenance(metadata or {}, digest=digest, byte_count=byte_count, word_count=word_count)
        return text, provenance

    def _normalize_provenance(self, metadata: Dict[str, Any], *, digest: str, byte_count: int, word_count: int) -> Dict[str, Any]:
        allowed_scalar = (str, int, float, bool, type(None))
        provenance: Dict[str, Any] = {}
        for key, value in dict(metadata or {}).items():
            if key in ("text", "content"):
                continue
            safe_key = str(key)[:80]
            if isinstance(value, allowed_scalar):
                provenance[safe_key] = value
            else:
                provenance[safe_key] = str(value)[:500]
        provenance.setdefault("received_at", time.time())
        provenance["word_count"] = int(word_count)
        provenance["byte_count"] = int(byte_count)
        provenance["text_hash"] = digest
        provenance.setdefault("source", "unknown")
        return provenance

    def _remember_hash(self, digest: str) -> None:
        if len(self._recent_hashes) == self._recent_hashes.maxlen and self._recent_hashes:
            old = self._recent_hashes.popleft()
            self._recent_hash_set.discard(old)
        self._recent_hashes.append(digest)
        self._recent_hash_set.add(digest)

    def _record_drop(self, reason: str) -> None:
        with self._lock:
            self._dropped_texts += 1
            self._dropped_reasons[reason] = self._dropped_reasons.get(reason, 0) + 1

    def _circuit_open(self) -> bool:
        return time.time() < self._circuit_open_until

    def _open_circuit(self, reason: str) -> None:
        with self._lock:
            self._circuit_open_until = time.time() + float(self.config.circuit_breaker_cooldown_seconds)
            self._circuit_opened_count += 1
            self._last_warning = reason

    def _train_batch(self, texts: Sequence[str], provenance: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if self._circuit_open():
            return self._training_report(
                False,
                reason="circuit_breaker_open",
                texts=len(texts),
                circuit_open_until=self._circuit_open_until,
                health_state=self.health_state(),
            )
        model = self._resolve_model()
        optimizer = self._resolve_optimizer()
        if model is None or optimizer is None or self._trainer is None:
            return self._training_report(False, reason="unavailable", texts=len(texts), health_state=self.health_state())
        started_at = time.time()
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
            duration = time.time() - started_at
            if not self._model_state_is_finite(model):
                raise FloatingPointError("non_finite_model_state_after_corpus_training")
        except Exception as exc:
            duration = time.time() - started_at
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            with self._lock:
                self._failed_train_attempts += 1
                self._consecutive_failures += 1
                self._last_train_duration = duration
                self._max_train_duration = max(self._max_train_duration, duration)
                self._last_error = f"Z3 corpus training failed: {exc}"
                consecutive = self._consecutive_failures
            if consecutive >= self.config.circuit_breaker_failure_threshold:
                self._open_circuit("failure_threshold_exceeded")
            return self._training_report(
                False,
                reason="language_sequence_training_failed",
                texts=len(texts),
                error_type=type(exc).__name__,
                error=str(exc)[:500],
                duration_seconds=round(duration, 6),
                health_state=self.health_state(),
            )

        now = time.time()
        numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}
        warning = ""
        if self.config.slow_train_seconds > 0 and duration > self.config.slow_train_seconds:
            warning = "slow_training_batch"
        with self._lock:
            self._last_metrics = numeric_metrics
            self._trained_steps += 1
            self._consecutive_failures = 0
            self._last_train_time = now
            self._last_train_duration = duration
            self._max_train_duration = max(self._max_train_duration, duration)
            self._last_error = ""
            self._last_warning = warning
            self._last_batch_size = len(texts)
            self._last_provenance = [dict(item) for item in provenance[-5:]]
            trained_steps = self._trained_steps
        if self.config.checkpoint_every_steps > 0 and trained_steps % self.config.checkpoint_every_steps == 0:
            self.save_checkpoint()
        return self._training_report(True, texts=len(texts), metrics=numeric_metrics, duration_seconds=round(duration, 6), health_state=self.health_state())

    def _model_state_is_finite(self, model: Any) -> bool:
        if torch is None or model is None:
            return True
        try:
            with torch.no_grad():
                if hasattr(model, "parameters"):
                    for parameter in model.parameters():
                        if parameter is not None and not bool(torch.isfinite(parameter).all().item()):
                            return False
                if hasattr(model, "buffers"):
                    for buffer in model.buffers():
                        if buffer is not None and buffer.numel() > 0 and not bool(torch.isfinite(buffer).all().item()):
                            return False
        except Exception:
            return True
        return True

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

    def _payload(self) -> Dict[str, Any]:
        with self._lock:
            ingestor = {
                "trained_steps": self._trained_steps,
                "texts_seen": self._texts_seen,
                "accepted_texts": self._accepted_texts,
                "dropped_texts": self._dropped_texts,
                "duplicate_texts": self._duplicate_texts,
                "oversized_texts": self._oversized_texts,
                "dropped_reasons": dict(self._dropped_reasons),
                "failed_train_attempts": self._failed_train_attempts,
                "consecutive_failures": self._consecutive_failures,
                "circuit_open_until": self._circuit_open_until,
                "circuit_opened_count": self._circuit_opened_count,
                "last_metrics": dict(self._last_metrics),
                "last_error": self._last_error,
                "last_warning": self._last_warning,
                "last_train_time": self._last_train_time,
                "last_train_duration": self._last_train_duration,
                "max_train_duration": self._max_train_duration,
                "last_checkpoint_time": time.time(),
                "last_checkpoint_version": CHECKPOINT_VERSION,
                "last_checkpoint_recovered_from": self._last_checkpoint_recovered_from,
                "last_batch_size": self._last_batch_size,
                "last_provenance": list(self._last_provenance),
                "buffer": list(self._buffer),
                "recent_hashes": list(self._recent_hashes),
            }
            model = self.model
            optimizer = self.optimizer
        payload: Dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "saved_at": time.time(),
            "config": asdict(self.config),
            "config_hash": _stable_hash(asdict(self.config)),
            "ingestor": ingestor,
            "own_model": self._own_model,
        }
        if self._own_model and model is not None:
            payload["model_state_dict"] = model.state_dict()
            payload["model_config"] = asdict(model.config)
            payload["z3_state"] = model.z3_state.detach().cpu()
            payload["zprime_state"] = model.zprime_state.detach().cpu()
            payload["last_model_metrics"] = model.last_metrics.detach().cpu()
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        return payload

    def save_checkpoint(self) -> bool:
        """Persist ingestor metadata plus optional standalone model and optimizer state."""
        if not self.config.enabled or torch is None:
            return False
        path = Path(self.config.checkpoint_path)
        if not str(path):
            return False
        try:
            payload = self._payload()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            previous_path = path.with_suffix(path.suffix + ".previous")
            torch.save(payload, tmp_path)
            if path.exists():
                shutil.copy2(path, previous_path)
            tmp_path.replace(path)
            with self._lock:
                self._last_checkpoint_time = time.time()
                self._last_checkpoint_version = CHECKPOINT_VERSION
            return True
        except Exception as exc:  # pragma: no cover
            with self._lock:
                self._last_error = f"Z3 corpus checkpoint failed: {exc}"
            return False

    def load_checkpoint(self, *, load_model: bool = False) -> bool:
        """Restore persisted ingestor state with previous-checkpoint fallback."""
        if torch is None:
            return False
        path = Path(self.config.checkpoint_path)
        if not path.exists():
            previous_path = path.with_suffix(path.suffix + ".previous")
            if not previous_path.exists():
                return False
            path = previous_path
        candidates = [path]
        previous = path.with_suffix(path.suffix + ".previous") if not str(path).endswith(".previous") else path
        if previous.exists() and previous not in candidates:
            candidates.append(previous)
        errors: List[str] = []
        for candidate in candidates:
            try:
                payload = torch.load(candidate, map_location=self.config.device)
                self._restore_payload(payload, load_model=load_model)
                with self._lock:
                    self._last_checkpoint_recovered_from = str(candidate)
                    self._last_error = ""
                return True
            except Exception as exc:  # pragma: no cover - intentionally handles corrupt checkpoints.
                errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
                self._quarantine_checkpoint(candidate)
        with self._lock:
            self._last_error = "Z3 corpus checkpoint restore failed: " + " | ".join(errors)[:800]
        return False

    def _restore_payload(self, payload: Dict[str, Any], *, load_model: bool) -> None:
        version = int(payload.get("checkpoint_version", 1) or 1)
        ingestor = dict(payload.get("ingestor") or {})
        with self._lock:
            self._trained_steps = int(ingestor.get("trained_steps", self._trained_steps) or 0)
            self._texts_seen = int(ingestor.get("texts_seen", self._texts_seen) or 0)
            self._accepted_texts = int(ingestor.get("accepted_texts", self._accepted_texts) or 0)
            self._dropped_texts = int(ingestor.get("dropped_texts", self._dropped_texts) or 0)
            self._duplicate_texts = int(ingestor.get("duplicate_texts", self._duplicate_texts) or 0)
            self._oversized_texts = int(ingestor.get("oversized_texts", self._oversized_texts) or 0)
            self._dropped_reasons = {str(k): int(v) for k, v in dict(ingestor.get("dropped_reasons") or {}).items() if isinstance(v, int)}
            self._failed_train_attempts = int(ingestor.get("failed_train_attempts", self._failed_train_attempts) or 0)
            self._consecutive_failures = int(ingestor.get("consecutive_failures", self._consecutive_failures) or 0)
            self._circuit_open_until = float(ingestor.get("circuit_open_until", self._circuit_open_until) or 0.0)
            self._circuit_opened_count = int(ingestor.get("circuit_opened_count", self._circuit_opened_count) or 0)
            self._last_metrics = {str(k): float(v) for k, v in dict(ingestor.get("last_metrics") or {}).items() if isinstance(v, (int, float))}
            self._last_error = str(ingestor.get("last_error") or "")
            self._last_warning = str(ingestor.get("last_warning") or "")
            self._last_train_time = float(ingestor.get("last_train_time", 0.0) or 0.0)
            self._last_train_duration = float(ingestor.get("last_train_duration", 0.0) or 0.0)
            self._max_train_duration = float(ingestor.get("max_train_duration", 0.0) or 0.0)
            self._last_checkpoint_time = float(ingestor.get("last_checkpoint_time", 0.0) or 0.0)
            self._last_checkpoint_version = int(ingestor.get("last_checkpoint_version", version) or version)
            self._last_batch_size = int(ingestor.get("last_batch_size", 0) or 0)
            self._last_provenance = [dict(item) for item in ingestor.get("last_provenance", []) if isinstance(item, dict)]
            self._buffer.clear()
            for item in ingestor.get("buffer", []):
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    self._buffer.append((str(item[0]), dict(item[1]) if isinstance(item[1], dict) else {}))
            self._recent_hashes.clear()
            self._recent_hash_set.clear()
            for digest in ingestor.get("recent_hashes", []):
                if isinstance(digest, str):
                    self._remember_hash(digest)
            if not self._recent_hashes:
                for _, provenance in self._buffer:
                    digest = str(provenance.get("text_hash") or "")
                    if digest:
                        self._remember_hash(digest)
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

    def _quarantine_checkpoint(self, path: Path) -> None:
        try:
            if path.exists():
                quarantine = path.with_suffix(path.suffix + f".corrupt.{int(time.time())}")
                path.replace(quarantine)
        except Exception:
            pass

    def health_state(self) -> str:
        """Return a compact production health state for monitoring."""
        now = time.time()
        if not self.config.enabled:
            return "disabled"
        if self._circuit_open_until > now:
            return "circuit_open"
        if self._last_error:
            return "degraded"
        if len(self._buffer) >= max(1, int(self.config.max_buffer_texts * self.config.backlog_warning_ratio)):
            return "backlogged"
        if self._last_warning:
            return "warning"
        if not self.available:
            return "unavailable"
        return "ready"

    def snapshot(self) -> Dict[str, Any]:
        """Expose ingestion state for runtime diagnostics."""
        model = self._resolve_model()
        optimizer = self._resolve_optimizer() if model is not None else None
        now = time.time()
        checkpoint_path = Path(self.config.checkpoint_path) if self.config.checkpoint_path else None
        with self._lock:
            return {
                "enabled": self.config.enabled,
                "available": self.config.enabled and self._trainer is not None and model is not None and optimizer is not None,
                "health_state": self.health_state(),
                "own_model": self._own_model,
                "buffer_size": len(self._buffer),
                "buffer_capacity": self.config.max_buffer_texts,
                "backlog_ratio": round(len(self._buffer) / max(1, self.config.max_buffer_texts), 6),
                "texts_seen": self._texts_seen,
                "accepted_texts": self._accepted_texts,
                "dropped_texts": self._dropped_texts,
                "duplicate_texts": self._duplicate_texts,
                "oversized_texts": self._oversized_texts,
                "dropped_reasons": dict(self._dropped_reasons),
                "trained_steps": self._trained_steps,
                "failed_train_attempts": self._failed_train_attempts,
                "consecutive_failures": self._consecutive_failures,
                "circuit_open": self._circuit_open_until > now,
                "circuit_open_until": self._circuit_open_until,
                "circuit_opened_count": self._circuit_opened_count,
                "last_train_time": self._last_train_time,
                "last_train_duration": self._last_train_duration,
                "max_train_duration": self._max_train_duration,
                "last_checkpoint_time": self._last_checkpoint_time,
                "last_checkpoint_version": self._last_checkpoint_version,
                "last_checkpoint_recovered_from": self._last_checkpoint_recovered_from,
                "last_batch_size": self._last_batch_size,
                "last_error": self._last_error,
                "last_warning": self._last_warning,
                "checkpoint_path": self.config.checkpoint_path,
                "checkpoint_exists": checkpoint_path.exists() if checkpoint_path else False,
                "checkpoint_previous_exists": checkpoint_path.with_suffix(checkpoint_path.suffix + ".previous").exists() if checkpoint_path else False,
                "batch_size": self.config.batch_size,
                "window_size": self.config.window_size,
                "stride": self.config.stride,
                "truncation_steps": self.config.truncation_steps,
                "learning_rate": self.config.learning_rate,
                "mode": self.config.mode,
                "device": self.config.device,
                "max_text_bytes": self.config.max_text_bytes,
                "deduplicate_texts": self.config.deduplicate_texts,
                "last_metrics": dict(self._last_metrics),
                "last_provenance": list(self._last_provenance),
            }


__all__ = ["CHECKPOINT_VERSION", "Z3CorpusIngestionConfig", "Z3CorpusNeuralIngestor"]
