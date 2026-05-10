"""
Native Online World Model
=========================
A lightweight self-supervised world model for the cognitive mesh.

The model learns directly from the observation stream without external model
APIs. It encodes observations into a fixed numeric vector, maintains a compact
latent state, predicts the next latent state, reconstructs inputs from an online
basis, and tracks memory-recall distance to prior latent families.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class WorldModelOutput:
    """Structured output from one world-model update."""

    iteration: int
    latent_state: List[float]
    total_loss: float
    prediction_loss: float
    reconstruction_loss: float
    memory_loss: float
    coherence_alignment_loss: float
    compression_ratio: float
    novelty: float
    nearest_memory_distance: float
    latent_norm: float
    learning_rate: float
    feature_count: int
    memory_size: int
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OnlineWorldModel:
    """Online PCA-style world model with next-state prediction.

    The intent is not to be a large foundation model. It is a native, measurable
    scaffold that gives the mesh a shared learned latent state and a real
    self-supervised objective.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        latent_dim: int = 8,
        learning_rate: float = 0.015,
        memory_size: int = 256,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.learning_rate = float(learning_rate)
        self.iteration = 0

        rng = np.random.default_rng(1337)
        basis = rng.normal(0.0, 0.1, size=(self.latent_dim, self.feature_dim))
        self.basis = self._orthonormalize_rows(basis)
        self.transition = np.eye(self.latent_dim, dtype=float) * 0.92

        self.feature_index: Dict[str, int] = {}
        self.feature_mean = np.zeros(self.feature_dim, dtype=float)
        self.feature_var = np.ones(self.feature_dim, dtype=float)
        self.feature_count = 0

        self.previous_latent: Optional[np.ndarray] = None
        self.predicted_latent: Optional[np.ndarray] = None
        self.current_latent = np.zeros(self.latent_dim, dtype=float)

        self.memory: deque = deque(maxlen=memory_size)
        self.loss_history: deque = deque(maxlen=512)
        self.last_output: Optional[WorldModelOutput] = None

    def observe(
        self,
        observation: Dict[str, Any],
        domain: str = "general",
        coordinator_state: Optional[Any] = None,
    ) -> WorldModelOutput:
        """Update the world model from one observation and return losses."""
        self.iteration += 1
        x_raw = self._encode_observation(observation, domain)
        x = self._normalize_update(x_raw)

        latent = self.basis @ x
        latent = np.tanh(latent)
        latent_norm = float(np.linalg.norm(latent) / max(math.sqrt(self.latent_dim), 1e-9))

        if self.predicted_latent is None:
            prediction_loss = 0.5
        else:
            prediction_loss = float(np.mean((latent - self.predicted_latent) ** 2))

        reconstruction = self.basis.T @ latent
        reconstruction_loss = float(np.mean((x - reconstruction) ** 2))

        nearest_distance = self._nearest_memory_distance(latent)
        memory_loss = float(min(1.0, nearest_distance))
        novelty = memory_loss

        coherence_alignment_loss = self._coherence_alignment_loss(latent_norm, coordinator_state)

        total_loss = float(
            (0.40 * min(1.0, prediction_loss))
            + (0.25 * min(1.0, reconstruction_loss))
            + (0.20 * memory_loss)
            + (0.15 * coherence_alignment_loss)
        )

        self._learn_basis(x, latent, reconstruction)
        self._learn_transition(latent)
        self.predicted_latent = self.transition @ latent
        self.previous_latent = latent.copy()
        self.current_latent = latent.copy()
        self.memory.append(latent.copy())

        output = WorldModelOutput(
            iteration=self.iteration,
            latent_state=[round(float(v), 6) for v in latent.tolist()],
            total_loss=round(total_loss, 6),
            prediction_loss=round(float(min(1.0, prediction_loss)), 6),
            reconstruction_loss=round(float(min(1.0, reconstruction_loss)), 6),
            memory_loss=round(memory_loss, 6),
            coherence_alignment_loss=round(coherence_alignment_loss, 6),
            compression_ratio=round(self.latent_dim / max(len(self.feature_index), 1), 6),
            novelty=round(novelty, 6),
            nearest_memory_distance=round(nearest_distance, 6),
            latent_norm=round(latent_norm, 6),
            learning_rate=round(self.learning_rate, 8),
            feature_count=len(self.feature_index),
            memory_size=len(self.memory),
            timestamp=time.time(),
        )
        self.last_output = output
        self.loss_history.append(output.to_dict())
        return output

    def get_state(self) -> Dict[str, Any]:
        latest = self.last_output.to_dict() if self.last_output else None
        recent = list(self.loss_history)[-50:]
        avg_loss = 0.0
        if recent:
            avg_loss = sum(float(r.get("total_loss", 0.0)) for r in recent) / len(recent)
        return {
            "iteration": self.iteration,
            "latest": latest,
            "recent": recent,
            "average_recent_loss": round(avg_loss, 6),
            "feature_count": len(self.feature_index),
            "latent_dim": self.latent_dim,
            "feature_dim": self.feature_dim,
            "memory_size": len(self.memory),
            "learning_rate": self.learning_rate,
        }

    def save_state(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "feature_index": dict(self.feature_index),
            "feature_mean": self.feature_mean.tolist(),
            "feature_var": self.feature_var.tolist(),
            "feature_count": self.feature_count,
            "basis": self.basis.tolist(),
            "transition": self.transition.tolist(),
            "current_latent": self.current_latent.tolist(),
            "predicted_latent": self.predicted_latent.tolist() if self.predicted_latent is not None else None,
            "memory": [m.tolist() for m in list(self.memory)],
            "loss_history": list(self.loss_history),
            "learning_rate": self.learning_rate,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.iteration = int(state.get("iteration", self.iteration))
        self.feature_index = dict(state.get("feature_index", self.feature_index))
        self.feature_mean = np.array(state.get("feature_mean", self.feature_mean), dtype=float)[: self.feature_dim]
        self.feature_var = np.array(state.get("feature_var", self.feature_var), dtype=float)[: self.feature_dim]
        self.feature_count = int(state.get("feature_count", self.feature_count))
        self.basis = np.array(state.get("basis", self.basis), dtype=float)[: self.latent_dim, : self.feature_dim]
        self.transition = np.array(state.get("transition", self.transition), dtype=float)[: self.latent_dim, : self.latent_dim]
        self.current_latent = np.array(state.get("current_latent", self.current_latent), dtype=float)[: self.latent_dim]
        pred = state.get("predicted_latent")
        self.predicted_latent = np.array(pred, dtype=float)[: self.latent_dim] if pred is not None else None
        self.memory.clear()
        for m in state.get("memory", [])[-self.memory.maxlen :]:
            self.memory.append(np.array(m, dtype=float)[: self.latent_dim])
        self.loss_history.clear()
        for item in state.get("loss_history", [])[-self.loss_history.maxlen :]:
            self.loss_history.append(dict(item))
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))

    def _encode_observation(self, observation: Dict[str, Any], domain: str) -> np.ndarray:
        x = np.zeros(self.feature_dim, dtype=float)
        flat = self._flatten(observation)
        flat[f"domain::{domain}"] = 1.0
        for key in sorted(flat.keys()):
            value = flat[key]
            if key not in self.feature_index:
                if len(self.feature_index) >= self.feature_dim:
                    continue
                self.feature_index[key] = len(self.feature_index)
            idx = self.feature_index[key]
            if idx >= self.feature_dim:
                continue
            x[idx] = self._value_to_float(value)
        return np.clip(x, -10.0, 10.0)

    def _flatten(self, obj: Any, prefix: str = "") -> Dict[str, Any]:
        items: Dict[str, Any] = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in {"timestamp", "source_url", "csv_path"}:
                    continue
                new_prefix = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, dict):
                    items.update(self._flatten(value, new_prefix))
                elif isinstance(value, (list, tuple)):
                    for i, sub in enumerate(value[:4]):
                        items[f"{new_prefix}.{i}"] = sub
                else:
                    items[new_prefix] = value
        return items

    def _value_to_float(self, value: Any) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        text = str(value)
        if not text:
            return 0.0
        h = 2166136261
        for ch in text[:128]:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return ((h / 0xFFFFFFFF) * 2.0) - 1.0

    def _normalize_update(self, x: np.ndarray) -> np.ndarray:
        self.feature_count += 1
        alpha = 1.0 / min(self.feature_count, 1000)
        delta = x - self.feature_mean
        self.feature_mean += alpha * delta
        self.feature_var = (1.0 - alpha) * self.feature_var + alpha * (delta ** 2)
        std = np.sqrt(np.maximum(self.feature_var, 1e-6))
        return np.clip((x - self.feature_mean) / std, -5.0, 5.0)

    def _learn_basis(self, x: np.ndarray, latent: np.ndarray, reconstruction: np.ndarray) -> None:
        error = x - reconstruction
        self.basis += self.learning_rate * np.outer(latent, error)
        self.basis = self._orthonormalize_rows(self.basis)

    def _learn_transition(self, latent: np.ndarray) -> None:
        if self.previous_latent is None:
            return
        pred = self.transition @ self.previous_latent
        err = latent - pred
        denom = float(np.dot(self.previous_latent, self.previous_latent) + 1e-6)
        self.transition += self.learning_rate * np.outer(err, self.previous_latent) / denom
        self.transition = np.clip(self.transition, -2.0, 2.0)

    def _nearest_memory_distance(self, latent: np.ndarray) -> float:
        if not self.memory:
            return 1.0
        distances = [float(np.linalg.norm(latent - m) / max(math.sqrt(self.latent_dim), 1e-9)) for m in self.memory]
        return min(distances) if distances else 1.0

    def _coherence_alignment_loss(self, latent_norm: float, coordinator_state: Optional[Any]) -> float:
        if coordinator_state is None:
            return 0.5
        phi = float(getattr(coordinator_state, "phi", 0.5) or 0.5)
        sigma = float(getattr(coordinator_state, "sigma", 0.5) or 0.5)
        target = max(0.0, min(1.0, phi * (1.0 - 0.5 * sigma)))
        return min(1.0, abs(latent_norm - target))

    def _orthonormalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        rows = []
        for row in matrix:
            v = row.astype(float).copy()
            for prev in rows:
                v -= np.dot(v, prev) * prev
            norm = np.linalg.norm(v)
            if norm < 1e-9:
                v = np.zeros_like(v)
                v[len(rows) % len(v)] = 1.0
            else:
                v /= norm
            rows.append(v)
        return np.vstack(rows)
