"""
Z³ language corpus training adapter
===================================

This module converts natural-language corpus text into temporal embedding streams
that train ``Z3NeuralDynamics`` through its native ``train_sequence_window`` API.
The stream is deliberately lightweight and deterministic, so it can operate as an
always-on ingestion bridge without depending on an external embedding provider.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

try:  # pragma: no cover - depends on deployment image.
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from Z3_neural_dynamics import prepare_embedding_pairs
except Exception as exc:  # pragma: no cover
    prepare_embedding_pairs = None  # type: ignore[assignment]
    _Z3_IMPORT_ERROR = exc
else:
    _Z3_IMPORT_ERROR = None

_TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
_SEGMENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class LanguageAdapterConfig:
    """Configuration for deterministic corpus-to-vector conversion."""

    input_dim: int = 16
    window_size: int = 24
    stride: int = 12
    min_tokens_per_step: int = 1
    lowercase: bool = True
    l2_normalize: bool = True
    include_position: bool = True

    def __post_init__(self) -> None:
        if self.input_dim < 4:
            raise ValueError("input_dim must be at least 4")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.min_tokens_per_step < 1:
            raise ValueError("min_tokens_per_step must be >= 1")


class LanguageEmbeddingAdapter:
    """Build Z³-compatible embedding streams from raw language."""

    def __init__(self, config: Optional[LanguageAdapterConfig] = None) -> None:
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError("PyTorch is required for language training tensors") from _TORCH_IMPORT_ERROR
        self.config = config or LanguageAdapterConfig()

    def encode_text(self, text: str) -> "torch.Tensor":
        """Encode one document or segment as ``[steps, input_dim]``."""
        tokens = self._tokenize(text)
        if len(tokens) < self.config.min_tokens_per_step:
            raise ValueError("text does not contain enough tokens to produce a language stream")
        windows = self._windows(tokens)
        vectors = [self._encode_window(window, index, len(windows)) for index, window in enumerate(windows)]
        stream = torch.stack(vectors, dim=0)
        if stream.shape[0] < 2:
            duplicate = stream.clone()
            if self.config.include_position:
                duplicate[:, -1] = 1.0
            stream = torch.cat([stream, duplicate], dim=0)
        return stream

    def encode_texts(self, texts: Sequence[str]) -> "torch.Tensor":
        """Encode multiple texts as padded ``[batch, steps, input_dim]`` streams."""
        cleaned = [text for text in texts if text and text.strip()]
        if not cleaned:
            raise ValueError("texts must contain at least one non-empty text item")
        streams = [self.encode_text(text) for text in cleaned]
        max_steps = max(stream.shape[0] for stream in streams)
        padded = []
        for stream in streams:
            if stream.shape[0] < max_steps:
                pad = stream[-1:, :].expand(max_steps - stream.shape[0], -1)
                stream = torch.cat([stream, pad], dim=0)
            padded.append(stream)
        return torch.stack(padded, dim=0)

    def prepare_training_pairs(self, texts: Sequence[str]) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return flattened next-step pairs for low-level training utilities."""
        if prepare_embedding_pairs is None:  # pragma: no cover
            raise ModuleNotFoundError("Z³ neural dynamics imports are unavailable") from _Z3_IMPORT_ERROR
        return prepare_embedding_pairs(self.encode_texts(texts))

    def _tokenize(self, text: str) -> List[str]:
        text = text or ""
        if self.config.lowercase:
            text = text.lower()
        return [token for token in _TOKEN_RE.findall(text) if token.strip()]

    def _windows(self, tokens: Sequence[str]) -> List[List[str]]:
        windows: List[List[str]] = []
        for start in range(0, len(tokens), self.config.stride):
            window = list(tokens[start : start + self.config.window_size])
            if len(window) >= self.config.min_tokens_per_step:
                windows.append(window)
            if start + self.config.window_size >= len(tokens):
                break
        return windows or [list(tokens)]

    def _encode_window(self, tokens: Sequence[str], index: int, total_windows: int) -> "torch.Tensor":
        cfg = self.config
        vector = torch.zeros(cfg.input_dim, dtype=torch.float32)
        lexical_dims = max(1, cfg.input_dim - 4)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % lexical_dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign / math.sqrt(max(1, len(tokens)))

        lengths = [len(token) for token in tokens]
        alpha_count = sum(1 for token in tokens if any(char.isalpha() for char in token))
        punct_count = sum(1 for token in tokens if not any(char.isalnum() for char in token))
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        mean_length = sum(lengths) / max(1, len(lengths))
        tail = cfg.input_dim - 4
        vector[tail] = min(1.0, mean_length / 12.0)
        vector[tail + 1] = unique_ratio
        vector[tail + 2] = punct_count / max(1, len(tokens))
        vector[tail + 3] = alpha_count / max(1, len(tokens))

        if cfg.include_position and cfg.input_dim >= 2:
            position = index / max(1, total_windows - 1)
            vector[-2] = math.sin(position * math.pi)
            vector[-1] = math.cos(position * math.pi)

        if cfg.l2_normalize:
            norm = torch.norm(vector, p=2).clamp_min(1e-6)
            vector = vector / norm
        return vector


def split_language_corpus(text: str) -> List[str]:
    """Split raw corpus text into sentence-like non-empty segments."""
    units = [unit.strip() for unit in _SEGMENT_RE.split(text or "") if unit.strip()]
    return units or ([text.strip()] if text and text.strip() else [])


def build_language_embedding_stream(
    texts: str | Sequence[str],
    *,
    input_dim: int,
    window_size: int = 24,
    stride: int = 12,
) -> "torch.Tensor":
    """Produce a Z³ language stream tensor from one text or many texts."""
    adapter = LanguageEmbeddingAdapter(LanguageAdapterConfig(input_dim=input_dim, window_size=window_size, stride=stride))
    if isinstance(texts, str):
        return adapter.encode_text(texts)
    return adapter.encode_texts(list(texts))


def train_z3_on_language_window(
    model,
    optimizer,
    texts: str | Sequence[str],
    *,
    truncation_steps: int = 16,
    window_size: int = 24,
    stride: int = 12,
    commit_recurrent_state: bool = True,
    add_noise: bool = True,
) -> dict[str, float]:
    """Train ``Z3NeuralDynamics`` directly on a natural-language corpus window."""
    stream = build_language_embedding_stream(
        texts,
        input_dim=model.config.input_dim,
        window_size=window_size,
        stride=stride,
    )
    return model.train_sequence_window(
        optimizer,
        stream,
        truncation_steps=truncation_steps,
        commit_recurrent_state=commit_recurrent_state,
        add_noise=add_noise,
    )
