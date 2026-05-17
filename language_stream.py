"""Language stream for Z³ training observations.

This module provides a synchronous language source for the Z³ neural network
runtime. It supports operator-supplied local text and a real remote corpus stream
(default: Project Gutenberg English via Hugging Face Datasets). The stream emits
raw text observations and can be consumed by the runtime's direct language
sequence trainer.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

_TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
_SEGMENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_SPACE_RE = re.compile(r"\s+")
_START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)


class LanguageStream:
    """Synchronous real-language source for Z³ observations."""

    DEFAULT_DOMAIN = "language:corpus"

    def __init__(self) -> None:
        state_dir = Path(os.environ.get("Z3_STATE_DIR", os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "data")))
        self.corpus_path = os.getenv("LANGUAGE_TRAINING_CORPUS_PATH", "").strip()
        self.inline_text = os.getenv("LANGUAGE_TRAINING_TEXT", "").strip()
        self.cache_path = Path(os.getenv("LANGUAGE_TRAINING_CACHE", str(state_dir / "language_corpus.txt")))
        self.batch_size = int(os.getenv("LANGUAGE_TRAINING_BATCH_SIZE", "25"))
        self.max_segments = int(os.getenv("LANGUAGE_TRAINING_MAX_SEGMENTS", "100000"))
        self.min_words = int(os.getenv("LANGUAGE_TRAINING_MIN_WORDS", "24"))
        self.dataset_name = os.getenv("LANGUAGE_TRAINING_DATASET", os.getenv("Z3_CORPUS_DATASET", "manu/project_gutenberg")).strip()
        self.dataset_config = os.getenv("LANGUAGE_TRAINING_DATASET_CONFIG", os.getenv("Z3_CORPUS_DATASET_CONFIG", "")).strip()
        self.dataset_split = os.getenv("LANGUAGE_TRAINING_DATASET_SPLIT", os.getenv("Z3_CORPUS_DATASET_SPLIT", "en")).strip()
        self.text_field = os.getenv("LANGUAGE_TRAINING_TEXT_FIELD", os.getenv("Z3_CORPUS_TEXT_FIELD", "text")).strip()
        self.disable_remote = os.getenv("DISABLE_REMOTE_CORPUS_STREAM", "").lower() in ("1", "true", "yes", "on")
        self._segments: List[str] = []
        self._offset = 0
        self._loaded_at: Optional[float] = None
        self._last_batch_at: Optional[float] = None
        self._dataset_iter: Optional[Iterator[Dict[str, Any]]] = None
        self._source = "idle"
        self._last_error: Optional[str] = None
        self._total_emitted = 0

    def ensure_loaded(self) -> Dict[str, Any]:
        """Load configured text or initialize the configured remote corpus stream."""
        text = self._load_text()
        if text:
            self._segments = self._split_segments(text)[: self.max_segments]
            self._source = "inline_text" if self.inline_text else ("local_file" if self.corpus_path else "cache")
        elif self.dataset_name and not self.disable_remote:
            self._initialize_dataset_stream()
        self._loaded_at = time.time()
        return self.status()

    def _load_text(self) -> str:
        if self.corpus_path:
            path = Path(self.corpus_path).expanduser()
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8", errors="ignore")
            self._last_error = f"Corpus path not found: {path}"
        if self.inline_text:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(self.inline_text, encoding="utf-8")
            return self.inline_text
        if self.cache_path.exists():
            return self.cache_path.read_text(encoding="utf-8", errors="ignore")
        return ""

    def _initialize_dataset_stream(self) -> None:
        try:
            from datasets import load_dataset  # type: ignore
        except ModuleNotFoundError as exc:
            self._last_error = f"Hugging Face datasets package unavailable: {exc}"
            self._source = "remote_unavailable"
            return
        try:
            kwargs = {"split": self.dataset_split, "streaming": True}
            if self.dataset_config:
                dataset = load_dataset(self.dataset_name, self.dataset_config, **kwargs)
            else:
                dataset = load_dataset(self.dataset_name, **kwargs)
            self._dataset_iter = iter(dataset)
            self._source = f"dataset:{self.dataset_name}:{self.dataset_split}"
            self._last_error = None
        except Exception as exc:
            self._dataset_iter = None
            self._source = "remote_failed"
            self._last_error = f"Dataset stream failed: {exc}"

    @staticmethod
    def _split_segments(text: str) -> List[str]:
        return [segment.strip() for segment in _SEGMENT_RE.split(text or "") if segment.strip()]

    @staticmethod
    def _clean_dataset_text(text: str) -> str:
        text = text or ""
        start_match = _START_RE.search(text)
        if start_match:
            text = text[start_match.end():]
        end_match = _END_RE.search(text)
        if end_match:
            text = text[:end_match.start()]
        return _SPACE_RE.sub(" ", text.replace("\x00", " ")).strip()

    def status(self) -> Dict[str, Any]:
        cache_exists = self.cache_path.exists()
        return {
            "loaded": bool(self._segments or self._dataset_iter is not None),
            "segments_loaded": len(self._segments),
            "offset": self._offset,
            "source": self._source,
            "last_error": self._last_error,
            "total_emitted": self._total_emitted,
            "cache_path": str(self.cache_path),
            "cache_exists": cache_exists,
            "cache_size_bytes": self.cache_path.stat().st_size if cache_exists else 0,
            "loaded_at": self._loaded_at,
            "last_batch_at": self._last_batch_at,
            "batch_size": self.batch_size,
            "max_segments": self.max_segments,
            "min_words": self.min_words,
            "dataset": self.dataset_info(),
        }

    def dataset_info(self) -> Dict[str, Any]:
        source = "operator_supplied_language_corpus"
        if self._dataset_iter is not None or (self.dataset_name and not self.disable_remote and not (self.inline_text or self.corpus_path)):
            source = "huggingface_streaming_dataset"
        return {
            "source": source,
            "domain": self.DEFAULT_DOMAIN,
            "corpus_path": self.corpus_path or None,
            "inline_text_configured": bool(self.inline_text),
            "remote_stream_enabled": bool(self.dataset_name and not self.disable_remote),
            "dataset_name": self.dataset_name or None,
            "dataset_config": self.dataset_config or None,
            "dataset_split": self.dataset_split or None,
            "text_field": self.text_field,
            "unit": "text_segment",
            "primary_observable": "token_count",
            "secondary_observable": "unique_ratio",
        }

    def fetch_batch(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        if not self._segments and self._dataset_iter is None:
            self.ensure_loaded()
        size = int(batch_size or self.batch_size)
        observations: List[Dict[str, Any]] = []
        if self._segments:
            for _ in range(max(0, size)):
                segment = self._segments[self._offset % len(self._segments)]
                if len(segment.split()) >= self.min_words:
                    observations.append(self.text_to_observation(segment, source="language_corpus"))
                    self._total_emitted += 1
                self._offset += 1
        elif self._dataset_iter is not None:
            attempts = 0
            max_attempts = max(size * 10, 25)
            while len(observations) < size and attempts < max_attempts:
                attempts += 1
                try:
                    row = next(self._dataset_iter)
                except StopIteration:
                    self._dataset_iter = None
                    self._source = "remote_exhausted"
                    break
                except Exception as exc:
                    self._last_error = f"Dataset read failed: {exc}"
                    break
                text = self._clean_dataset_text(str(row.get(self.text_field) or ""))
                if len(text.split()) < self.min_words:
                    continue
                observation = self.text_to_observation(text, source=f"dataset:{self.dataset_name}")
                observation["dataset"] = self.dataset_name
                observation["dataset_split"] = self.dataset_split
                if row.get("id") is not None:
                    observation["corpus_id"] = str(row.get("id"))
                if row.get("title") is not None:
                    observation["title"] = str(row.get("title"))
                observations.append(observation)
                self._total_emitted += 1
        self._last_batch_at = time.time()
        return {
            "dataset": self.dataset_info(),
            "domain": self.DEFAULT_DOMAIN,
            "count": len(observations),
            "offset": self._offset,
            "observations": observations,
        }

    @staticmethod
    def text_to_observation(text: str, *, source: str = "language") -> Dict[str, Any]:
        tokens = [token for token in _TOKEN_RE.findall(text or "") if token.strip()]
        digest = hashlib.blake2b((text or "").encode("utf-8"), digest_size=12).hexdigest()
        token_count = len(tokens)
        unique_ratio = len(set(token.lower() for token in tokens)) / max(1, token_count)
        punctuation_count = sum(1 for token in tokens if not any(ch.isalnum() for ch in token))
        alpha_count = sum(1 for token in tokens if any(ch.isalpha() for ch in token))
        mean_token_length = sum(len(token) for token in tokens) / max(1, token_count)
        return {
            "entity_id": f"language_{digest}",
            "symbol": f"language_{digest[:8]}",
            "domain_prefix": "language",
            "concept": "language_observation",
            "value": float(token_count),
            "secondary_value": float(unique_ratio),
            "timestamp": time.time(),
            "source": source,
            "text": text,
            "content": text,
            "token_count": token_count,
            "unique_ratio": unique_ratio,
            "punctuation_ratio": punctuation_count / max(1, token_count),
            "alpha_ratio": alpha_count / max(1, token_count),
            "mean_token_length": mean_token_length,
        }
