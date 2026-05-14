"""Language stream for Z³ training observations.

This module provides a standalone, synchronous language source for the Z³ neural
network runtime. It intentionally uses real text supplied by the operator through
``LANGUAGE_TRAINING_CORPUS_PATH`` or ``LANGUAGE_TRAINING_TEXT``. If no corpus is
configured, it remains safely idle while the `/chat` and `/observe` endpoints can
still provide live language observations.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
_SEGMENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


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
        self._segments: List[str] = []
        self._offset = 0
        self._loaded_at: Optional[float] = None
        self._last_batch_at: Optional[float] = None

    def ensure_loaded(self) -> Dict[str, Any]:
        """Load configured real language text from path, inline env, or cache."""
        text = self._load_text()
        self._segments = self._split_segments(text)[: self.max_segments]
        self._loaded_at = time.time()
        return self.status()

    def _load_text(self) -> str:
        if self.corpus_path:
            path = Path(self.corpus_path).expanduser()
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8", errors="ignore")
        if self.inline_text:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(self.inline_text, encoding="utf-8")
            return self.inline_text
        if self.cache_path.exists():
            return self.cache_path.read_text(encoding="utf-8", errors="ignore")
        return ""

    @staticmethod
    def _split_segments(text: str) -> List[str]:
        return [segment.strip() for segment in _SEGMENT_RE.split(text or "") if segment.strip()]

    def status(self) -> Dict[str, Any]:
        cache_exists = self.cache_path.exists()
        return {
            "loaded": bool(self._segments),
            "segments_loaded": len(self._segments),
            "offset": self._offset,
            "cache_path": str(self.cache_path),
            "cache_exists": cache_exists,
            "cache_size_bytes": self.cache_path.stat().st_size if cache_exists else 0,
            "loaded_at": self._loaded_at,
            "last_batch_at": self._last_batch_at,
            "batch_size": self.batch_size,
            "max_segments": self.max_segments,
            "dataset": self.dataset_info(),
        }

    def dataset_info(self) -> Dict[str, Any]:
        return {
            "source": "operator_supplied_language_corpus",
            "domain": self.DEFAULT_DOMAIN,
            "corpus_path": self.corpus_path or None,
            "inline_text_configured": bool(self.inline_text),
            "unit": "text_segment",
            "primary_observable": "token_count",
            "secondary_observable": "unique_ratio",
        }

    def fetch_batch(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        if not self._segments:
            self.ensure_loaded()
        size = int(batch_size or self.batch_size)
        observations: List[Dict[str, Any]] = []
        if self._segments:
            for _ in range(max(0, size)):
                segment = self._segments[self._offset % len(self._segments)]
                observations.append(self.text_to_observation(segment, source="language_corpus"))
                self._offset += 1
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
