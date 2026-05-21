"""Curriculum stream for Z³ multi-domain training observations.

This module turns a layered curriculum into the same observation schema consumed by
``/observe``. It is intentionally synchronous and Railway-safe: operator-supplied
JSONL/local data is preferred, while Hugging Face streaming datasets can be enabled
per curriculum kind without making remote access a hard dependency at runtime.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from language_stream import LanguageStream


class CurriculumStream:
    """Synchronous curriculum source for Z³ observations.

    The stream supports six practical curriculum kinds:
    language, dialogue, contradiction, event, anomaly, and identity. Each row is
    converted into a structured observation with common fields so the world
    model, resonant memory, and Z³ core can all consume it through the existing
    integrated observe path.
    """

    DEFAULT_DOMAIN_PREFIX = "curriculum"
    DEFAULT_KINDS = ("language", "dialogue", "contradiction", "event", "anomaly", "identity")
    DEFAULT_DATASETS: Dict[str, Dict[str, str]] = {
        "language": {"dataset": "manu/project_gutenberg", "split": "en", "text_field": "text"},
        "dialogue": {"dataset": "daily_dialog", "split": "train", "text_field": "dialog"},
        "contradiction": {"dataset": "snli", "split": "train", "text_field": "premise"},
        "event": {"dataset": "zeroshot/twitter-financial-news-topic", "split": "train", "text_field": "text"},
        "anomaly": {"dataset": "mstz/real-life-industrial-dataset-of-casting-product", "split": "train", "text_field": "label"},
        "identity": {"dataset": "manu/project_gutenberg", "split": "en", "text_field": "text"},
    }

    def __init__(self) -> None:
        state_dir = Path(os.environ.get("Z3_STATE_DIR", os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "data")))
        self.curriculum_path = os.getenv("Z3_CURRICULUM_PATH", "").strip()
        self.inline_jsonl = os.getenv("Z3_CURRICULUM_INLINE_JSONL", "").strip()
        self.cache_path = Path(os.getenv("Z3_CURRICULUM_CACHE", str(state_dir / "curriculum_observations.jsonl")))
        self.batch_size = int(os.getenv("Z3_CURRICULUM_BATCH_SIZE", "18"))
        self.max_local_rows = int(os.getenv("Z3_CURRICULUM_MAX_LOCAL_ROWS", "100000"))
        self.sources = self._parse_sources(os.getenv("Z3_CURRICULUM_SOURCES", ",".join(self.DEFAULT_KINDS)))
        self.remote_enabled = os.getenv("Z3_CURRICULUM_REMOTE_ENABLED", "false").lower() in ("1", "true", "yes", "on")
        self.min_text_words = int(os.getenv("Z3_CURRICULUM_MIN_TEXT_WORDS", "4"))

        self._rows: List[Dict[str, Any]] = []
        self._offset = 0
        self._loaded_at: Optional[float] = None
        self._last_batch_at: Optional[float] = None
        self._dataset_iters: Dict[str, Iterator[Dict[str, Any]]] = {}
        self._source = "idle"
        self._last_error: Optional[str] = None
        self._total_emitted = 0

    def ensure_loaded(self) -> Dict[str, Any]:
        """Load local/inline curriculum data and optionally initialize remote streams."""
        rows = self._load_operator_rows()
        if rows:
            self._rows = rows[: self.max_local_rows]
            self._source = "inline_jsonl" if self.inline_jsonl else ("local_file" if self.curriculum_path else "cache")
        elif self.remote_enabled:
            for kind in self.sources:
                self._initialize_dataset_stream(kind)
            if self._dataset_iters:
                self._source = "huggingface_streaming_curriculum"
        self._loaded_at = time.time()
        return self.status()

    def _parse_sources(self, raw: str) -> List[str]:
        values = [item.strip().lower() for item in (raw or "").split(",") if item.strip()]
        valid = [item for item in values if item in self.DEFAULT_KINDS]
        return valid or list(self.DEFAULT_KINDS)

    def _load_operator_rows(self) -> List[Dict[str, Any]]:
        text = ""
        if self.curriculum_path:
            path = Path(self.curriculum_path).expanduser()
            if path.exists() and path.is_file():
                if path.suffix.lower() == ".csv":
                    return self._read_csv(path)
                text = path.read_text(encoding="utf-8", errors="ignore")
            else:
                self._last_error = f"Curriculum path not found: {path}"
        elif self.inline_jsonl:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(self.inline_jsonl, encoding="utf-8")
            text = self.inline_jsonl
        elif self.cache_path.exists():
            text = self.cache_path.read_text(encoding="utf-8", errors="ignore")
        return self._read_jsonl_text(text)

    def _read_jsonl_text(self, text: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for line_no, line in enumerate((text or "").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = {"kind": "language", "text": line, "source": "plain_text_line"}
            if isinstance(value, dict):
                value.setdefault("source", "operator_curriculum")
                value.setdefault("row_index", line_no)
                rows.append(value)
        return rows

    def _read_csv(self, path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader, start=1):
                cleaned = {str(k): v for k, v in row.items() if k is not None}
                cleaned.setdefault("kind", "event")
                cleaned.setdefault("source", f"csv:{path.name}")
                cleaned.setdefault("row_index", idx)
                rows.append(cleaned)
                if len(rows) >= self.max_local_rows:
                    break
        return rows

    def _initialize_dataset_stream(self, kind: str) -> None:
        try:
            from datasets import load_dataset  # type: ignore
        except ModuleNotFoundError as exc:
            self._last_error = f"Hugging Face datasets package unavailable: {exc}"
            return
        spec = self._dataset_spec(kind)
        try:
            kwargs = {"split": spec["split"], "streaming": True}
            config = spec.get("config") or ""
            dataset = load_dataset(spec["dataset"], config, **kwargs) if config else load_dataset(spec["dataset"], **kwargs)
            self._dataset_iters[kind] = iter(dataset)
            self._last_error = None
        except Exception as exc:
            self._last_error = f"Curriculum dataset stream failed for {kind}: {exc}"

    def _dataset_spec(self, kind: str) -> Dict[str, str]:
        base = dict(self.DEFAULT_DATASETS.get(kind, self.DEFAULT_DATASETS["language"]))
        prefix = f"Z3_CURRICULUM_{kind.upper()}"
        base["dataset"] = os.getenv(f"{prefix}_DATASET", base["dataset"]).strip()
        base["config"] = os.getenv(f"{prefix}_CONFIG", base.get("config", "")).strip()
        base["split"] = os.getenv(f"{prefix}_SPLIT", base["split"]).strip()
        base["text_field"] = os.getenv(f"{prefix}_TEXT_FIELD", base.get("text_field", "text")).strip()
        return base

    def status(self) -> Dict[str, Any]:
        cache_exists = self.cache_path.exists()
        return {
            "loaded": bool(self._rows or self._dataset_iters),
            "source": self._source,
            "sources": list(self.sources),
            "remote_enabled": self.remote_enabled,
            "rows_loaded": len(self._rows),
            "remote_streams": sorted(self._dataset_iters.keys()),
            "offset": self._offset,
            "batch_size": self.batch_size,
            "total_emitted": self._total_emitted,
            "last_error": self._last_error,
            "cache_path": str(self.cache_path),
            "cache_exists": cache_exists,
            "cache_size_bytes": self.cache_path.stat().st_size if cache_exists else 0,
            "loaded_at": self._loaded_at,
            "last_batch_at": self._last_batch_at,
            "datasets": {kind: self._dataset_spec(kind) for kind in self.sources},
        }

    def fetch_batch(self, batch_size: Optional[int] = None, *, source: Optional[str] = None) -> Dict[str, Any]:
        if not self._rows and not self._dataset_iters:
            self.ensure_loaded()
        size = int(batch_size or self.batch_size)
        requested_sources = [source.lower()] if source and source.lower() in self.DEFAULT_KINDS else list(self.sources)
        observations: List[Dict[str, Any]] = []

        if self._rows:
            attempts = 0
            max_attempts = max(size * 8, len(self._rows))
            while len(observations) < size and attempts < max_attempts:
                attempts += 1
                row = self._rows[self._offset % len(self._rows)]
                self._offset += 1
                kind = str(row.get("kind") or row.get("curriculum_kind") or "language").lower()
                if kind not in requested_sources:
                    continue
                observations.append(self.row_to_observation(row, fallback_kind=kind))
                self._total_emitted += 1
        else:
            cursor = 0
            attempts = 0
            max_attempts = max(size * 10, 30)
            while len(observations) < size and attempts < max_attempts and requested_sources:
                attempts += 1
                kind = requested_sources[cursor % len(requested_sources)]
                cursor += 1
                iterator = self._dataset_iters.get(kind)
                if iterator is None:
                    continue
                try:
                    row = next(iterator)
                except StopIteration:
                    self._dataset_iters.pop(kind, None)
                    continue
                except Exception as exc:
                    self._last_error = f"Curriculum dataset read failed for {kind}: {exc}"
                    continue
                observations.append(self.row_to_observation(row, fallback_kind=kind, remote=True))
                self._total_emitted += 1

        self._last_batch_at = time.time()
        return {
            "dataset": self.status(),
            "domain": f"{self.DEFAULT_DOMAIN_PREFIX}:mixed" if not source else f"{self.DEFAULT_DOMAIN_PREFIX}:{source}",
            "count": len(observations),
            "offset": self._offset,
            "observations": observations,
        }

    def row_to_observation(self, row: Dict[str, Any], *, fallback_kind: str = "language", remote: bool = False) -> Dict[str, Any]:
        kind = str(row.get("kind") or row.get("curriculum_kind") or fallback_kind or "language").strip().lower()
        if kind not in self.DEFAULT_KINDS:
            kind = "language"
        text = self._extract_text(row, kind)
        numeric_values = self._extract_numeric_values(row)
        digest_payload = json.dumps(row, sort_keys=True, default=str)[:4096]
        digest = hashlib.blake2b(digest_payload.encode("utf-8"), digest_size=12).hexdigest()
        token_count = len(text.split()) if text else 0
        value = self._primary_value(kind, row, numeric_values, token_count)
        secondary = self._secondary_value(kind, row, numeric_values, text)
        observation = {
            "entity_id": str(row.get("entity_id") or row.get("id") or f"{kind}_{digest}"),
            "symbol": str(row.get("symbol") or f"{kind}_{digest[:8]}"),
            "domain_prefix": self.DEFAULT_DOMAIN_PREFIX,
            "concept": str(row.get("concept") or f"{kind}_curriculum_observation"),
            "pattern_id": str(row.get("pattern_id") or kind),
            "value": float(value),
            "secondary_value": float(secondary),
            "timestamp": self._timestamp(row),
            "source": str(row.get("source") or ("huggingface_curriculum" if remote else "operator_curriculum")),
            "curriculum_kind": kind,
            "curriculum_stage": self._stage_for_kind(kind),
            "text": text,
            "content": text,
            "token_count": token_count,
            "numeric_feature_count": len(numeric_values),
            "novelty_target": self._novelty_target(kind, row),
            "coherence_target": self._coherence_target(kind, row),
            "contradiction_target": self._contradiction_target(kind, row),
            "salience_hint": self._salience_hint(kind, row),
        }
        for key in ("label", "premise", "hypothesis", "question", "answer", "title", "speaker", "domain"):
            if row.get(key) is not None:
                observation[key] = row.get(key)
        return observation

    def _extract_text(self, row: Dict[str, Any], kind: str) -> str:
        for key in ("text", "content", "message", "utterance", "dialogue", "premise", "hypothesis", "question", "answer", "title"):
            value = row.get(key)
            if isinstance(value, list):
                value = " ".join(str(item) for item in value)
            if value is not None and str(value).strip():
                break
        else:
            value = " ".join(str(v) for v in row.values() if isinstance(v, (str, int, float)) and str(v).strip())
        if kind == "contradiction":
            premise = str(row.get("premise") or "").strip()
            hypothesis = str(row.get("hypothesis") or "").strip()
            if premise or hypothesis:
                return f"Premise: {premise} Hypothesis: {hypothesis}".strip()
        return str(value or "").strip()

    def _extract_numeric_values(self, row: Dict[str, Any]) -> List[float]:
        values: List[float] = []
        for key, value in row.items():
            if key in {"timestamp", "row_index"}:
                continue
            if isinstance(value, bool):
                values.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
            elif isinstance(value, str):
                try:
                    numeric = float(value)
                except ValueError:
                    continue
                if math.isfinite(numeric):
                    values.append(numeric)
        return values[:32]

    def _primary_value(self, kind: str, row: Dict[str, Any], values: List[float], token_count: int) -> float:
        if row.get("value") is not None:
            return self._safe_float(row.get("value"), 0.0)
        if kind in {"event", "anomaly"} and values:
            return sum(values) / max(1, len(values))
        return float(token_count)

    def _secondary_value(self, kind: str, row: Dict[str, Any], values: List[float], text: str) -> float:
        if row.get("secondary_value") is not None:
            return self._safe_float(row.get("secondary_value"), 0.0)
        if kind == "contradiction":
            return self._contradiction_target(kind, row)
        if values:
            mean = sum(values) / max(1, len(values))
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
            return math.sqrt(variance)
        words = [word.lower() for word in text.split() if word.strip()]
        return len(set(words)) / max(1, len(words))

    def _timestamp(self, row: Dict[str, Any]) -> float:
        return self._safe_float(row.get("timestamp"), time.time())

    def _stage_for_kind(self, kind: str) -> int:
        return {"language": 1, "dialogue": 2, "event": 3, "anomaly": 4, "contradiction": 5, "identity": 6}.get(kind, 1)

    def _novelty_target(self, kind: str, row: Dict[str, Any]) -> float:
        if row.get("novelty_target") is not None:
            return self._clamp(self._safe_float(row.get("novelty_target"), 0.5))
        return {"language": 0.35, "dialogue": 0.45, "event": 0.55, "anomaly": 0.90, "contradiction": 0.75, "identity": 0.60}.get(kind, 0.5)

    def _coherence_target(self, kind: str, row: Dict[str, Any]) -> float:
        if row.get("coherence_target") is not None:
            return self._clamp(self._safe_float(row.get("coherence_target"), 0.5))
        return {"language": 0.70, "dialogue": 0.72, "event": 0.62, "anomaly": 0.50, "contradiction": 0.58, "identity": 0.82}.get(kind, 0.65)

    def _contradiction_target(self, kind: str, row: Dict[str, Any]) -> float:
        if row.get("contradiction_target") is not None:
            return self._clamp(self._safe_float(row.get("contradiction_target"), 0.0))
        label = str(row.get("label") or row.get("gold_label") or "").strip().lower()
        if kind == "contradiction" or label in {"contradiction", "contradictory", "0"}:
            return 1.0 if label in {"contradiction", "contradictory", "0", ""} else 0.5
        return 0.0

    def _salience_hint(self, kind: str, row: Dict[str, Any]) -> float:
        if row.get("salience") is not None:
            return self._clamp(self._safe_float(row.get("salience"), 0.5))
        return {"language": 0.45, "dialogue": 0.62, "event": 0.58, "anomaly": 0.85, "contradiction": 0.76, "identity": 0.90}.get(kind, 0.5)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            numeric = float(value)
            return numeric if math.isfinite(numeric) else default
        except Exception:
            return default

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))


__all__ = ["CurriculumStream"]
