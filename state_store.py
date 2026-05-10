"""Durable state helpers for the Z³ neural runtime.

Railway containers can restart, so learned in-memory state must be exported to a
mounted volume or local data directory. This module intentionally stays small: it
stores the PyTorch neural checkpoint separately from JSON-serializable world
model and resonant memory state.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_STATE_DIR = os.environ.get(
    "Z3_STATE_DIR",
    os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "/data" if Path("/data").exists() else "data"),
)


class StateStore:
    """Small file-backed persistence layer for Railway or local development."""

    def __init__(self, state_dir: str | os.PathLike[str] = DEFAULT_STATE_DIR) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.neural_path = self.state_dir / "z3_neural_dynamics.pt"
        self.world_model_path = self.state_dir / "world_model.json"
        self.memory_path = self.state_dir / "resonant_memory.json"
        self.manifest_path = self.state_dir / "manifest.json"

    def manifest(self) -> Dict[str, Any]:
        files = {
            "neural_checkpoint": self.neural_path,
            "world_model": self.world_model_path,
            "resonant_memory": self.memory_path,
            "manifest": self.manifest_path,
        }
        return {
            "state_dir": str(self.state_dir),
            "files": {
                name: {
                    "path": str(path),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else 0,
                    "updated_at": path.stat().st_mtime if path.exists() else None,
                }
                for name, path in files.items()
            },
        }

    def save_all(self, *, model: Any, world_model: Any, memory: Any) -> Dict[str, Any]:
        saved_at = time.time()
        model.save_checkpoint(self.neural_path)
        self._write_json(self.world_model_path, world_model.save_state())
        self._write_json(self.memory_path, memory.export_state())
        manifest = {"saved_at": saved_at, **self.manifest()}
        self._write_json(self.manifest_path, manifest)
        return manifest

    def load_all(self, *, model: Optional[Any], model_cls: Any, world_model: Any, memory: Any) -> Dict[str, Any]:
        loaded: Dict[str, Any] = {"loaded_at": time.time(), "loaded": {}}
        if self.neural_path.exists():
            loaded_model = model_cls.load_checkpoint(self.neural_path, map_location="cpu")
            model = loaded_model
            loaded["loaded"]["neural_checkpoint"] = True
        else:
            loaded["loaded"]["neural_checkpoint"] = False

        world_state = self._read_json(self.world_model_path)
        if world_state:
            world_model.load_state(world_state)
            loaded["loaded"]["world_model"] = True
        else:
            loaded["loaded"]["world_model"] = False

        memory_state = self._read_json(self.memory_path)
        if memory_state:
            memory.load_state(memory_state)
            loaded["loaded"]["resonant_memory"] = True
        else:
            loaded["loaded"]["resonant_memory"] = False

        loaded["manifest"] = self.manifest()
        loaded["model"] = model
        return loaded

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(path)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
