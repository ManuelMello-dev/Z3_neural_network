"""Autonomous always-on runtime loop for Z³.

The loop is deliberately small and dependency-light. It supervises periodic ticks,
tracks heartbeat/error history, and delegates actual learning/persistence work to
callbacks owned by the FastAPI runtime membrane.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Optional


TickCallback = Callable[[], Dict[str, Any]]
SaveCallback = Callable[[], Dict[str, Any]]


@dataclass
class RuntimeLoopConfig:
    """Configuration for autonomous runtime execution."""

    interval_seconds: float = 30.0
    autosave_every_ticks: int = 5
    max_history: int = 64


@dataclass
class RuntimeLoopState:
    """Runtime loop state exposed through the API and dashboard."""

    running: bool = False
    tick_count: int = 0
    error_count: int = 0
    last_started_at: Optional[float] = None
    last_stopped_at: Optional[float] = None
    last_tick_at: Optional[float] = None
    last_autosave_at: Optional[float] = None
    last_error: Optional[str] = None
    last_tick: Dict[str, Any] = field(default_factory=dict)


class AutonomousRuntimeLoop:
    """Thread-backed periodic tick loop for always-on learning."""

    def __init__(
        self,
        *,
        tick_callback: TickCallback,
        save_callback: Optional[SaveCallback] = None,
        config: Optional[RuntimeLoopConfig] = None,
    ) -> None:
        self.tick_callback = tick_callback
        self.save_callback = save_callback
        self.config = config or RuntimeLoopConfig()
        self.state = RuntimeLoopState()
        self.history: Deque[Dict[str, Any]] = deque(maxlen=self.config.max_history)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self, *, interval_seconds: Optional[float] = None, autosave_every_ticks: Optional[int] = None) -> Dict[str, Any]:
        """Start the background loop if it is not already running."""
        with self._lock:
            if interval_seconds is not None:
                self.config.interval_seconds = max(1.0, float(interval_seconds))
            if autosave_every_ticks is not None:
                self.config.autosave_every_ticks = max(1, int(autosave_every_ticks))
            if self.state.running and self._thread and self._thread.is_alive():
                return self.status()
            self._stop_event.clear()
            self.state.running = True
            self.state.last_started_at = time.time()
            self._thread = threading.Thread(target=self._run, name="z3-autonomous-runtime", daemon=True)
            self._thread.start()
            return self.status()

    def stop(self) -> Dict[str, Any]:
        """Stop the background loop and wait briefly for the worker to exit."""
        thread = self._thread
        with self._lock:
            self._stop_event.set()
            self.state.running = False
            self.state.last_stopped_at = time.time()
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, min(float(self.config.interval_seconds), 10.0)))
        return self.status()

    def tick_once(self) -> Dict[str, Any]:
        """Run one autonomous tick synchronously."""
        started = time.time()
        try:
            result = self.tick_callback()
            with self._lock:
                self.state.tick_count += 1
                self.state.last_tick_at = time.time()
                self.state.last_tick = result
                record = {
                    "tick": self.state.tick_count,
                    "started_at": started,
                    "finished_at": self.state.last_tick_at,
                    "duration_seconds": round(self.state.last_tick_at - started, 6),
                    "result": result,
                }
                self.history.append(record)
                should_save = self.save_callback is not None and self.state.tick_count % self.config.autosave_every_ticks == 0
            if should_save:
                manifest = self.save_callback() if self.save_callback else {}
                with self._lock:
                    self.state.last_autosave_at = time.time()
                    self.state.last_tick = {**self.state.last_tick, "autosave": manifest}
                    if self.history:
                        self.history[-1]["autosave"] = manifest
            return self.status()
        except Exception as exc:  # pragma: no cover - defensive runtime membrane.
            with self._lock:
                self.state.error_count += 1
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                self.state.last_tick_at = time.time()
                self.history.append(
                    {
                        "tick": self.state.tick_count,
                        "started_at": started,
                        "finished_at": self.state.last_tick_at,
                        "error": self.state.last_error,
                    }
                )
            return self.status()

    def status(self) -> Dict[str, Any]:
        """Return loop status and recent tick history."""
        thread_alive = bool(self._thread and self._thread.is_alive())
        return {
            "running": bool(self.state.running and thread_alive),
            "thread_alive": thread_alive,
            "config": {
                "interval_seconds": self.config.interval_seconds,
                "autosave_every_ticks": self.config.autosave_every_ticks,
                "max_history": self.config.max_history,
            },
            "tick_count": self.state.tick_count,
            "error_count": self.state.error_count,
            "last_started_at": self.state.last_started_at,
            "last_stopped_at": self.state.last_stopped_at,
            "last_tick_at": self.state.last_tick_at,
            "last_autosave_at": self.state.last_autosave_at,
            "last_error": self.state.last_error,
            "last_tick": self.state.last_tick,
            "recent_history": list(self.history)[-10:],
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.tick_once()
            self._stop_event.wait(self.config.interval_seconds)
        with self._lock:
            self.state.running = False
            self.state.last_stopped_at = time.time()
