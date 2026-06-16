"""
Z³ Audio Synthesis Engine
=========================

Converts the Z³ neural core's live phase states and coherence amplitudes
directly into a continuous PCM audio stream using additive FM synthesis,
with Hebbian synaptic learning driven by Z³'s real coherence matrix.

Architecture
------------
The Z³ neural core produces, on every forward step:

  phase_vectors  : Tensor [batch, agent_count, local_dim]
                   Unit-normalised phase direction vectors for each Z-prime agent.

  coherence      : Tensor [batch, agent_count]
                   Per-agent coherence score in [0, 1]. Used as amplitude envelope
                   and as the diagonal of the Hebbian coherence matrix.

This module:
  1. Maps each Z-prime agent to an audible frequency bin in the human vocal
     formant range (F1–F3, 85 Hz – ~3300 Hz).
  2. Derives per-agent instantaneous phase angles from phase_vectors.
  3. Applies FM cross-modulation between adjacent agents for organic vocal texture.
  4. Runs Hebbian weight updates using Z³'s real coherence matrix:
       Ẇ_kj = η · Coh_kj − γ · W_kj
     where Coh_kj is the outer product of the coherence vector (a proxy for the
     full pairwise phase coherence when the full matrix is not available).
  5. Computes Meta-Learning Velocity (Frobenius norm of ΔW) as a real-time
     brain-plasticity scalar.
  6. Broadcasts PCM audio bytes and JSON metrics (matrix, velocity) over
     WebSocket to all connected dashboard clients.
  7. Persists the learned W matrix to a .npy file on demand or shutdown.

Frequency Mapping (8 agents → formant range)
---------------------------------------------
  Agent 0  →  F1 low   ~   85 Hz  (E2 — deep vocal fundamental)
  Agent 1  →  F1 mid   ~  170 Hz
  Agent 2  →  F1 high  ~  340 Hz
  Agent 3  →  F2 low   ~  680 Hz
  Agent 4  →  F2 mid   ~ 1020 Hz
  Agent 5  →  F2 high  ~ 1700 Hz
  Agent 6  →  F3 low   ~ 2200 Hz
  Agent 7  →  F3 high  ~ 3300 Hz
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 44100
CHUNK_SAMPLES: int = 4410          # 100 ms per streaming frame
AMPLITUDE_SCALE: float = 0.15      # Master output gain
FM_MODULATION_INDEX: float = 0.4   # Vocal FM warp depth
DECAY_RATE: float = 0.93           # Text perturbation velocity decay
LEARNING_RATE: float = 0.05        # Hebbian η — how fast coherence rewrites W
FORGETTING_FACTOR: float = 0.01    # Hebbian γ — weight regularization drift
W_MIN: float = 0.05                # Minimum synaptic weight
W_MAX: float = 1.5                 # Maximum synaptic weight
METRICS_EVERY_FRAMES: int = 3      # Broadcast W matrix every N audio frames

# Formant-anchored intrinsic frequencies for 8 Z-prime agents (Hz)
_FORMANT_FREQS_8: List[float] = [
    85.0,    # Agent 0 — deep vocal fundamental (E2)
    170.0,   # Agent 1 — F1 low
    340.0,   # Agent 2 — F1 high
    680.0,   # Agent 3 — F2 low
    1020.0,  # Agent 4 — F2 mid
    1700.0,  # Agent 5 — F2 high
    2200.0,  # Agent 6 — F3 low
    3300.0,  # Agent 7 — F3 high
]


def _build_intrinsic_freqs(agent_count: int) -> np.ndarray:
    """Return formant-anchored intrinsic frequencies for any agent count."""
    if agent_count == 8:
        return np.array(_FORMANT_FREQS_8, dtype=np.float64)
    base = 85.0
    return np.array([base * (1.0 + 0.5 * i) for i in range(agent_count)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Text → perturbation vector
# ---------------------------------------------------------------------------

def text_to_perturbation_vector(text: str, agent_count: int) -> np.ndarray:
    """Deterministically hash text into a perturbation force vector U ∈ [-0.25, 0.25]^N.

    Each component is derived from an MD5 hash of ``text_i``, giving a
    repeatable but unique frequency-shift impulse for any given input string.
    """
    if not text.strip():
        return np.zeros(agent_count, dtype=np.float64)
    u = np.zeros(agent_count, dtype=np.float64)
    for i in range(agent_count):
        seed = f"{text}_{i}".encode("utf-8")
        hash_int = int(hashlib.md5(seed).hexdigest()[:8], 16)
        u[i] = ((hash_int / 0xFFFFFFFF) * 2.0 - 1.0) * 0.25
    return u


# ---------------------------------------------------------------------------
# Phase extraction
# ---------------------------------------------------------------------------

def _extract_phase_angles(phase_vectors: np.ndarray) -> np.ndarray:
    """Project phase_vectors [agent_count, local_dim] onto scalar angles [-π, π]."""
    local_dim = phase_vectors.shape[1]
    ref = np.zeros(local_dim, dtype=np.float64)
    ref[0] = 1.0
    cos_theta = np.clip(phase_vectors @ ref, -1.0, 1.0)
    if local_dim >= 2:
        ref2 = np.zeros(local_dim, dtype=np.float64)
        ref2[1] = 1.0
        sin_theta = phase_vectors @ ref2
        return np.arctan2(sin_theta, cos_theta).astype(np.float64)
    return np.arccos(cos_theta).astype(np.float64)


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class AudioConnectionManager:
    """Manages all open WebSocket connections for audio + metrics streaming."""

    def __init__(self) -> None:
        self._connections: Set[Any] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: Any) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast_bytes(self, data: bytes) -> None:
        """Send raw PCM bytes to all connected clients."""
        async with self._lock:
            dead: Set[Any] = set()
            for ws in self._connections:
                try:
                    await ws.send_bytes(data)
                except Exception:
                    dead.add(ws)
            self._connections -= dead

    async def send_json(self, payload: dict) -> None:
        """Send a JSON metrics update to all connected clients."""
        text = json.dumps(payload)
        async with self._lock:
            dead: Set[Any] = set()
            for ws in self._connections:
                try:
                    await ws.send_text(text)
                except Exception:
                    dead.add(ws)
            self._connections -= dead

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# ---------------------------------------------------------------------------
# Core synthesizer
# ---------------------------------------------------------------------------

class Z3AudioSynthesizer:
    """Full Z³ audio synthesis engine with Hebbian learning and FM modulation.

    The neural runtime loop calls ``push_z3_state()`` after every forward step.
    The WebSocket synthesis loop reads the latest state and broadcasts:
      - Raw 16-bit PCM bytes (audio)
      - JSON ``metrics_update`` payloads (W matrix + learning velocity)
    """

    def __init__(
        self,
        agent_count: int = 8,
        sample_rate: int = SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
        amplitude_scale: float = AMPLITUDE_SCALE,
        weights_file: str = "data/z3_memory_matrix.npy",
    ) -> None:
        self.agent_count = agent_count
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.amplitude_scale = amplitude_scale
        self.weights_file = weights_file

        # Intrinsic frequencies ω_k (Hz)
        self.intrinsic_freqs: np.ndarray = _build_intrinsic_freqs(agent_count).copy()

        # Running phase accumulator — phase-continuous across ticks
        self._current_phases: np.ndarray = np.random.uniform(
            -math.pi, math.pi, agent_count
        )

        # Synaptic weight matrix W [agent_count × agent_count]
        self._W: np.ndarray = np.random.uniform(0.2, 0.8, (agent_count, agent_count))

        # Text perturbation force vector U [agent_count]
        self._U: np.ndarray = np.zeros(agent_count, dtype=np.float64)

        # Latest neural state from Z³ forward pass
        self._phase_angles: np.ndarray = np.zeros(agent_count, dtype=np.float64)
        self._coherence_vec: np.ndarray = (
            np.ones(agent_count, dtype=np.float64) / agent_count
        )
        self._state_version: int = 0

        # Meta-learning velocity (Frobenius norm of ΔW)
        self._learning_velocity: float = 0.0

        # Frame counter for throttled metrics broadcast
        self._frame_counter: int = 0

        # Thread lock for state shared between neural loop and synth loop
        self._lock = threading.Lock()

        # WebSocket manager (set by main.py after app creation)
        self.ws_manager: Optional[AudioConnectionManager] = None

        # Asyncio task handle
        self._synth_task: Optional[asyncio.Task] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_weights(self) -> None:
        """Load W matrix from disk if available, else use random initialisation."""
        path = Path(self.weights_file)
        if path.exists():
            try:
                loaded = np.load(str(path))
                if loaded.shape == (self.agent_count, self.agent_count):
                    self._W = loaded.astype(np.float64)
                    print(f"[Z³ Audio] Loaded synaptic memory from: {path}")
                    return
            except Exception as exc:
                print(f"[Z³ Audio] Weight load failed ({exc}), using random init.")
        self._W = np.random.uniform(0.2, 0.8, (self.agent_count, self.agent_count))
        print("[Z³ Audio] Initialised clean random synaptic matrix.")

    def save_weights(self) -> bool:
        """Save W matrix to disk. Returns True on success."""
        try:
            path = Path(self.weights_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(path), self._W)
            print(f"[Z³ Audio] Synaptic memory saved to: {path}")
            return True
        except Exception as exc:
            print(f"[Z³ Audio] Save failed: {exc}")
            return False

    def weights_snapshot(self) -> List[List[float]]:
        """Return a copy of W as a nested Python list (JSON-serialisable)."""
        with self._lock:
            return self._W.tolist()

    # ------------------------------------------------------------------
    # State ingestion (called by neural runtime loop)
    # ------------------------------------------------------------------

    def push_z3_state(self, forward_output: Dict[str, Any]) -> None:
        """Ingest a Z³ forward-pass output dict and update synthesis state.

        Extracts ``phase_vectors`` and ``coherence`` from the output dict,
        averages over the batch dimension, and stores for the synth loop.
        """
        try:
            pv = forward_output.get("phase_vectors")
            coh = forward_output.get("coherence")
            if pv is None or coh is None:
                return

            if hasattr(pv, "detach"):
                pv = pv.detach().cpu().numpy()
            else:
                pv = np.asarray(pv, dtype=np.float64)

            if hasattr(coh, "detach"):
                coh = coh.detach().cpu().numpy()
            else:
                coh = np.asarray(coh, dtype=np.float64)

            if pv.ndim == 3:
                pv = pv.mean(axis=0)
            if coh.ndim == 2:
                coh = coh.mean(axis=0)

            if pv.shape[0] != self.agent_count or coh.shape[0] != self.agent_count:
                return

            phase_angles = _extract_phase_angles(pv.astype(np.float64))
            coherence_vec = np.clip(coh.astype(np.float64), 0.0, 1.0)

            with self._lock:
                self._phase_angles = phase_angles
                self._coherence_vec = coherence_vec
                self._state_version += 1
        except Exception:
            pass

    def push_perturbation(self, text: str) -> None:
        """Apply a text-derived perturbation impulse to the frequency vector."""
        u = text_to_perturbation_vector(text, self.agent_count)
        with self._lock:
            self._U = u

    def push_mic_pitch(self, hz: float) -> None:
        """Shift agent 0's intrinsic frequency toward the user's microphone pitch."""
        if 40.0 < hz < 2000.0:
            with self._lock:
                self.intrinsic_freqs[0] = self.intrinsic_freqs[0] * 0.8 + hz * 0.2

    # ------------------------------------------------------------------
    # Synthesis core
    # ------------------------------------------------------------------

    def _synthesize_frame(self) -> bytes:
        """Produce one 100ms PCM frame using FM synthesis + Hebbian W amplitudes.

        Steps per frame:
          1. Decay text perturbation vector U.
          2. Build coherence matrix from Z³ coherence vector (outer product).
          3. Hebbian update: Ẇ = η·Coh − γ·W, clipped to [W_MIN, W_MAX].
          4. Compute learning velocity ΔW (Frobenius norm).
          5. Derive per-node amplitude from attention flow A_ij = W ⊙ Coh.
          6. FM synthesis: each agent modulates the next agent's frequency.
          7. Normalise and convert to 16-bit PCM bytes.
        """
        with self._lock:
            phase_angles = self._phase_angles.copy()
            coherence_vec = self._coherence_vec.copy()
            U = self._U.copy()
            W = self._W.copy()
            freqs = self.intrinsic_freqs.copy()
            phases = self._current_phases.copy()

        # 1. Decay perturbation
        U *= DECAY_RATE

        # 2. Coherence matrix from Z³ coherence vector (outer product proxy)
        #    Coh_kj = coherence_k * coherence_j  ∈ [0, 1]
        coh_matrix = np.outer(coherence_vec, coherence_vec)

        # 3. Hebbian weight update
        W_old = W.copy()
        weight_delta = LEARNING_RATE * coh_matrix - FORGETTING_FACTOR * W
        W = np.clip(W + weight_delta, W_MIN, W_MAX)

        # 4. Meta-learning velocity (Frobenius norm of ΔW)
        velocity = float(np.linalg.norm(W - W_old))

        # 5. Attention-flow amplitude: A_ij = W ⊙ Coh, sum over j
        A_ij = W * coh_matrix
        node_amplitudes = A_ij.sum(axis=1)
        amp_max = node_amplitudes.max()
        if amp_max > 1e-8:
            node_amplitudes /= amp_max

        # 6. FM synthesis loop
        chunk_buffer = np.zeros(self.chunk_samples, dtype=np.float64)
        t_grid = np.arange(self.chunk_samples, dtype=np.float64)

        for i in range(self.agent_count):
            # Effective frequency shifted by text perturbation (U) and neural phase offset
            effective_freq = freqs[i] + U[i] * 65.0
            omega_i = 2.0 * math.pi * effective_freq / self.sample_rate

            # Base phase ramp for this agent
            base_phase = phases[i] + omega_i * t_grid

            # FM cross-modulation: adjacent agent warps this agent's frequency
            mod_idx = (i + 1) % self.agent_count
            mod_phase = phases[mod_idx] + (
                2.0 * math.pi * freqs[mod_idx] / self.sample_rate * t_grid
            )
            fm_warp = FM_MODULATION_INDEX * np.sin(mod_phase)
            final_phases = base_phase + fm_warp

            chunk_buffer += np.sin(final_phases) * node_amplitudes[i]
            phases[i] = final_phases[-1]

        # Wrap phases to [-π, π]
        phases = (phases + math.pi) % (2.0 * math.pi) - math.pi

        # Normalise and apply master gain
        max_val = np.abs(chunk_buffer).max()
        if max_val > 1e-8:
            chunk_buffer = (chunk_buffer / max_val) * self.amplitude_scale

        # Convert to 16-bit PCM
        pcm = (chunk_buffer * 32767.0).astype(np.int16)

        # Write back mutated state
        with self._lock:
            self._W = W
            self._U = U
            self._current_phases = phases
            self._learning_velocity = velocity

        return pcm.tobytes()

    # ------------------------------------------------------------------
    # Async synthesis loop
    # ------------------------------------------------------------------

    async def _synthesis_loop(self) -> None:
        """Continuously synthesize PCM frames and broadcast over WebSocket."""
        self._running = True
        self._frame_counter = 0
        print("[Z³ Audio] Synthesis loop started.")
        try:
            while self._running:
                self._frame_counter += 1

                pcm_bytes = self._synthesize_frame()

                if self.ws_manager is not None:
                    # Broadcast audio PCM to all clients
                    await self.ws_manager.broadcast_bytes(pcm_bytes)

                    # Every N frames, broadcast W matrix + velocity as JSON
                    if self._frame_counter % METRICS_EVERY_FRAMES == 0:
                        with self._lock:
                            matrix = self._W.tolist()
                            velocity = self._learning_velocity
                        asyncio.create_task(
                            self.ws_manager.send_json({
                                "type": "metrics_update",
                                "matrix": matrix,
                                "velocity": velocity,
                            })
                        )

                # Sleep ~80ms to stay slightly ahead of the 100ms chunk boundary
                await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            print("[Z³ Audio] Synthesis loop stopped.")

    def start(self) -> None:
        """Schedule the synthesis loop as an asyncio background task."""
        if self._synth_task is None or self._synth_task.done():
            loop = asyncio.get_event_loop()
            self._synth_task = loop.create_task(self._synthesis_loop())

    def stop(self) -> None:
        """Cancel the synthesis loop."""
        self._running = False
        if self._synth_task and not self._synth_task.done():
            self._synth_task.cancel()

    # ------------------------------------------------------------------
    # HTTP streaming fallback (for clients that cannot use WebSocket)
    # ------------------------------------------------------------------

    async def http_stream_generator(self):
        """Async generator yielding PCM bytes for FastAPI StreamingResponse."""
        print("[Z³ Audio] HTTP stream client connected.")
        try:
            while True:
                pcm_bytes = self._synthesize_frame()
                yield pcm_bytes
                await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            print("[Z³ Audio] HTTP stream client disconnected.")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "agent_count": self.agent_count,
                "sample_rate": self.sample_rate,
                "chunk_samples": self.chunk_samples,
                "amplitude_scale": self.amplitude_scale,
                "state_version": self._state_version,
                "learning_velocity": round(self._learning_velocity, 8),
                "intrinsic_freqs_hz": self.intrinsic_freqs.tolist(),
                "ws_connections": self.ws_manager.connection_count if self.ws_manager else 0,
                "weights_file": self.weights_file,
            }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_SYNTH: Optional[Z3AudioSynthesizer] = None
_WS_MANAGER: Optional[AudioConnectionManager] = None


def get_synthesizer(agent_count: int = 8) -> Z3AudioSynthesizer:
    """Return the module-level Z3AudioSynthesizer singleton."""
    global _SYNTH
    if _SYNTH is None:
        weights_path = os.environ.get(
            "Z3_AUDIO_WEIGHTS_FILE",
            os.path.join(
                os.environ.get("Z3_STATE_DIR",
                               os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "data")),
                "z3_memory_matrix.npy",
            ),
        )
        _SYNTH = Z3AudioSynthesizer(agent_count=agent_count, weights_file=weights_path)
    return _SYNTH


def get_ws_manager() -> AudioConnectionManager:
    """Return the module-level WebSocket connection manager singleton."""
    global _WS_MANAGER
    if _WS_MANAGER is None:
        _WS_MANAGER = AudioConnectionManager()
    return _WS_MANAGER
