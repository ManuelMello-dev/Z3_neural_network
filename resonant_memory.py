"""
Resonant Memory Geometry
========================
Models memory as a sequence of phase-related state rings rather than a bag of
stored copies. Each observation becomes a ring in temporal state-space and is
recalled by resonance with later rings.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List



def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))



def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


@dataclass
class ResonantRing:
    """A temporal ring in the mesh memory geometry."""

    ring_id: str
    timestamp: float
    domain: str
    entity_id: str
    phase_position: float
    salience: float
    state_vector: Dict[str, float] = field(default_factory=dict)
    anchors: List[str] = field(default_factory=list)
    resonance_links: List[Dict[str, Any]] = field(default_factory=list)
    reconstruction_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ring_id": self.ring_id,
            "timestamp": self.timestamp,
            "domain": self.domain,
            "entity_id": self.entity_id,
            "phase_position": round(self.phase_position, 6),
            "salience": round(self.salience, 6),
            "state_vector": {k: round(v, 6) for k, v in self.state_vector.items()},
            "anchors": list(self.anchors),
            "resonance_links": [dict(link) for link in self.resonance_links],
            "reconstruction_confidence": round(self.reconstruction_confidence, 6),
        }


class ResonantMemoryGeometry:
    """
    Memory as geometry.

    The present ring does not retrieve a past copy. It probes prior rings for
    phase-compatible structure and reconstructs the most coherent latent trace.
    """

    def __init__(
        self,
        max_rings: int = 256,
        resonance_horizon: int = 72,
        phase_decay: float = 0.045,
    ) -> None:
        self.max_rings = max_rings
        self.resonance_horizon = resonance_horizon
        self.phase_decay = phase_decay
        self.rings: Deque[ResonantRing] = deque(maxlen=max_rings)
        self.total_observations = 0
        self.total_resonance_events = 0
        self.total_reconstructions = 0
        self.peak_resonance = 0.0
        self.average_resonance = 0.0
        self.last_reconstruction_confidence = 0.0
        self.phi_access_window = 0
        self.last_top_matches: List[Dict[str, Any]] = []

    def _extract_anchors(self, observation: Dict[str, Any], domain: str) -> List[str]:
        anchors = []
        entity_id = str(observation.get("entity_id") or observation.get("symbol") or "").strip().upper()
        if entity_id:
            anchors.append(entity_id)

        domain_prefix = str(observation.get("domain_prefix") or (domain.split(":")[0] if ":" in domain else domain)).strip().lower()
        if domain_prefix:
            anchors.append(domain_prefix)

        direction = str(observation.get("direction") or "").strip().lower()
        if direction:
            anchors.append(direction)

        for key in ("concept", "concept_id", "pattern_id"):
            value = str(observation.get(key) or "").strip().lower()
            if value:
                anchors.append(value)

        numeric_markers = {
            "price_band": round(_safe_float(observation.get("price") or observation.get("value"), 0.0), 1),
            "pct_band": round(_safe_float(observation.get("pct_change"), 0.0), 1),
            "vol_band": round(_safe_float(observation.get("volatility_5"), 0.0), 1),
        }
        for key, value in numeric_markers.items():
            anchors.append(f"{key}:{value}")

        return sorted({a for a in anchors if a})

    def _build_state_vector(
        self,
        observation: Dict[str, Any],
        constitutional_context: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        price = _safe_float(observation.get("price") or observation.get("value"), 0.0)
        volume = _safe_float(observation.get("volume"), 0.0)
        pct_change = _safe_float(observation.get("pct_change"), 0.0)
        volatility = _safe_float(observation.get("volatility_5"), 0.0)
        secondary = _safe_float(observation.get("secondary_value"), 0.0)
        constitutional_context = constitutional_context or {}
        phi = _safe_float(
            observation.get("constitutional_phi", constitutional_context.get("phi")),
            0.5,
        )
        sigma = _safe_float(
            observation.get("constitutional_sigma", constitutional_context.get("sigma")),
            0.5,
        )
        coherence = _safe_float(
            observation.get("constitutional_coherence", constitutional_context.get("coherence")),
            0.0,
        )
        drift = _safe_float(
            observation.get("constitutional_drift", constitutional_context.get("drift")),
            0.0,
        )
        collapse_probability = _safe_float(
            observation.get("constitutional_collapse_probability", constitutional_context.get("collapse_probability")),
            0.0,
        )
        wave_state = constitutional_context.get("wave_state", {}) if constitutional_context else {}
        checkpoint_state = constitutional_context.get("checkpoint_state", {}) if constitutional_context else {}
        interference_state = constitutional_context.get("interference_state", {}) if constitutional_context else {}
        logos_state = constitutional_context.get("logos_state", {}) if constitutional_context else {}
        wave_coherence = _safe_float(
            observation.get("constitutional_wave_coherence", wave_state.get("coherence")),
            0.0,
        )
        checkpoint_continuity = _safe_float(
            observation.get("constitutional_checkpoint_continuity", checkpoint_state.get("continuity")),
            0.0,
        )
        checkpoint_amplification = _safe_float(
            observation.get("constitutional_checkpoint_amplification", checkpoint_state.get("amplification")),
            0.0,
        )
        interference_net = _safe_float(
            observation.get("constitutional_interference_net", interference_state.get("net")),
            0.0,
        )
        logos_reflective_energy = _safe_float(
            observation.get("constitutional_logos_reflective_energy", logos_state.get("reflective_energy")),
            0.0,
        )
        logos_resonance = _safe_float(
            observation.get("constitutional_logos_resonance", logos_state.get("resonance")),
            0.0,
        )
        logos_clarity = _safe_float(
            observation.get("constitutional_logos_clarity", logos_state.get("clarity")),
            0.0,
        )
        logos_depth = _safe_float(
            observation.get("constitutional_logos_depth", logos_state.get("depth")),
            0.0,
        )
        logos_stability = _safe_float(
            observation.get("constitutional_logos_stability", logos_state.get("stability")),
            0.0,
        )

        return {
            "price": math.tanh(price / 1000.0),
            "volume": math.tanh(volume / 1000000.0),
            "pct_change": math.tanh(pct_change / 10.0),
            "volatility": math.tanh(volatility / 10.0),
            "secondary": math.tanh(secondary / 1000.0),
            "phi": (2.0 * _clamp(phi)) - 1.0,
            "sigma": (2.0 * _clamp(sigma)) - 1.0,
            "coherence": math.tanh(coherence * 4.0),
            "drift": math.tanh(drift * 4.0),
            "collapse_probability": (2.0 * _clamp(collapse_probability)) - 1.0,
            "wave_coherence": (2.0 * _clamp(wave_coherence)) - 1.0,
            "checkpoint_continuity": (2.0 * _clamp(checkpoint_continuity)) - 1.0,
            "checkpoint_amplification": (2.0 * _clamp(checkpoint_amplification)) - 1.0,
            "interference_net": max(-1.0, min(1.0, interference_net)),
            "logos_reflective_energy": math.tanh(logos_reflective_energy * 4.0),
            "logos_resonance": (2.0 * _clamp(logos_resonance)) - 1.0,
            "logos_clarity": (2.0 * _clamp(logos_clarity)) - 1.0,
            "logos_depth": (2.0 * _clamp(logos_depth)) - 1.0,
            "logos_stability": (2.0 * _clamp(logos_stability)) - 1.0,
        }

    def _phase_position(self, anchors: List[str], state_vector: Dict[str, float]) -> float:
        anchor_mass = sum(sum(ord(ch) for ch in anchor) for anchor in anchors)
        vector_mass = sum((idx + 1) * int(abs(value) * 1000) for idx, value in enumerate(state_vector.values()))
        raw = (anchor_mass + vector_mass) % 3600
        return (raw / 3600.0) * (2.0 * math.pi)

    def _vector_similarity(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        dot = sum(a.get(key, 0.0) * b.get(key, 0.0) for key in set(a) | set(b))
        norm_a = math.sqrt(sum(value * value for value in a.values()))
        norm_b = math.sqrt(sum(value * value for value in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        cosine = dot / (norm_a * norm_b)
        return _clamp((cosine + 1.0) / 2.0)

    def _anchor_overlap(self, a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        sa = set(a)
        sb = set(b)
        union = sa | sb
        if not union:
            return 0.0
        return len(sa & sb) / len(union)

    def _phase_alignment(self, phase_a: float, phase_b: float) -> float:
        diff = abs(phase_a - phase_b)
        diff = min(diff, (2.0 * math.pi) - diff)
        return _clamp(1.0 - (diff / math.pi))

    def observe(
        self,
        observation: Dict[str, Any],
        domain: str,
        phi_hint: float = 0.5,
        sigma_hint: float = 0.5,
        constitutional_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        now = _safe_float(observation.get("timestamp"), time.time())
        entity_id = str(observation.get("entity_id") or observation.get("symbol") or "unknown").strip().upper()
        anchors = self._extract_anchors(observation, domain)
        constitutional_context = constitutional_context or {}
        regime = str(observation.get("constitutional_regime") or constitutional_context.get("regime") or "critical").strip().lower()
        if regime:
            anchors.append(f"regime:{regime}")
        attractor_id = str(constitutional_context.get("attractor_id") or "").strip()
        if attractor_id:
            anchors.append(attractor_id)
        state_vector = self._build_state_vector(observation, constitutional_context=constitutional_context)
        phase_position = self._phase_position(anchors, state_vector)

        candidate_links: List[Dict[str, Any]] = []
        prior_rings = list(self.rings)[-self.resonance_horizon :]
        for age, prior in enumerate(reversed(prior_rings), start=1):
            vector_similarity = self._vector_similarity(state_vector, prior.state_vector)
            anchor_overlap = self._anchor_overlap(anchors, prior.anchors)
            phase_alignment = self._phase_alignment(phase_position, prior.phase_position)
            temporal_decay = math.exp(-self.phase_decay * max(age - 1, 0))
            resonance_score = (
                0.45 * vector_similarity
                + 0.30 * phase_alignment
                + 0.25 * anchor_overlap
            )
            resonance_score *= temporal_decay
            resonance_score *= (0.55 + 0.45 * _clamp(phi_hint))
            resonance_score *= (1.0 - 0.20 * _clamp(sigma_hint))
            resonance_score = _clamp(resonance_score)

            if resonance_score >= 0.33:
                candidate_links.append(
                    {
                        "ring_id": prior.ring_id,
                        "entity_id": prior.entity_id,
                        "domain": prior.domain,
                        "resonance": round(resonance_score, 6),
                        "phase_alignment": round(phase_alignment, 6),
                        "anchor_overlap": round(anchor_overlap, 6),
                        "vector_similarity": round(vector_similarity, 6),
                        "age": age,
                    }
                )

        candidate_links.sort(key=lambda item: item["resonance"], reverse=True)
        top_links = candidate_links[:5]
        accessible_rings = len(candidate_links)
        reconstruction_confidence = 0.0
        if top_links:
            reconstruction_confidence = _clamp(
                (sum(link["resonance"] for link in top_links) / len(top_links))
                * min(1.0, accessible_rings / 6.0)
            )

        salience = _clamp(
            0.28
            + 0.28 * reconstruction_confidence
            + 0.12 * _clamp(phi_hint)
            + 0.10 * _clamp(abs(state_vector.get("pct_change", 0.0)))
            + 0.10 * _clamp(abs(state_vector.get("coherence", 0.0)))
            + 0.06 * _clamp(abs(state_vector.get("wave_coherence", 0.0)))
            + 0.06 * _clamp(abs(state_vector.get("checkpoint_continuity", 0.0)))
            + 0.06 * _clamp(abs(state_vector.get("checkpoint_amplification", 0.0)))
            + 0.05 * _clamp(abs(state_vector.get("interference_net", 0.0)))
            + 0.04 * _clamp(abs(state_vector.get("logos_reflective_energy", 0.0)))
            + 0.03 * _clamp(abs(state_vector.get("logos_resonance", 0.0)))
            + 0.03 * _clamp(abs(state_vector.get("logos_clarity", 0.0)))
            + 0.03 * _clamp(abs(state_vector.get("logos_depth", 0.0)))
            + 0.03 * _clamp(abs(state_vector.get("logos_stability", 0.0)))
        )

        ring = ResonantRing(
            ring_id=f"ring_{self.total_observations + 1}",
            timestamp=now,
            domain=domain,
            entity_id=entity_id,
            phase_position=phase_position,
            salience=salience,
            state_vector=state_vector,
            anchors=anchors,
            resonance_links=top_links,
            reconstruction_confidence=reconstruction_confidence,
        )
        self.rings.append(ring)

        self.total_observations += 1
        if top_links:
            self.total_resonance_events += len(top_links)
            self.total_reconstructions += 1
            self.peak_resonance = max(self.peak_resonance, top_links[0]["resonance"])
            total_events = max(self.total_reconstructions, 1)
            self.average_resonance = (
                ((self.average_resonance * (total_events - 1)) + top_links[0]["resonance"])
                / total_events
            )
        self.last_reconstruction_confidence = reconstruction_confidence
        self.phi_access_window = max(self.phi_access_window, accessible_rings)
        self.last_top_matches = top_links

        return {
            "ring_id": ring.ring_id,
            "accessible_rings": accessible_rings,
            "reconstruction_confidence": round(reconstruction_confidence, 6),
            "top_matches": top_links,
            "phase_position": round(phase_position, 6),
            "salience": round(salience, 6),
            "checkpoint_bias": round(_clamp(abs(state_vector.get("checkpoint_continuity", 0.0))), 6),
            "interference_bias": round(max(-1.0, min(1.0, state_vector.get("interference_net", 0.0))), 6),
            "logos_bias": round(_clamp(abs(state_vector.get("logos_reflective_energy", 0.0))), 6),
            "logos_resonance_bias": round(_clamp(abs(state_vector.get("logos_resonance", 0.0))), 6),
            "logos_clarity_bias": round(_clamp(abs(state_vector.get("logos_clarity", 0.0))), 6),
        }

    def get_snapshot(self, recent_ring_count: int = 12) -> Dict[str, Any]:
        unique_anchors = len({anchor for ring in self.rings for anchor in ring.anchors})
        return {
            "metrics": {
                "rings": len(self.rings),
                "total_observations": self.total_observations,
                "total_resonance_events": self.total_resonance_events,
                "total_reconstructions": self.total_reconstructions,
                "peak_resonance": round(self.peak_resonance, 6),
                "average_resonance": round(self.average_resonance, 6),
                "last_reconstruction_confidence": round(self.last_reconstruction_confidence, 6),
                "phi_access_window": self.phi_access_window,
                "state_space_size": unique_anchors,
            },
            "recent_rings": [ring.to_dict() for ring in list(self.rings)[-recent_ring_count:]],
            "top_matches": list(self.last_top_matches),
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "config": {
                "max_rings": self.max_rings,
                "resonance_horizon": self.resonance_horizon,
                "phase_decay": self.phase_decay,
            },
            "rings": [ring.to_dict() for ring in self.rings],
            "metrics": {
                "total_observations": self.total_observations,
                "total_resonance_events": self.total_resonance_events,
                "total_reconstructions": self.total_reconstructions,
                "peak_resonance": self.peak_resonance,
                "average_resonance": self.average_resonance,
                "last_reconstruction_confidence": self.last_reconstruction_confidence,
                "phi_access_window": self.phi_access_window,
            },
            "last_top_matches": list(self.last_top_matches),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return

        config = state.get("config", {})
        self.max_rings = int(config.get("max_rings", self.max_rings) or self.max_rings)
        self.resonance_horizon = int(config.get("resonance_horizon", self.resonance_horizon) or self.resonance_horizon)
        self.phase_decay = float(config.get("phase_decay", self.phase_decay) or self.phase_decay)
        self.rings = deque(maxlen=self.max_rings)

        for raw_ring in state.get("rings", []):
            self.rings.append(
                ResonantRing(
                    ring_id=str(raw_ring.get("ring_id", f"ring_{len(self.rings) + 1}")),
                    timestamp=_safe_float(raw_ring.get("timestamp"), time.time()),
                    domain=str(raw_ring.get("domain", "general")),
                    entity_id=str(raw_ring.get("entity_id", "unknown")),
                    phase_position=_safe_float(raw_ring.get("phase_position"), 0.0),
                    salience=_safe_float(raw_ring.get("salience"), 0.0),
                    state_vector={k: _safe_float(v, 0.0) for k, v in dict(raw_ring.get("state_vector", {})).items()},
                    anchors=list(raw_ring.get("anchors", [])),
                    resonance_links=[dict(link) for link in raw_ring.get("resonance_links", [])],
                    reconstruction_confidence=_safe_float(raw_ring.get("reconstruction_confidence"), 0.0),
                )
            )

        metrics = state.get("metrics", {})
        self.total_observations = int(metrics.get("total_observations", len(self.rings)) or len(self.rings))
        self.total_resonance_events = int(metrics.get("total_resonance_events", 0) or 0)
        self.total_reconstructions = int(metrics.get("total_reconstructions", 0) or 0)
        self.peak_resonance = _safe_float(metrics.get("peak_resonance"), 0.0)
        self.average_resonance = _safe_float(metrics.get("average_resonance"), 0.0)
        self.last_reconstruction_confidence = _safe_float(metrics.get("last_reconstruction_confidence"), 0.0)
        self.phi_access_window = int(metrics.get("phi_access_window", 0) or 0)
        self.last_top_matches = [dict(link) for link in state.get("last_top_matches", [])]
