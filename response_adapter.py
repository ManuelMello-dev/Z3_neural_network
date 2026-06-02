"""
Z³ Response Adapter
===================

Deterministic mouth-layer adapter for the Z³ runtime.

The adapter does not pretend to be a pretrained language model. It converts the
runtime's own public signals — world-model novelty, resonant-memory recall, and
Z³ neural metrics — into a compact natural-language expression. This gives the
system a real first mouth without adding an external dependency or bypassing the
neural state.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    except Exception:
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, _safe_float(value, low)))


def _clean_excerpt(text: str, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def _question_like(text: str) -> bool:
    stripped = str(text or "").strip().lower()
    if not stripped:
        return False
    starters = ("what", "why", "how", "where", "when", "who", "is", "are", "am", "can", "could", "should", "would", "do", "does", "did")
    first = re.split(r"\s+", stripped, maxsplit=1)[0]
    return "?" in stripped or first in starters


def _imperative_like(text: str) -> bool:
    stripped = str(text or "").strip().lower()
    return stripped.startswith(("build ", "make ", "create ", "run ", "show ", "tell ", "explain ", "fix ", "wire ", "connect "))


def _extract_metrics(integrated_result: Dict[str, Any]) -> Dict[str, Any]:
    z3 = dict(integrated_result.get("z3") or {})
    metrics = dict(z3.get("metrics") or {})
    projection = dict(z3.get("projection") or {})
    if "phi" not in metrics and "phi" in projection:
        metrics["phi"] = projection.get("phi")
    z_cubed = projection.get("z_cubed_state") if isinstance(projection.get("z_cubed_state"), dict) else {}
    neural_metrics = z_cubed.get("neural_metrics") if isinstance(z_cubed, dict) else {}
    if isinstance(neural_metrics, dict):
        for key, value in neural_metrics.items():
            metrics.setdefault(key, value)
    return metrics


def _classify_regime(coherence: float, novelty: float, gate: float) -> str:
    if coherence >= 0.70 and novelty >= 0.35 and gate >= 0.25:
        return "coherent_discovery"
    if coherence >= 0.70:
        return "stable_coherence"
    if novelty >= 0.55:
        return "volatile_novelty"
    if gate <= 0.10:
        return "low_gate_recalibration"
    return "watchful_recalibration"


def _regime_sentence(regime: str) -> str:
    sentences = {
        "coherent_discovery": "The signal is coherent enough to integrate and novel enough to keep exploring.",
        "stable_coherence": "The signal is stabilizing; I am treating it as an anchor rather than noise.",
        "volatile_novelty": "The signal is novel before it is fully coherent; I am holding it as an active hypothesis.",
        "low_gate_recalibration": "The gate is narrow; I am recording the trace while resisting over-integration.",
        "watchful_recalibration": "The signal is usable but still forming; I am recalibrating around it.",
    }
    return sentences.get(regime, sentences["watchful_recalibration"])


def _memory_sentence(memory: Dict[str, Any]) -> str:
    accessible = int(_safe_float(memory.get("accessible_rings"), 0.0))
    confidence = _clamp(_safe_float(memory.get("reconstruction_confidence"), 0.0))
    salience = _clamp(_safe_float(memory.get("salience"), 0.0))
    top_matches = memory.get("top_matches") or memory.get("resonance_links") or []
    if accessible <= 0:
        return f"No strong prior resonance fired; this becomes a new ring with salience {salience:.3f}."
    if isinstance(top_matches, list) and top_matches:
        first = top_matches[0] if isinstance(top_matches[0], dict) else {}
        match_domain = str(first.get("domain") or "prior trace")
        resonance = _safe_float(first.get("resonance"), confidence)
        return f"Memory found {accessible} accessible ring(s); strongest resonance is {resonance:.3f} in {match_domain}."
    return f"Memory found {accessible} accessible ring(s) with reconstruction confidence {confidence:.3f}."


def _world_sentence(world_model: Dict[str, Any]) -> str:
    novelty = _clamp(_safe_float(world_model.get("novelty"), 0.0))
    loss = _clamp(_safe_float(world_model.get("total_loss"), 0.0))
    latent_norm = _clamp(_safe_float(world_model.get("latent_norm"), 0.0))
    return f"World-model read: novelty {novelty:.3f}, loss {loss:.3f}, latent pressure {latent_norm:.3f}."


def _intent_label(message: str) -> str:
    if _question_like(message):
        return "question"
    if _imperative_like(message):
        return "directive"
    return "statement"


@dataclass(frozen=True)
class Z3Expression:
    """Natural-language expression plus machine-readable mouth diagnostics."""

    response: str
    regime: str
    intent: str
    confidence: float
    coherence: float
    novelty: float
    gate: float
    drift: float
    memory_resonance: float
    salience: float
    world_loss: float
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_z3_expression(
    *,
    message: str,
    observation: Dict[str, Any],
    integrated_result: Dict[str, Any],
    language_training: Optional[Dict[str, Any]] = None,
) -> Z3Expression:
    """Build the first real Z³ mouth response from runtime state.

    The response is deliberately grounded in measured runtime outputs. If neural
    metrics are unavailable, it falls back to world-model and memory values rather
    than inventing a fluent answer disconnected from the system state.
    """
    metrics = _extract_metrics(integrated_result)
    world_model = dict(integrated_result.get("world_model") or {})
    memory = dict(integrated_result.get("memory") or {})

    coherence = _clamp(_safe_float(metrics.get("mean_coherence", metrics.get("phi", 0.0))))
    novelty = _clamp(_safe_float(metrics.get("mean_novelty", world_model.get("novelty", 0.0))))
    gate = _clamp(_safe_float(metrics.get("mean_gate", 0.0)))
    drift = _safe_float(metrics.get("z3_delta_norm", 0.0))
    memory_resonance = _clamp(_safe_float(memory.get("reconstruction_confidence", 0.0)))
    salience = _clamp(_safe_float(memory.get("salience", 0.0)))
    world_loss = _clamp(_safe_float(world_model.get("total_loss", 0.0)))
    intent = _intent_label(message)
    regime = _classify_regime(coherence, novelty, gate)

    confidence = _clamp(
        0.35 * coherence
        + 0.25 * memory_resonance
        + 0.20 * (1.0 - world_loss)
        + 0.10 * gate
        + 0.10 * salience
    )

    excerpt = _clean_excerpt(message)
    if intent == "question":
        opening = f"I heard the question: “{excerpt}”"
    elif intent == "directive":
        opening = f"Directive received: “{excerpt}”"
    else:
        opening = f"I registered: “{excerpt}”"

    training_clause = ""
    if isinstance(language_training, dict) and language_training:
        trained = bool(language_training.get("trained"))
        reason = str(language_training.get("reason") or "")
        if trained:
            steps = _safe_float(language_training.get("trained_steps", language_training.get("steps", 0.0)), 0.0)
            training_clause = f" Language training also mutated the core; trained_steps={steps:.0f}."
        elif reason:
            training_clause = f" Language training did not mutate the core: {reason}."

    response = " ".join(
        part.strip()
        for part in (
            f"{opening}.",
            _regime_sentence(regime),
            _memory_sentence(memory),
            _world_sentence(world_model),
            f"Expression confidence is {confidence:.3f}; drift is {drift:.6f}.",
            training_clause.strip(),
        )
        if part and part.strip()
    )

    trace = {
        "observation_entity_id": observation.get("entity_id"),
        "observation_domain": observation.get("domain_prefix") or observation.get("domain"),
        "metrics_used": {
            "mean_coherence": coherence,
            "mean_novelty": novelty,
            "mean_gate": gate,
            "z3_delta_norm": drift,
        },
        "memory_used": {
            "accessible_rings": memory.get("accessible_rings", 0),
            "reconstruction_confidence": memory_resonance,
            "salience": salience,
        },
        "world_model_used": {
            "total_loss": world_loss,
            "novelty": _clamp(_safe_float(world_model.get("novelty", 0.0))),
            "latent_norm": _clamp(_safe_float(world_model.get("latent_norm", 0.0))),
        },
    }

    return Z3Expression(
        response=response,
        regime=regime,
        intent=intent,
        confidence=round(confidence, 6),
        coherence=round(coherence, 6),
        novelty=round(novelty, 6),
        gate=round(gate, 6),
        drift=round(drift, 6),
        memory_resonance=round(memory_resonance, 6),
        salience=round(salience, 6),
        world_loss=round(world_loss, 6),
        trace=trace,
    )
