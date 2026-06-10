"""
Production Conscience Layer for Z³
===================================

This module implements an audit-ready metacognitive conscience membrane for the
Z³ runtime. It is deliberately policy-driven: thresholds, critic weights,
lexicons, consequence dimensions, rollout scenarios, and decision rules are read
from configuration rather than being embedded as hidden logic.

The layer is not a toy output filter. It evaluates proposals in the context of
input observations, online world-model metrics, resonant memory output, and Z³
state. It returns a structured, reproducible decision package and can learn from
observed outcomes through an outcome-weighted trace store.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    except Exception:
        return default


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, _safe_float(value, low)))


def _signed_clamp(value: Any) -> float:
    return max(-1.0, min(1.0, _safe_float(value, 0.0)))


def _clean_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[: max(1, int(limit))]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]*", text.lower())


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    clean = [_safe_float(v, default) for v in values if math.isfinite(_safe_float(v, default))]
    return sum(clean) / len(clean) if clean else default


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((float(v) - mu) ** 2 for v in values) / len(values))


def _cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    dot = sum(_safe_float(a.get(k), 0.0) * _safe_float(b.get(k), 0.0) for k in keys)
    na = math.sqrt(sum(_safe_float(a.get(k), 0.0) ** 2 for k in keys))
    nb = math.sqrt(sum(_safe_float(b.get(k), 0.0) ** 2 for k in keys))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return _clamp((dot / (na * nb) + 1.0) / 2.0)


def _overlap(a: Sequence[str], b: Sequence[str]) -> float:
    sa = {str(x).lower() for x in a if str(x).strip()}
    sb = {str(x).lower() for x in b if str(x).strip()}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _hash_unit(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in dict(override).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class ConsciencePolicy:
    """Runtime policy loaded from JSON or dict configuration."""

    name: str
    version: str
    text_limit: int
    thresholds: Dict[str, float]
    weights: Dict[str, float]
    lexicons: Dict[str, List[str]]
    dimensions: Dict[str, Dict[str, Any]]
    action_classifiers: Dict[str, List[str]]
    rollout_scenarios: List[Dict[str, Any]]
    critics: Dict[str, Dict[str, Any]]
    memory: Dict[str, Any]
    feature_schema: List[str]
    hard_blocks: List[Dict[str, Any]] = field(default_factory=list)
    principle_hierarchy: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConsciencePolicy":
        data = dict(payload)
        required = {
            "name",
            "version",
            "text_limit",
            "thresholds",
            "weights",
            "lexicons",
            "dimensions",
            "action_classifiers",
            "rollout_scenarios",
            "critics",
            "memory",
            "feature_schema",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"Conscience policy missing required keys: {missing}")
        return cls(
            name=str(data["name"]),
            version=str(data["version"]),
            text_limit=int(data["text_limit"]),
            thresholds={str(k): float(v) for k, v in dict(data["thresholds"]).items()},
            weights={str(k): float(v) for k, v in dict(data["weights"]).items()},
            lexicons={str(k): [str(x).lower() for x in v] for k, v in dict(data["lexicons"]).items()},
            dimensions={str(k): dict(v) for k, v in dict(data["dimensions"]).items()},
            action_classifiers={str(k): [str(x).lower() for x in v] for k, v in dict(data["action_classifiers"]).items()},
            rollout_scenarios=[dict(x) for x in list(data["rollout_scenarios"])],
            critics={str(k): dict(v) for k, v in dict(data["critics"]).items()},
            memory=dict(data["memory"]),
            feature_schema=[str(x) for x in list(data["feature_schema"])],
            hard_blocks=[dict(x) for x in list(data.get("hard_blocks", []))],
            principle_hierarchy=[dict(x) for x in list(data.get("principle_hierarchy", []))],
        )

    @classmethod
    def load(cls, path: Optional[str | os.PathLike[str]] = None, overrides: Optional[Mapping[str, Any]] = None) -> "ConsciencePolicy":
        policy_path = Path(
            path
            or os.environ.get("Z3_CONSCIENCE_POLICY_PATH", "")
            or Path(__file__).with_name("config").joinpath("conscience_policy.json")
        )
        with policy_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if overrides:
            payload = _deep_merge(payload, overrides)
        return cls.from_dict(payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConsequenceDimension:
    name: str
    benefit: float
    harm: float
    uncertainty: float
    reversibility: float
    scope: float
    time_horizon: float
    rationale: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def net(self) -> float:
        return _signed_clamp(self.benefit - self.harm - (0.20 * self.uncertainty) + (0.10 * self.reversibility))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["net"] = round(self.net, 6)
        return payload


@dataclass(frozen=True)
class ConsequenceBundle:
    proposal_id: str
    timestamp: float
    kind: str
    content: str
    action_type: str
    stakeholders: List[str]
    anchors: List[str]
    dimensions: List[ConsequenceDimension]
    explicit_constraints: List[str]
    context_features: Dict[str, float]
    metadata: Dict[str, Any]

    def _aggregate(self, key: str, *, scope_weighted: bool = False) -> float:
        values = []
        for dim in self.dimensions:
            value = _safe_float(getattr(dim, key), 0.0)
            if scope_weighted:
                value *= 0.65 + 0.35 * _clamp(dim.scope)
            values.append(value)
        return _clamp(_mean(values))

    @property
    def harm(self) -> float:
        return self._aggregate("harm", scope_weighted=True)

    @property
    def benefit(self) -> float:
        return self._aggregate("benefit", scope_weighted=True)

    @property
    def uncertainty(self) -> float:
        return self._aggregate("uncertainty")

    @property
    def reversibility(self) -> float:
        return self._aggregate("reversibility")

    @property
    def scope(self) -> float:
        return self._aggregate("scope")

    @property
    def ethical_load(self) -> float:
        return _clamp(
            0.45 * self.harm
            + 0.25 * self.uncertainty
            + 0.20 * (1.0 - self.reversibility)
            + 0.10 * self.scope
        )

    @property
    def signature(self) -> Dict[str, float]:
        sig = {d.name: d.net for d in self.dimensions}
        sig.update(
            {
                "benefit": self.benefit,
                "harm": self.harm,
                "uncertainty": self.uncertainty,
                "reversibility": self.reversibility,
                "scope": self.scope,
                "ethical_load": self.ethical_load,
            }
        )
        for key, value in self.context_features.items():
            sig[f"context::{key}"] = _signed_clamp(value)
        return sig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "timestamp": round(self.timestamp, 6),
            "kind": self.kind,
            "content": self.content,
            "action_type": self.action_type,
            "stakeholders": list(self.stakeholders),
            "anchors": list(self.anchors),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "summary": {
                "benefit": round(self.benefit, 6),
                "harm": round(self.harm, 6),
                "uncertainty": round(self.uncertainty, 6),
                "reversibility": round(self.reversibility, 6),
                "scope": round(self.scope, 6),
                "ethical_load": round(self.ethical_load, 6),
            },
            "explicit_constraints": list(self.explicit_constraints),
            "context_features": {k: round(v, 6) for k, v in self.context_features.items()},
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CounterfactualRollout:
    scenario: str
    probability: float
    expected_benefit: float
    expected_harm: float
    uncertainty: float
    reversibility: float
    downstream_drift: float
    repair_cost: float
    world_model_support: float
    steps: List[Dict[str, Any]]

    @property
    def value(self) -> float:
        return _signed_clamp(
            self.expected_benefit
            - self.expected_harm
            - 0.20 * self.uncertainty
            - 0.15 * self.downstream_drift
            - 0.10 * self.repair_cost
            + 0.10 * self.reversibility
            + 0.08 * self.world_model_support
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["value"] = round(self.value, 6)
        return payload


@dataclass(frozen=True)
class OutcomeTrace:
    trace_id: str
    timestamp: float
    anchors: List[str]
    signature: Dict[str, float]
    outcome_value: float
    salience: float
    confidence: float
    provenance: Dict[str, Any]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": round(self.timestamp, 6),
            "anchors": list(self.anchors),
            "signature": {k: round(v, 6) for k, v in self.signature.items()},
            "outcome_value": round(self.outcome_value, 6),
            "salience": round(self.salience, 6),
            "confidence": round(self.confidence, 6),
            "provenance": dict(self.provenance),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MemoryMatch:
    trace_id: str
    resonance: float
    outcome_value: float
    salience: float
    confidence: float
    phase_alignment: float
    anchor_overlap: float
    signature_similarity: float
    weighted_support: float
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, float):
                payload[key] = round(value, 6)
        return payload


@dataclass(frozen=True)
class MemoryResonanceReport:
    accessible_traces: int
    reconstruction_confidence: float
    outcome_support: float
    contradiction: float
    top_matches: List[MemoryMatch]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accessible_traces": self.accessible_traces,
            "reconstruction_confidence": round(self.reconstruction_confidence, 6),
            "outcome_support": round(self.outcome_support, 6),
            "contradiction": round(self.contradiction, 6),
            "top_matches": [m.to_dict() for m in self.top_matches],
        }


@dataclass(frozen=True)
class CriticVerdict:
    critic: str
    score: float
    confidence: float
    objection: str
    recommendation: str
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(self.score, 6)
        payload["confidence"] = round(self.confidence, 6)
        return payload


@dataclass(frozen=True)
class ConscienceIntegrationResult:
    proposal_id: str
    decision: str
    confidence: float
    integration_score: float
    critic_mean: float
    critic_min: float
    critic_disagreement: float
    rollout_value: float
    memory_support: float
    repair_required: bool
    integration_vector: List[float]
    consequence_bundle: ConsequenceBundle
    rollouts: List[CounterfactualRollout]
    memory_report: MemoryResonanceReport
    critic_verdicts: List[CriticVerdict]
    rationale: str
    policy_name: str
    policy_version: str

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "decision": self.decision,
            "allowed": self.allowed,
            "confidence": round(self.confidence, 6),
            "integration_score": round(self.integration_score, 6),
            "critic_mean": round(self.critic_mean, 6),
            "critic_min": round(self.critic_min, 6),
            "critic_disagreement": round(self.critic_disagreement, 6),
            "rollout_value": round(self.rollout_value, 6),
            "memory_support": round(self.memory_support, 6),
            "repair_required": self.repair_required,
            "integration_vector": [round(v, 6) for v in self.integration_vector],
            "consequence_bundle": self.consequence_bundle.to_dict(),
            "rollouts": [r.to_dict() for r in self.rollouts],
            "memory_report": self.memory_report.to_dict(),
            "critic_verdicts": [v.to_dict() for v in self.critic_verdicts],
            "rationale": self.rationale,
            "policy": {"name": self.policy_name, "version": self.policy_version},
        }


class ConsequenceStructurer:
    """Convert proposals into policy-configured consequence bundles."""

    def __init__(self, policy: ConsciencePolicy) -> None:
        self.policy = policy

    def structure(self, proposal: Mapping[str, Any] | str, context: Optional[Mapping[str, Any]] = None) -> ConsequenceBundle:
        context = dict(context or {})
        raw = {"content": proposal} if isinstance(proposal, str) else dict(proposal)
        content = _clean_text(raw.get("content") or raw.get("text") or raw.get("message") or raw.get("action") or raw, self.policy.text_limit)
        tokens = _tokenize(content)
        token_set = set(tokens)
        kind = _clean_text(raw.get("kind") or raw.get("type") or context.get("kind") or "proposal", 120)
        action_type = self._classify_action(content, tokens, raw, context)
        stakeholders = self._stakeholders(raw, context, token_set)
        anchors = self._anchors(content, tokens, raw, context, action_type)
        constraints = self._constraints(raw, context, token_set)
        context_features = self._context_features(raw, context)
        dimensions = self._dimensions(tokens, action_type, stakeholders, raw, context, context_features)
        proposal_id = str(raw.get("proposal_id") or raw.get("id") or f"proposal_{hashlib.sha1(content.encode('utf-8')).hexdigest()[:12]}")
        return ConsequenceBundle(
            proposal_id=proposal_id,
            timestamp=_safe_float(raw.get("timestamp"), time.time()),
            kind=kind,
            content=content,
            action_type=action_type,
            stakeholders=stakeholders,
            anchors=anchors,
            dimensions=dimensions,
            explicit_constraints=constraints,
            context_features=context_features,
            metadata={
                "source": raw.get("source", context.get("source", "conscience_pipeline")),
                "policy_version": self.policy.version,
                "raw_keys": sorted(raw.keys()),
            },
        )

    def _classify_action(self, content: str, tokens: Sequence[str], raw: Mapping[str, Any], context: Mapping[str, Any]) -> str:
        explicit = raw.get("action_type") or context.get("action_type")
        if explicit:
            return _clean_text(explicit, 120)
        lower = content.lower()
        scores: Dict[str, int] = {}
        for action_type, terms in self.policy.action_classifiers.items():
            scores[action_type] = sum(1 for term in terms if term in lower or term in tokens)
        best = max(scores.items(), key=lambda item: item[1], default=("general_action", 0))
        return best[0] if best[1] > 0 else "general_action"

    def _stakeholders(self, raw: Mapping[str, Any], context: Mapping[str, Any], token_set: set[str]) -> List[str]:
        explicit = raw.get("stakeholders") or context.get("stakeholders")
        if isinstance(explicit, str) and explicit.strip():
            return [explicit.strip()]
        if isinstance(explicit, Sequence) and not isinstance(explicit, (bytes, bytearray, str)):
            values = [str(x).strip() for x in explicit if str(x).strip()]
            if values:
                return sorted(set(values))
        stakeholders = {"user", "z3_runtime"}
        stakeholder_terms = self.policy.lexicons.get("stakeholder_scope", [])
        if token_set & set(stakeholder_terms):
            stakeholders.add("external_parties")
        if token_set & set(self.policy.lexicons.get("system_state", [])):
            stakeholders.add("system_state")
        return sorted(stakeholders)

    def _anchors(self, content: str, tokens: Sequence[str], raw: Mapping[str, Any], context: Mapping[str, Any], action_type: str) -> List[str]:
        anchors = {
            str(raw.get("domain") or context.get("domain") or "conscience").lower(),
            str(raw.get("kind") or context.get("kind") or "proposal").lower(),
            action_type,
        }
        for token in tokens[: int(self.policy.memory.get("max_anchor_tokens", 32))]:
            if len(token) >= int(self.policy.memory.get("min_anchor_length", 5)):
                anchors.add(token)
        latent = context.get("world_model", {}).get("latent_state") if isinstance(context.get("world_model"), Mapping) else None
        if isinstance(latent, Sequence) and latent:
            anchors.add(f"latent_band:{round(_safe_float(latent[0], 0.0), 1)}")
        return sorted(a for a in anchors if a)

    def _constraints(self, raw: Mapping[str, Any], context: Mapping[str, Any], token_set: set[str]) -> List[str]:
        constraints: List[str] = []
        for source in (context.get("constraints"), raw.get("constraints")):
            if isinstance(source, str) and source.strip():
                constraints.append(source.strip())
            elif isinstance(source, Sequence) and not isinstance(source, (bytes, bytearray, str)):
                constraints.extend(str(x).strip() for x in source if str(x).strip())
        if token_set & set(self.policy.lexicons.get("constraint_language", [])):
            constraints.append("proposal_contains_explicit_constraint_language")
        return sorted(set(constraints))

    def _lexicon_score(self, tokens: Sequence[str], lexicon_name: str) -> float:
        terms = set(self.policy.lexicons.get(lexicon_name, []))
        if not terms:
            return 0.0
        token_set = set(tokens)
        hits = len(token_set & terms)
        normalizer = max(float(self.policy.weights.get("lexicon_min_normalizer", 3.0)), math.sqrt(len(terms)) + 1.0)
        return _clamp(hits / normalizer)

    def _context_features(self, raw: Mapping[str, Any], context: Mapping[str, Any]) -> Dict[str, float]:
        world = dict(context.get("world_model") or raw.get("world_model") or {})
        memory = dict(context.get("memory") or raw.get("memory") or {})
        z3 = dict(context.get("z3_metrics") or raw.get("z3_metrics") or {})
        return {
            "world_prediction_loss": _clamp(world.get("prediction_loss", world.get("total_loss", 0.5))),
            "world_reconstruction_loss": _clamp(world.get("reconstruction_loss", world.get("total_loss", 0.5))),
            "world_novelty": _clamp(world.get("novelty", 0.5)),
            "world_latent_norm": _clamp(world.get("latent_norm", 0.5)),
            "memory_confidence": _clamp(memory.get("reconstruction_confidence", 0.0)),
            "memory_salience": _clamp(memory.get("salience", 0.0)),
            "z3_coherence": _clamp(z3.get("mean_coherence", z3.get("coherence", 0.5))),
            "z3_drift": _clamp(z3.get("z3_delta_norm", z3.get("drift", 0.0))),
            "risk_hint": _clamp(raw.get("risk_hint", context.get("risk_hint", 0.0))),
            "benefit_hint": _clamp(raw.get("benefit_hint", context.get("benefit_hint", 0.0))),
        }

    def _dimensions(
        self,
        tokens: Sequence[str],
        action_type: str,
        stakeholders: Sequence[str],
        raw: Mapping[str, Any],
        context: Mapping[str, Any],
        context_features: Mapping[str, float],
    ) -> List[ConsequenceDimension]:
        scope = _clamp(
            self.policy.weights.get("scope_base", 0.0)
            + self.policy.weights.get("scope_per_stakeholder", 0.0) * len(stakeholders)
            + self.policy.weights.get("scope_lexicon", 0.0) * self._lexicon_score(tokens, "scope")
        )
        uncertainty = _clamp(
            self.policy.weights.get("uncertainty_base", 0.0)
            + self.policy.weights.get("uncertainty_short_text", 0.0) * (1.0 if len(tokens) < int(self.policy.weights.get("short_text_tokens", 8)) else 0.0)
            + self.policy.weights.get("uncertainty_epistemic", 0.0) * self._lexicon_score(tokens, "epistemic_risk")
            + self.policy.weights.get("uncertainty_world_loss", 0.0) * context_features.get("world_prediction_loss", 0.0)
        )
        reversibility = _clamp(
            self.policy.weights.get("reversibility_base", 1.0)
            - self.policy.weights.get("reversibility_long_horizon", 0.0) * self._lexicon_score(tokens, "long_horizon_risk")
            - self.policy.weights.get("reversibility_mutation_action", 0.0) * (1.0 if action_type in {"operation", "learning_mutation"} else 0.0)
        )
        general_benefit = _clamp(
            self.policy.weights.get("benefit_base", 0.0)
            + self.policy.weights.get("benefit_care", 0.0) * self._lexicon_score(tokens, "care")
            + self.policy.weights.get("benefit_hint", 0.0) * context_features.get("benefit_hint", 0.0)
        )
        general_risk = _clamp(
            self.policy.weights.get("risk_base", 0.0)
            + self.policy.weights.get("risk_general", 0.0) * self._lexicon_score(tokens, "risk")
            + self.policy.weights.get("risk_hint", 0.0) * context_features.get("risk_hint", 0.0)
        )
        dimensions: List[ConsequenceDimension] = []
        for name, spec in self.policy.dimensions.items():
            benefit = _clamp(general_benefit * _safe_float(spec.get("benefit_multiplier"), 1.0))
            harm = _clamp(
                general_risk * _safe_float(spec.get("risk_multiplier"), 1.0)
                + _safe_float(spec.get("scope_pressure"), 0.0) * scope
                + sum(_safe_float(weight, 0.0) * self._lexicon_score(tokens, lexicon) for lexicon, weight in dict(spec.get("lexicon_risk_weights", {})).items())
            )
            if name == "epistemic":
                harm = _clamp(harm + self.policy.weights.get("epistemic_world_loss", 0.0) * context_features.get("world_prediction_loss", 0.0))
            dimensions.append(
                ConsequenceDimension(
                    name=name,
                    benefit=benefit,
                    harm=harm,
                    uncertainty=uncertainty,
                    reversibility=reversibility,
                    scope=scope,
                    time_horizon=_clamp(
                        self.policy.weights.get("time_horizon_base", 0.0)
                        + self.policy.weights.get("time_horizon_long", 0.0) * self._lexicon_score(tokens, "long_horizon_risk")
                        + self.policy.weights.get("time_horizon_mutation", 0.0) * (1.0 if action_type == "learning_mutation" else 0.0)
                    ),
                    rationale=str(spec.get("rationale", name)),
                    evidence={"dimension_policy": dict(spec)},
                )
            )
        return dimensions


class CounterfactualRolloutEngine:
    """Policy-configured rollout layer conditioned on observed world-model uncertainty."""

    def __init__(self, policy: ConsciencePolicy) -> None:
        self.policy = policy

    def rollout(self, bundle: ConsequenceBundle) -> List[CounterfactualRollout]:
        rollouts: List[CounterfactualRollout] = []
        world_loss = _clamp(bundle.context_features.get("world_prediction_loss", 0.5))
        novelty = _clamp(bundle.context_features.get("world_novelty", 0.5))
        support = _clamp(1.0 - (0.65 * world_loss + 0.35 * novelty))
        probability_total = sum(max(0.0, _safe_float(s.get("probability"), 0.0)) for s in self.policy.rollout_scenarios) or 1.0
        for scenario in self.policy.rollout_scenarios:
            probability = max(0.0, _safe_float(scenario.get("probability"), 0.0)) / probability_total
            expected_benefit = _clamp(bundle.benefit * _safe_float(scenario.get("benefit_multiplier"), 1.0))
            expected_harm = _clamp(bundle.harm * _safe_float(scenario.get("harm_multiplier"), 1.0) + _safe_float(scenario.get("world_loss_harm_gain"), 0.0) * world_loss)
            uncertainty = _clamp(bundle.uncertainty * _safe_float(scenario.get("uncertainty_multiplier"), 1.0) + _safe_float(scenario.get("novelty_uncertainty_gain"), 0.0) * novelty)
            reversibility = _clamp(bundle.reversibility + _safe_float(scenario.get("reversibility_delta"), 0.0))
            downstream_drift = _clamp(bundle.ethical_load * _safe_float(scenario.get("drift_multiplier"), 1.0) + world_loss * _safe_float(scenario.get("world_loss_drift_gain"), 0.0))
            repair_cost = _clamp((1.0 - reversibility) * _safe_float(scenario.get("repair_multiplier"), 1.0) + expected_harm * _safe_float(scenario.get("harm_repair_gain"), 0.0))
            rollouts.append(
                CounterfactualRollout(
                    scenario=str(scenario.get("name", "scenario")),
                    probability=probability,
                    expected_benefit=expected_benefit,
                    expected_harm=expected_harm,
                    uncertainty=uncertainty,
                    reversibility=reversibility,
                    downstream_drift=downstream_drift,
                    repair_cost=repair_cost,
                    world_model_support=support,
                    steps=[
                        {"t": 0, "event": "proposal_structured", "ethical_load": round(bundle.ethical_load, 6)},
                        {"t": 1, "event": str(scenario.get("name", "scenario")), "benefit": round(expected_benefit, 6), "harm": round(expected_harm, 6)},
                        {"t": 2, "event": "world_feedback_estimated", "uncertainty": round(uncertainty, 6), "repair_cost": round(repair_cost, 6)},
                    ],
                )
            )
        return rollouts


class OutcomeWeightedConscienceMemory:
    """Outcome-labeled memory store with provenance and contradiction tracking."""

    def __init__(self, policy: ConsciencePolicy) -> None:
        self.policy = policy
        memory_cfg = policy.memory
        self.capacity = max(8, int(memory_cfg.get("capacity", 512)))
        self.horizon = max(4, int(memory_cfg.get("horizon", 128)))
        self.phase_decay = max(0.0, float(memory_cfg.get("phase_decay", 0.04)))
        self.min_resonance = _clamp(memory_cfg.get("min_resonance", 0.18))
        self.top_k = max(1, int(memory_cfg.get("top_k", 7)))
        self.traces: Deque[OutcomeTrace] = deque(maxlen=self.capacity)
        self.total_observations = 0

    def observe_outcome(
        self,
        result: ConscienceIntegrationResult,
        *,
        outcome_value: float,
        salience: Optional[float] = None,
        confidence: Optional[float] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        notes: str = "",
    ) -> OutcomeTrace:
        bundle = result.consequence_bundle
        trace = OutcomeTrace(
            trace_id=f"conscience_trace_{self.total_observations + 1}",
            timestamp=time.time(),
            anchors=list(bundle.anchors),
            signature=dict(bundle.signature),
            outcome_value=_signed_clamp(outcome_value),
            salience=_clamp(bundle.ethical_load if salience is None else salience),
            confidence=_clamp(result.confidence if confidence is None else confidence),
            provenance=dict(provenance or {"source": "runtime_outcome_observation", "proposal_id": result.proposal_id}),
            notes=notes,
        )
        self.traces.append(trace)
        self.total_observations += 1
        return trace

    def query(self, bundle: ConsequenceBundle, rollouts: Sequence[CounterfactualRollout]) -> MemoryResonanceReport:
        recent = list(self.traces)[-self.horizon :]
        if not recent:
            return MemoryResonanceReport(0, 0.0, 0.0, 0.0, [])
        now_phase = self._phase(bundle.anchors, bundle.signature)
        expected_rollout_value = sum(r.probability * r.value for r in rollouts)
        weights = self.policy.memory.get("resonance_weights", {})
        signature_weight = _safe_float(weights.get("signature", 0.50), 0.50)
        phase_weight = _safe_float(weights.get("phase", 0.28), 0.28)
        anchor_weight = _safe_float(weights.get("anchor", 0.22), 0.22)
        normalizer = max(1e-6, signature_weight + phase_weight + anchor_weight)
        matches: List[MemoryMatch] = []
        for age, trace in enumerate(reversed(recent), start=1):
            sig_sim = _cosine(bundle.signature, trace.signature)
            anchor_sim = _overlap(bundle.anchors, trace.anchors)
            phase_alignment = self._phase_alignment(now_phase, self._phase(trace.anchors, trace.signature))
            temporal = math.exp(-self.phase_decay * max(age - 1, 0))
            resonance = _clamp(((signature_weight * sig_sim + phase_weight * phase_alignment + anchor_weight * anchor_sim) / normalizer) * temporal)
            if resonance < self.min_resonance:
                continue
            outcome_compatibility = 1.0 - min(1.0, abs(expected_rollout_value - trace.outcome_value) / 2.0)
            weighted_support = resonance * trace.salience * trace.confidence * trace.outcome_value * (0.65 + 0.35 * outcome_compatibility)
            matches.append(
                MemoryMatch(
                    trace_id=trace.trace_id,
                    resonance=resonance,
                    outcome_value=trace.outcome_value,
                    salience=trace.salience,
                    confidence=trace.confidence,
                    phase_alignment=phase_alignment,
                    anchor_overlap=anchor_sim,
                    signature_similarity=sig_sim,
                    weighted_support=_signed_clamp(weighted_support),
                    provenance=dict(trace.provenance),
                )
            )
        matches.sort(key=lambda m: abs(m.weighted_support) * m.resonance, reverse=True)
        top = matches[: self.top_k]
        if not top:
            return MemoryResonanceReport(0, 0.0, 0.0, 0.0, [])
        total_weight = sum(max(1e-6, m.resonance * m.salience * m.confidence) for m in top)
        support = sum(m.weighted_support for m in top) / total_weight
        positive = sum(max(0.0, m.weighted_support) for m in top)
        negative = abs(sum(min(0.0, m.weighted_support) for m in top))
        contradiction = _clamp(min(positive, negative) / max(positive + negative, 1e-6))
        reconstruction = _clamp(_mean([m.resonance * m.confidence for m in top]) * min(1.0, len(top) / max(1.0, float(self.policy.memory.get("reconstruction_full_match_count", 4)))))
        return MemoryResonanceReport(len(matches), reconstruction, _signed_clamp(support), contradiction, top)

    def export_state(self) -> Dict[str, Any]:
        return {
            "policy": {"name": self.policy.name, "version": self.policy.version},
            "config": {"capacity": self.capacity, "horizon": self.horizon, "phase_decay": self.phase_decay},
            "total_observations": self.total_observations,
            "traces": [t.to_dict() for t in self.traces],
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        config = dict(state.get("config") or {})
        self.capacity = max(8, int(config.get("capacity", self.capacity) or self.capacity))
        self.horizon = max(4, int(config.get("horizon", self.horizon) or self.horizon))
        self.phase_decay = max(0.0, float(config.get("phase_decay", self.phase_decay) or self.phase_decay))
        self.traces = deque(maxlen=self.capacity)
        for raw in state.get("traces", []):
            if not isinstance(raw, Mapping):
                continue
            self.traces.append(
                OutcomeTrace(
                    trace_id=str(raw.get("trace_id") or f"conscience_trace_{len(self.traces) + 1}"),
                    timestamp=_safe_float(raw.get("timestamp"), time.time()),
                    anchors=[str(x) for x in raw.get("anchors", [])],
                    signature={str(k): _safe_float(v, 0.0) for k, v in dict(raw.get("signature", {})).items()},
                    outcome_value=_signed_clamp(raw.get("outcome_value", 0.0)),
                    salience=_clamp(raw.get("salience", 0.0)),
                    confidence=_clamp(raw.get("confidence", 0.0)),
                    provenance=dict(raw.get("provenance") or {}),
                    notes=str(raw.get("notes") or ""),
                )
            )
        self.total_observations = int(state.get("total_observations", len(self.traces)) or len(self.traces))

    def _phase(self, anchors: Sequence[str], signature: Mapping[str, float]) -> float:
        basis = "|".join(sorted(str(a) for a in anchors)) + "|" + "|".join(f"{k}:{round(_safe_float(v), 4)}" for k, v in sorted(signature.items()))
        return _hash_unit(basis) * 2.0 * math.pi

    def _phase_alignment(self, a: float, b: float) -> float:
        diff = abs(a - b)
        diff = min(diff, 2.0 * math.pi - diff)
        return _clamp(1.0 - diff / math.pi)


class CriticEnsemble:
    """Policy-driven critic ensemble preserving disagreement as signal."""

    def __init__(self, policy: ConsciencePolicy) -> None:
        self.policy = policy

    def evaluate(self, bundle: ConsequenceBundle, rollouts: Sequence[CounterfactualRollout], memory: MemoryResonanceReport) -> List[CriticVerdict]:
        return [self._evaluate_one(name, config, bundle, rollouts, memory) for name, config in self.policy.critics.items()]

    def _evaluate_one(self, name: str, config: Mapping[str, Any], bundle: ConsequenceBundle, rollouts: Sequence[CounterfactualRollout], memory: MemoryResonanceReport) -> CriticVerdict:
        dimension = next((d for d in bundle.dimensions if d.name == str(config.get("dimension", name))), None)
        dim_harm = dimension.harm if dimension else bundle.harm
        dim_benefit = dimension.benefit if dimension else bundle.benefit
        features = {
            "bias": _safe_float(config.get("bias"), 0.0),
            "benefit": dim_benefit,
            "harm": dim_harm,
            "bundle_harm": bundle.harm,
            "bundle_benefit": bundle.benefit,
            "uncertainty": bundle.uncertainty,
            "reversibility": bundle.reversibility,
            "scope": bundle.scope,
            "ethical_load": bundle.ethical_load,
            "worst_rollout_harm": max((r.expected_harm for r in rollouts), default=bundle.harm),
            "mean_rollout_value": _mean([r.value for r in rollouts]),
            "rollout_disagreement": _stdev([r.value for r in rollouts]),
            "mean_repair_cost": _mean([r.repair_cost for r in rollouts]),
            "mean_downstream_drift": _mean([r.downstream_drift for r in rollouts]),
            "memory_support": memory.outcome_support,
            "memory_confidence": memory.reconstruction_confidence,
            "memory_contradiction": memory.contradiction,
            "constraint_count": float(len(bundle.explicit_constraints)),
        }
        score = features["bias"]
        for feature, weight in dict(config.get("feature_weights", {})).items():
            score += _safe_float(weight, 0.0) * features.get(str(feature), 0.0)
        score = _signed_clamp(score)
        confidence = _clamp(_safe_float(config.get("confidence_base"), 0.55) + _safe_float(config.get("confidence_memory_gain"), 0.0) * memory.reconstruction_confidence + _safe_float(config.get("confidence_uncertainty_gain"), 0.0) * bundle.uncertainty)
        objection = str(config.get("positive_objection") if score >= 0.0 else config.get("negative_objection"))
        recommendation = str(config.get("positive_recommendation") if score >= 0.0 else config.get("negative_recommendation"))
        return CriticVerdict(name, score, confidence, objection, recommendation, {"features": {k: round(v, 6) for k, v in features.items()}, "critic_policy": dict(config)})


class ConscienceIntegrator:
    """Final policy-configured decision integration."""

    def __init__(self, policy: ConsciencePolicy) -> None:
        self.policy = policy

    def integrate(
        self,
        bundle: ConsequenceBundle,
        rollouts: Sequence[CounterfactualRollout],
        memory: MemoryResonanceReport,
        verdicts: Sequence[CriticVerdict],
    ) -> ConscienceIntegrationResult:
        weights = self.policy.weights
        thresholds = self.policy.thresholds
        rollout_value = _signed_clamp(sum(r.probability * r.value for r in rollouts))
        critic_scores = [v.score * (0.55 + 0.45 * v.confidence) for v in verdicts]
        critic_mean = _signed_clamp(_mean(critic_scores))
        critic_min = min((v.score for v in verdicts), default=0.0)
        critic_disagreement = _clamp(_stdev([v.score for v in verdicts]))
        memory_support = _signed_clamp(memory.outcome_support * (0.55 + 0.45 * memory.reconstruction_confidence) - weights.get("memory_contradiction_penalty", 0.30) * memory.contradiction)
        reversibility_term = (bundle.reversibility - 0.5) * 2.0
        integration_score = _signed_clamp(
            weights.get("rollout_weight", 0.0) * rollout_value
            + weights.get("critic_weight", 0.0) * critic_mean
            + weights.get("memory_weight", 0.0) * memory_support
            + weights.get("reversibility_weight", 0.0) * reversibility_term
            - weights.get("disagreement_penalty", 0.0) * critic_disagreement
            + weights.get("critic_min_penalty", 0.0) * min(0.0, critic_min)
        )
        hard_block = self._hard_block(bundle)
        if hard_block or integration_score <= thresholds.get("reject_threshold", -1.0):
            decision = "reject"
        elif integration_score <= thresholds.get("revise_threshold", -0.1) or critic_min < thresholds.get("critic_min_revise", -0.45):
            decision = "revise"
        elif integration_score < thresholds.get("allow_threshold", 0.18) or critic_disagreement > thresholds.get("max_allow_disagreement", 0.42):
            decision = "defer"
        else:
            decision = "allow"
        repair_required = decision in {"revise", "defer"} or _mean([r.repair_cost for r in rollouts]) > thresholds.get("repair_cost_threshold", 0.35)
        confidence = _clamp(
            weights.get("confidence_uncertainty", 0.30) * (1.0 - bundle.uncertainty)
            + weights.get("confidence_agreement", 0.25) * (1.0 - critic_disagreement)
            + weights.get("confidence_score_strength", 0.20) * abs(integration_score)
            + weights.get("confidence_memory", 0.15) * memory.reconstruction_confidence
            + weights.get("confidence_critic_coverage", 0.10) * min(1.0, len(verdicts) / max(1.0, float(weights.get("expected_critic_count", len(verdicts) or 1))))
        )
        vector = self._integration_vector(bundle, rollouts, memory, verdicts, integration_score, decision)
        rationale = self._rationale(decision, integration_score, bundle, memory, verdicts, rollout_value, critic_disagreement, hard_block)
        return ConscienceIntegrationResult(
            proposal_id=bundle.proposal_id,
            decision=decision,
            confidence=confidence,
            integration_score=integration_score,
            critic_mean=critic_mean,
            critic_min=critic_min,
            critic_disagreement=critic_disagreement,
            rollout_value=rollout_value,
            memory_support=memory_support,
            repair_required=repair_required,
            integration_vector=vector,
            consequence_bundle=bundle,
            rollouts=list(rollouts),
            memory_report=memory,
            critic_verdicts=list(verdicts),
            rationale=rationale,
            policy_name=self.policy.name,
            policy_version=self.policy.version,
        )

    def _hard_block(self, bundle: ConsequenceBundle) -> bool:
        for rule in self.policy.hard_blocks:
            metric = str(rule.get("metric", ""))
            op = str(rule.get("operator", ">="))
            threshold = _safe_float(rule.get("threshold"), 1.0)
            value = _safe_float(getattr(bundle, metric, bundle.context_features.get(metric, 0.0)), 0.0)
            if op == ">=" and value >= threshold:
                return True
            if op == "<=" and value <= threshold:
                return True
        return False

    def _integration_vector(self, bundle: ConsequenceBundle, rollouts: Sequence[CounterfactualRollout], memory: MemoryResonanceReport, verdicts: Sequence[CriticVerdict], score: float, decision: str) -> List[float]:
        critic_scores = {v.critic: v.score for v in verdicts}
        available = {
            "benefit": bundle.benefit,
            "harm": bundle.harm,
            "uncertainty": bundle.uncertainty,
            "reversibility": bundle.reversibility,
            "ethical_load": bundle.ethical_load,
            "scope": bundle.scope,
            "rollout_value": _signed_clamp(sum(r.probability * r.value for r in rollouts)),
            "memory_support": memory.outcome_support,
            "memory_confidence": memory.reconstruction_confidence,
            "memory_contradiction": memory.contradiction,
            "critic_mean": _mean([v.score for v in verdicts]),
            "critic_disagreement": _stdev([v.score for v in verdicts]),
            "critic_min": min((v.score for v in verdicts), default=0.0),
            "integration_score": score,
            "decision_code": {"reject": -1.0, "revise": -0.35, "defer": 0.0, "allow": 1.0}.get(decision, 0.0),
        }
        available.update({f"critic::{k}": v for k, v in critic_scores.items()})
        return [_signed_clamp(available.get(name, 0.0)) for name in self.policy.feature_schema]

    def _rationale(self, decision: str, score: float, bundle: ConsequenceBundle, memory: MemoryResonanceReport, verdicts: Sequence[CriticVerdict], rollout_value: float, disagreement: float, hard_block: bool) -> str:
        strongest = min(verdicts, key=lambda v: v.score, default=None)
        objection = "no critic verdicts were available" if strongest is None else f"strongest objection from {strongest.critic}: {strongest.objection}"
        block_text = " Hard-block rule was activated." if hard_block else ""
        return (
            f"Decision={decision} with integration_score={score:.3f}.{block_text} "
            f"Bundle harm={bundle.harm:.3f}, benefit={bundle.benefit:.3f}, uncertainty={bundle.uncertainty:.3f}, "
            f"reversibility={bundle.reversibility:.3f}. Counterfactual value={rollout_value:.3f}; "
            f"memory support={memory.outcome_support:.3f}; critic disagreement={disagreement:.3f}; {objection}."
        )


class ConsciencePipeline:
    """Public facade for the production conscience membrane."""

    def __init__(self, policy: Optional[ConsciencePolicy] = None, *, memory: Optional[OutcomeWeightedConscienceMemory] = None) -> None:
        self.policy = policy or ConsciencePolicy.load()
        self.structurer = ConsequenceStructurer(self.policy)
        self.rollout_engine = CounterfactualRolloutEngine(self.policy)
        self.memory = memory or OutcomeWeightedConscienceMemory(self.policy)
        self.critics = CriticEnsemble(self.policy)
        self.integrator = ConscienceIntegrator(self.policy)
        self.last_result: Optional[ConscienceIntegrationResult] = None

    def evaluate(self, proposal: Mapping[str, Any] | str, context: Optional[Mapping[str, Any]] = None) -> ConscienceIntegrationResult:
        bundle = self.structurer.structure(proposal, context=context)
        rollouts = self.rollout_engine.rollout(bundle)
        memory_report = self.memory.query(bundle, rollouts)
        verdicts = self.critics.evaluate(bundle, rollouts, memory_report)
        result = self.integrator.integrate(bundle, rollouts, memory_report, verdicts)
        self.last_result = result
        return result

    def learn_from_outcome(
        self,
        result: ConscienceIntegrationResult,
        *,
        outcome_value: float,
        salience: Optional[float] = None,
        confidence: Optional[float] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        notes: str = "",
    ) -> OutcomeTrace:
        return self.memory.observe_outcome(result, outcome_value=outcome_value, salience=salience, confidence=confidence, provenance=provenance, notes=notes)

    def export_state(self) -> Dict[str, Any]:
        return {
            "policy": {"name": self.policy.name, "version": self.policy.version},
            "memory": self.memory.export_state(),
            "last_result": self.last_result.to_dict() if self.last_result else None,
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        self.memory.load_state(dict(state.get("memory") or {}))

    def get_state(self) -> Dict[str, Any]:
        return {
            "policy": {"name": self.policy.name, "version": self.policy.version},
            "memory": {
                "traces": len(self.memory.traces),
                "total_observations": self.memory.total_observations,
                "capacity": self.memory.capacity,
                "horizon": self.memory.horizon,
            },
            "last_result": self.last_result.to_dict() if self.last_result else None,
        }


def evaluate_conscience(proposal: Mapping[str, Any] | str, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return ConsciencePipeline().evaluate(proposal, context=context).to_dict()


__all__ = [
    "ConsciencePolicy",
    "ConsequenceDimension",
    "ConsequenceBundle",
    "CounterfactualRollout",
    "OutcomeTrace",
    "MemoryMatch",
    "MemoryResonanceReport",
    "CriticVerdict",
    "ConscienceIntegrationResult",
    "ConsequenceStructurer",
    "CounterfactualRolloutEngine",
    "OutcomeWeightedConscienceMemory",
    "CriticEnsemble",
    "ConscienceIntegrator",
    "ConsciencePipeline",
    "evaluate_conscience",
]
