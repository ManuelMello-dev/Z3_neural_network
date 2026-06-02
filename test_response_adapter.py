"""
Standalone tests for the Z³ response adapter.

Run with:
    python test_response_adapter.py
"""
from __future__ import annotations

from response_adapter import build_z3_expression

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: object = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} -- {detail}")


print("=" * 60)
print("TESTING Z³ RESPONSE ADAPTER")
print("=" * 60)

sample_result = {
    "world_model": {
        "novelty": 0.42,
        "total_loss": 0.18,
        "latent_norm": 0.51,
    },
    "memory": {
        "accessible_rings": 3,
        "reconstruction_confidence": 0.64,
        "salience": 0.72,
        "top_matches": [
            {
                "domain": "language:chat",
                "resonance": 0.71,
            }
        ],
    },
    "z3": {
        "metrics": {
            "mean_coherence": 0.78,
            "mean_novelty": 0.44,
            "mean_gate": 0.36,
            "z3_delta_norm": 0.012345,
        }
    },
}

expression = build_z3_expression(
    message="Are we building the mouth now?",
    observation={"entity_id": "CHAT_1", "domain_prefix": "language"},
    integrated_result=sample_result,
)
payload = expression.to_dict()

check("Question intent is detected", expression.intent == "question", expression.intent)
check("Coherent discovery regime is selected", expression.regime == "coherent_discovery", expression.regime)
check("Response is not the old passive acknowledgement", expression.response != "Message observed by the Z³ language runtime.", expression.response)
check("Response references measured memory rings", "3 accessible ring" in expression.response, expression.response)
check("Response references world-model read", "World-model read" in expression.response, expression.response)
check("Confidence is bounded", 0.0 <= expression.confidence <= 1.0, expression.confidence)
check("Payload exposes expression fields", all(key in payload for key in ("response", "regime", "confidence", "trace")), payload.keys())
check("Trace exposes metrics used", payload["trace"]["metrics_used"]["mean_coherence"] == 0.78, payload["trace"])

low_signal = build_z3_expression(
    message="build the actuator bridge",
    observation={"entity_id": "CHAT_2"},
    integrated_result={
        "world_model": {"novelty": 0.10, "total_loss": 0.90, "latent_norm": 0.10},
        "memory": {"accessible_rings": 0, "reconstruction_confidence": 0.0, "salience": 0.31},
        "z3": {"metrics": {"mean_coherence": 0.20, "mean_novelty": 0.10, "mean_gate": 0.01, "z3_delta_norm": 0.0}},
    },
)

check("Directive intent is detected", low_signal.intent == "directive", low_signal.intent)
check("Low gate regime is selected", low_signal.regime == "low_gate_recalibration", low_signal.regime)
check("No-memory path is explicit", "No strong prior resonance" in low_signal.response, low_signal.response)
check("Low confidence remains bounded", 0.0 <= low_signal.confidence <= 1.0, low_signal.confidence)

training_expression = build_z3_expression(
    message="Train on this language trace.",
    observation={"entity_id": "CHAT_3"},
    integrated_result=sample_result,
    language_training={"trained": True, "trained_steps": 4},
)
check("Training mutation is surfaced", "trained_steps=4" in training_expression.response, training_expression.response)

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL}")
raise SystemExit(1 if FAIL else 0)
