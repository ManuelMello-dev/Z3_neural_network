"""Smoke tests for layered Z³ curriculum ingestion.

Run with:
    python test_curriculum_ingestion.py
"""
from __future__ import annotations

import json
import os

PASS = 0
FAIL = 0
SKIP = 0


def check(name: str, condition: bool, detail: object = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} -- {detail}")


def skip(name: str, detail: object = "") -> None:
    global SKIP
    SKIP += 1
    print(f"  - {name} -- {detail}")


print("=" * 60)
print("TESTING LAYERED Z³ CURRICULUM INGESTION")
print("=" * 60)

rows = [
    {
        "kind": "language",
        "text": "Recursive language grounding preserves symbolic rhythm across observer updates.",
        "source": "test_language_curriculum",
    },
    {
        "kind": "dialogue",
        "speaker": "operator",
        "message": "Track the state, remember the callback, and answer from continuity.",
        "source": "test_dialogue_curriculum",
    },
    {
        "kind": "contradiction",
        "premise": "The local agent is coherent with the global observer.",
        "hypothesis": "The local agent is completely unrelated to the observer.",
        "label": "contradiction",
        "source": "test_contradiction_curriculum",
    },
    {
        "kind": "event",
        "entity_id": "MARKET_REGIME_A",
        "value": 101.5,
        "secondary_value": 0.42,
        "concept": "temporal_event",
        "source": "test_event_curriculum",
    },
    {
        "kind": "anomaly",
        "entity_id": "ANOMALY_SPIKE_A",
        "value": 9.7,
        "secondary_value": 3.2,
        "novelty_target": 0.95,
        "source": "test_anomaly_curriculum",
    },
    {
        "kind": "identity",
        "text": "Z cubed is the persistent observer while Z prime agents test local hypotheses.",
        "salience": 0.92,
        "source": "test_identity_curriculum",
    },
]

os.environ["Z3_CURRICULUM_INLINE_JSONL"] = "\n".join(json.dumps(row) for row in rows)
os.environ["Z3_CURRICULUM_REMOTE_ENABLED"] = "false"
os.environ["Z3_CURRICULUM_SOURCES"] = "language,dialogue,contradiction,event,anomaly,identity"
os.environ.setdefault("LANGUAGE_TRAINING_TEXT", "Z3 language smoke text keeps the runtime import path stable.")

from curriculum_stream import CurriculumStream

stream = CurriculumStream()
status = stream.ensure_loaded()
check("Curriculum stream loads inline JSONL", status.get("loaded") and status.get("rows_loaded") == len(rows), status)

batch = stream.fetch_batch(batch_size=6)
observations = batch.get("observations", [])
kinds = {obs.get("curriculum_kind") for obs in observations}
check("Curriculum stream emits all requested observations", batch.get("count") == 6, batch)
check("Curriculum observations cover layered sources", kinds == {"language", "dialogue", "contradiction", "event", "anomaly", "identity"}, kinds)
check("Curriculum observations expose common Z³ schema", all("entity_id" in obs and "value" in obs and "secondary_value" in obs and "curriculum_stage" in obs for obs in observations), observations)
check("Contradiction curriculum carries contradiction target", any(obs.get("curriculum_kind") == "contradiction" and obs.get("contradiction_target", 0.0) > 0.0 for obs in observations), observations)
check("Anomaly curriculum carries high novelty target", any(obs.get("curriculum_kind") == "anomaly" and obs.get("novelty_target", 0.0) >= 0.9 for obs in observations), observations)

try:
    import torch  # noqa: F401
    import main
except ModuleNotFoundError as exc:
    skip("Integrated neural curriculum ingestion", str(exc))
else:
    summary = main._ingest_curriculum_batch_for_runtime(batch_size=3, source=None, train=False, learning_rate=1e-3)
    check("Integrated curriculum ingestion returns events", summary.get("count") == 3, summary)
    check("Integrated curriculum summary exposes kind counts", bool(summary.get("summary", {}).get("curriculum_kinds")), summary)
    check("Integrated curriculum updates memory/world flow", summary.get("summary", {}).get("events_ingested") == 3, summary)

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
raise SystemExit(1 if FAIL else 0)
