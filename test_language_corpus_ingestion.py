"""Smoke tests for direct Z³ language corpus ingestion."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

PASS = 0
FAIL = 0
SKIP = 0


def check(name: str, condition: bool, detail="") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


def skip(name: str, detail="") -> None:
    global SKIP
    SKIP += 1
    print(f"  - {name} skipped — {detail}")


print("=" * 60)
print("TESTING DIRECT Z³ LANGUAGE CORPUS INGESTION")
print("=" * 60)

os.environ["LANGUAGE_TRAINING_TEXT"] = (
    "Z3 now receives an actual corpus ingestion stream. "
    "The stream preserves raw language text for the neural sequence trainer. "
    "Corpus windows train the global observer and the local z prime agents. "
    "This test verifies direct text sequence training rather than helper-only wiring."
)
os.environ["LANGUAGE_TRAINING_BATCH_SIZE"] = "4"
os.environ["LANGUAGE_TRAINING_MIN_WORDS"] = "3"
os.environ["DISABLE_REMOTE_CORPUS_STREAM"] = "1"
os.environ["Z3_RUNTIME_LANGUAGE_TRAIN"] = "true"
os.environ["Z3_LANGUAGE_WINDOW_SIZE"] = "6"
os.environ["Z3_LANGUAGE_STRIDE"] = "3"
os.environ["Z3_LANGUAGE_TRUNCATION_STEPS"] = "2"

from language_stream import LanguageStream

stream = LanguageStream()
status = stream.ensure_loaded()
batch = stream.fetch_batch(batch_size=4)
check("Language stream loads inline corpus", status.get("loaded") is True, status)
check("Language stream emits corpus observations", batch.get("count", 0) > 0, batch)
check("Corpus observations carry text", all(obs.get("text") for obs in batch.get("observations", [])), batch)

try:
    import torch
    from Z3_neural_dynamics import Z3NeuralDynamics
    from z3_language_training import build_language_embedding_stream, train_z3_on_language_window
except ModuleNotFoundError as exc:
    skip("Direct neural language training", str(exc))
else:
    model = Z3NeuralDynamics()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    texts = [obs["text"] for obs in batch["observations"]]
    embedding_stream = build_language_embedding_stream(texts, input_dim=model.config.input_dim, window_size=6, stride=3)
    before = model.z3_state.detach().clone()
    metrics = train_z3_on_language_window(
        model,
        optimizer,
        texts,
        truncation_steps=2,
        window_size=6,
        stride=3,
        commit_recurrent_state=True,
        add_noise=False,
    )
    after = model.z3_state.detach().clone()
    check("Language embeddings match Z³ input dimension", embedding_stream.shape[-1] == model.config.input_dim, embedding_stream.shape)
    check("Direct language training returns metrics", isinstance(metrics, dict) and "window_loss" in metrics, metrics)
    check("Direct corpus training advances Z³ state", bool(torch.norm(after - before) > 0), metrics)

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
raise SystemExit(1 if FAIL else 0)
