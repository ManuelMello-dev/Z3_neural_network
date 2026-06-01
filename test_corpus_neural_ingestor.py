"""Smoke tests for the canonical Z³ corpus neural ingestor."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from corpus_neural_ingestor import Z3CorpusIngestionConfig, Z3CorpusNeuralIngestor

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


class FakeModel:
    pass


class FakeOptimizer:
    def __init__(self) -> None:
        self.zeroed = False
        self.loaded = False

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zeroed = True

    def state_dict(self):
        return {"fake_optimizer": True}

    def load_state_dict(self, state):
        self.loaded = bool(state.get("fake_optimizer"))


print("=" * 60)
print("TESTING CANONICAL Z³ CORPUS NEURAL INGESTOR")
print("=" * 60)

os.environ["Z3_CORPUS_TRAIN_BATCH_SIZE"] = "not-an-int"
os.environ["Z3_CORPUS_LEARNING_RATE"] = "not-a-float"
safe_config = Z3CorpusIngestionConfig.from_env()
check("Invalid integer env falls back safely", safe_config.batch_size >= 1, safe_config)
check("Invalid float env falls back safely", safe_config.learning_rate > 0.0, safe_config)
os.environ.pop("Z3_CORPUS_TRAIN_BATCH_SIZE", None)
os.environ.pop("Z3_CORPUS_LEARNING_RATE", None)

train_calls = []


def successful_trainer(model, optimizer, texts, **kwargs):
    train_calls.append((list(texts), dict(kwargs)))
    return {"window_loss": 0.25, "texts": len(texts)}


config = Z3CorpusIngestionConfig(
    enabled=True,
    batch_size=2,
    min_words=2,
    max_buffer_texts=8,
    checkpoint_every_steps=0,
    checkpoint_path=str(Path(tempfile.gettempdir()) / "z3_test_corpus_ingestor.pt"),
)
ingestor = Z3CorpusNeuralIngestor(
    config,
    model=FakeModel(),
    optimizer=FakeOptimizer(),
    trainer=successful_trainer,
)
first = ingestor.observe_text("alpha beta gamma", {"source": "unit-test-a"})
second = ingestor.observe_text("delta epsilon zeta", {"source": "unit-test-b"})
check("First text buffers without premature training", first.get("buffer_size") == 1, first)
check("Second text triggers a train batch", second.get("trained_steps") == 1, second)
check("Successful training drains the buffer", second.get("buffer_size") == 0, second)
check("Metrics are retained for diagnostics", second.get("last_metrics", {}).get("window_loss") == 0.25, second)
check("Provenance metadata is retained", second.get("last_provenance", [{}])[-1].get("source") == "unit-test-b", second)

failing_optimizer = FakeOptimizer()


def failing_trainer(model, optimizer, texts, **kwargs):
    raise RuntimeError("simulated corpus failure")


failing = Z3CorpusNeuralIngestor(
    config,
    model=FakeModel(),
    optimizer=failing_optimizer,
    trainer=failing_trainer,
)
failing.observe_text("one two three", {"source": "failure-a"})
failure_snapshot = failing.observe_text("four five six", {"source": "failure-b"})
check("Failed training increments failure counter", failure_snapshot.get("failed_train_attempts") == 1, failure_snapshot)
check("Failed batch is restored to buffer", failure_snapshot.get("buffer_size") == 2, failure_snapshot)
check("Optimizer gradients are cleared on failure", failing_optimizer.zeroed is True, failure_snapshot)
check("Failure reason is exposed", "training failed" in failure_snapshot.get("last_error", ""), failure_snapshot)

try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:
    skip("Ingestor checkpoint metadata restoration", str(exc))
else:
    checkpoint_path = Path(tempfile.gettempdir()) / "z3_test_corpus_ingestor_restore.pt"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    persisted_config = Z3CorpusIngestionConfig(
        enabled=True,
        batch_size=2,
        min_words=2,
        max_buffer_texts=8,
        checkpoint_every_steps=0,
        checkpoint_path=str(checkpoint_path),
    )
    persisted = Z3CorpusNeuralIngestor(
        persisted_config,
        model=FakeModel(),
        optimizer=FakeOptimizer(),
        trainer=successful_trainer,
    )
    persisted.train_texts_now(["persist alpha beta", "persist gamma delta"])
    saved = persisted.save_checkpoint()
    restored_optimizer = FakeOptimizer()
    restored = Z3CorpusNeuralIngestor(
        persisted_config,
        model=FakeModel(),
        optimizer=restored_optimizer,
        trainer=successful_trainer,
    )
    loaded = restored.load_checkpoint(load_model=False)
    restored_snapshot = restored.snapshot()
    check("Ingestor checkpoint saves", saved is True, restored_snapshot)
    check("Ingestor checkpoint loads", loaded is True, restored_snapshot)
    check("Checkpoint restores trained step count", restored_snapshot.get("trained_steps") == 1, restored_snapshot)
    check("Checkpoint restores optimizer metadata", restored_optimizer.loaded is True, restored_snapshot)

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
raise SystemExit(1 if FAIL else 0)
