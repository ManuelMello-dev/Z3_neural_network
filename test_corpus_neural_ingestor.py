"""Smoke tests for the canonical Z³ corpus neural ingestor."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from corpus_neural_ingestor import CHECKPOINT_VERSION, Z3CorpusIngestionConfig, Z3CorpusNeuralIngestor

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
check("Health is ready after successful training", second.get("health_state") == "ready", second)

hardened_config = Z3CorpusIngestionConfig(
    enabled=True,
    batch_size=4,
    min_words=2,
    max_buffer_texts=2,
    checkpoint_every_steps=0,
    max_text_bytes=16,
    checkpoint_path=str(Path(tempfile.gettempdir()) / "z3_test_corpus_ingestor_hardened.pt"),
)
hardened = Z3CorpusNeuralIngestor(
    hardened_config,
    model=FakeModel(),
    optimizer=FakeOptimizer(),
    trainer=successful_trainer,
)
hardened.observe_text("short", {"source": "too-short"})
hardened.observe_text("this segment is definitely too large", {"source": "oversized"})
hardened.observe_text("aa bb", {"source": "dedup-a"})
hardened.observe_text("aa bb", {"source": "dedup-b"})
hardened_snapshot = hardened.snapshot()
check("Too-short text is rejected with reason", hardened_snapshot.get("dropped_reasons", {}).get("too_few_words") == 1, hardened_snapshot)
check("Oversized text is rejected with reason", hardened_snapshot.get("dropped_reasons", {}).get("oversized_text") == 1, hardened_snapshot)
check("Duplicate text is rejected with reason", hardened_snapshot.get("dropped_reasons", {}).get("duplicate_text") == 1, hardened_snapshot)
check("Accepted text gets normalized hash provenance", hardened_snapshot.get("buffer_size") == 1, hardened_snapshot)

backlogged_config = Z3CorpusIngestionConfig(
    enabled=True,
    batch_size=10,
    min_words=2,
    max_buffer_texts=2,
    checkpoint_every_steps=0,
    backlog_warning_ratio=0.5,
    checkpoint_path=str(Path(tempfile.gettempdir()) / "z3_test_corpus_ingestor_backlog.pt"),
)
backlogged = Z3CorpusNeuralIngestor(
    backlogged_config,
    model=FakeModel(),
    optimizer=FakeOptimizer(),
    trainer=successful_trainer,
)
backlogged.observe_text("first accepted", {"source": "backlog"})
check("Backlog pressure changes health state", backlogged.snapshot().get("health_state") == "backlogged", backlogged.snapshot())

failing_optimizer = FakeOptimizer()


def failing_trainer(model, optimizer, texts, **kwargs):
    raise RuntimeError("simulated corpus failure")


failing_config = Z3CorpusIngestionConfig(
    enabled=True,
    batch_size=2,
    min_words=2,
    max_buffer_texts=8,
    checkpoint_every_steps=0,
    circuit_breaker_failure_threshold=1,
    circuit_breaker_cooldown_seconds=60,
    checkpoint_path=str(Path(tempfile.gettempdir()) / "z3_test_corpus_ingestor_failure.pt"),
)
failing = Z3CorpusNeuralIngestor(
    failing_config,
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
check("Circuit breaker opens after threshold", failure_snapshot.get("health_state") == "circuit_open", failure_snapshot)
blocked = failing.train_texts_now(["seven eight nine"])
check("Circuit breaker blocks immediate training", blocked.get("reason") == "circuit_breaker_open", blocked)
resumed = failing.resume_training()
check("Circuit breaker can be resumed", resumed.get("circuit_open") is False, resumed)

try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:
    skip("Ingestor checkpoint metadata restoration", str(exc))
else:
    checkpoint_path = Path(tempfile.gettempdir()) / "z3_test_corpus_ingestor_restore.pt"
    previous_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".previous")
    for path in (checkpoint_path, previous_path):
        if path.exists():
            path.unlink()
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
    persisted.train_texts_now(["persist epsilon zeta", "persist eta theta"])
    saved_again = persisted.save_checkpoint()
    restored_optimizer = FakeOptimizer()
    restored = Z3CorpusNeuralIngestor(
        persisted_config,
        model=FakeModel(),
        optimizer=restored_optimizer,
        trainer=successful_trainer,
    )
    loaded = restored.load_checkpoint(load_model=False)
    restored_snapshot = restored.snapshot()
    check("Ingestor checkpoint saves", saved is True and saved_again is True, restored_snapshot)
    check("Ingestor checkpoint keeps previous copy", previous_path.exists() is True, restored_snapshot)
    check("Ingestor checkpoint loads", loaded is True, restored_snapshot)
    check("Checkpoint restores trained step count", restored_snapshot.get("trained_steps") >= 1, restored_snapshot)
    check("Checkpoint restores optimizer metadata", restored_optimizer.loaded is True, restored_snapshot)
    check("Checkpoint version is exposed", restored_snapshot.get("last_checkpoint_version") == CHECKPOINT_VERSION, restored_snapshot)

    checkpoint_path.write_bytes(b"not a torch checkpoint")
    fallback_optimizer = FakeOptimizer()
    fallback = Z3CorpusNeuralIngestor(
        persisted_config,
        model=FakeModel(),
        optimizer=fallback_optimizer,
        trainer=successful_trainer,
    )
    fallback_loaded = fallback.load_checkpoint(load_model=False)
    fallback_snapshot = fallback.snapshot()
    quarantined = list(checkpoint_path.parent.glob(checkpoint_path.name + ".corrupt.*"))
    check("Corrupt checkpoint falls back to previous", fallback_loaded is True, fallback_snapshot)
    check("Corrupt checkpoint is quarantined", bool(quarantined), quarantined)

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
raise SystemExit(1 if FAIL else 0)
