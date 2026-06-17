"""Smoke test for Z³ end-to-end language decoder."""
import sys
sys.path.insert(0, ".")

import torch
from Z3_neural_dynamics import Z3NeuralDynamics, Z3Config
from z3_language_decoder import (
    CharTokenizer, get_tokenizer, get_decoder,
    train_language_step, generate, save_decoder, load_decoder,
    BOS_ID, EOS_ID,
)

print("=== Z³ Language Decoder Smoke Test ===\n")

# 1. Tokenizer
tok = get_tokenizer()
print(f"Vocab size: {tok.vocab_size}")
ids = tok.encode("Hello, world!")
decoded = tok.decode(ids)
assert "Hello, world!" in decoded, f"decode failed: {repr(decoded)}"
print(f"Tokenizer OK — encode/decode roundtrip: {repr(decoded)}")

# 2. Decoder init
cfg = Z3Config()
model = Z3NeuralDynamics(cfg)
dec = get_decoder(
    input_dim=cfg.input_dim,
    evidence_dim=cfg.evidence_dim,
    state_dim=cfg.state_dim,
    context_dim=cfg.context_dim,
)
z3_params = sum(p.numel() for p in model.parameters())
dec_params = sum(p.numel() for p in dec.parameters())
print(f"Z³ params: {z3_params:,}")
print(f"Decoder params: {dec_params:,}")
print(f"Total trainable: {z3_params + dec_params:,}")

# 3. Training step
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(dec.parameters()), lr=1e-3
)
metrics = train_language_step(
    model, dec, optimizer,
    "The mind is not the brain. It is a field that the brain tunes into.",
    max_seq_len=64,
)
print(f"\nTrain step metrics: {metrics}")
assert metrics.get("trained") is True, f"Training failed: {metrics}"
print("Training step OK")

# 4. Generation (untrained — output will be noise, but must not crash)
text = generate(model, dec, prompt="The", max_new_tokens=40, temperature=1.0)
print(f"\nGenerated (untrained, noise expected): {repr(text)}")
assert isinstance(text, str), "generate() must return a string"
print("Generation OK")

# 5. Save/load roundtrip
import tempfile, os
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
    tmp_path = tmp.name
dec.save(tmp_path)
dec2 = get_decoder.__wrapped__(  # bypass singleton for test
    input_dim=cfg.input_dim,
    evidence_dim=cfg.evidence_dim,
    state_dim=cfg.state_dim,
    context_dim=cfg.context_dim,
) if hasattr(get_decoder, "__wrapped__") else type(dec)(
    input_dim=cfg.input_dim,
    evidence_dim=cfg.evidence_dim,
    state_dim=cfg.state_dim,
    context_dim=cfg.context_dim,
    vocab_size=tok.vocab_size,
)
dec2.load(tmp_path)
os.unlink(tmp_path)
print("Save/load roundtrip OK")

print("\n=== ALL TESTS PASSED ===")
