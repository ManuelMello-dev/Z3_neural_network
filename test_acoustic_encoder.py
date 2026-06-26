"""Smoke test for Z³ Acoustic Input Encoder."""
import sys
sys.path.insert(0, ".")

import torch
from z3_acoustic_encoder import get_acoustic_encoder

print("=== Z³ Acoustic Input Encoder Smoke Test ===\n")

# 1. Initialize encoder
encoder = get_acoustic_encoder(input_dim=16)
print(f"Encoder initialized: {sum(p.numel() for p in encoder.parameters()):,} parameters.")

# 2. Generate fake audio (100ms at 22050 Hz = 2205 samples)
batch_size = 2
n_samples = 2205
fake_audio = torch.randn(batch_size, n_samples)
print(f"Input audio shape: {fake_audio.shape}")

# 3. Forward pass
z3_input = encoder(fake_audio)
assert z3_input.shape == (batch_size, 16)
assert (z3_input >= -1).all() and (z3_input <= 1).all()
print(f"Z³ input vector shape: {z3_input.shape} ✓")
print(f"Sample vector: {z3_input[0].detach().numpy()}")

# 4. Gradient check
fake_audio.requires_grad = True
z3_input = encoder(fake_audio)
z3_input.sum().backward()
assert fake_audio.grad is not None
print(f"Gradients through audio: {fake_audio.grad.norm().item():.4f} ✓")

print("\n=== ALL TESTS PASSED ===")
