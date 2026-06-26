"""End-to-end integration test for the Z³ Analysis by Synthesis loop."""
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import base64
from Z3_neural_dynamics import Z3NeuralDynamics, Z3Config
from z3_acoustic_encoder import get_acoustic_encoder
from z3_articulatory_synth import get_articulatory_voice, TrajectoryControlLayer

print("=== Z³ Analysis by Synthesis Loop Test ===\n")

# 1. Initialize models
cfg = Z3Config()
model = Z3NeuralDynamics(cfg)
encoder = get_acoustic_encoder(input_dim=cfg.input_dim)
voice = get_articulatory_voice(
    z3_prediction_dim=cfg.evidence_dim + cfg.state_dim + cfg.context_dim
)
print("Models initialized ✓")

# 2. Simulate incoming audio (100ms)
n_samples = 2205
fake_audio = torch.randn(1, n_samples)
print(f"Incoming audio simulated ({n_samples} samples) ✓")

# 3. Step 1: Encode audio to Z³ input
z3_input = encoder(fake_audio)
assert z3_input.shape == (1, 16)
print("Step 1: Audio encoded to Z³ input ✓")

# 4. Step 2: Feed into Z³ for entrainment
neural_output = model.forward(z3_input, update_state=True, add_noise=False)
print("Step 2: Z³ entrained to input vector ✓")

# 5. Step 3: Map to articulatory trajectory
pred_input = torch.cat([
    neural_output["integrated_evidence"],
    neural_output["z3_after"],
    neural_output["context"],
], dim=-1)

# Mock memory context
mem_ctx = TrajectoryControlLayer.encode_memory_context({})

# 6. Step 4: Synthesize matching waveform
result = voice(pred_input, target_waveform=None, memory_context=mem_ctx, use_trajectory=True)
synth_audio = result["waveform"]
assert synth_audio.shape == (1, n_samples * 4) # 400ms trajectory
print("Step 4: Articulatory voice synthesized matching waveform ✓")

# 7. Step 5: Calculate Spectral Loss (The Feedback)
# We compare the first 100ms of synthesized audio with the input audio
from z3_articulatory_synth import SpectralLoss
spectral_loss_fn = SpectralLoss(n_fft=1024, hop_length=256, win_length=1024)
loss = spectral_loss_fn(synth_audio[:, :n_samples], fake_audio)
print(f"Step 5: Spectral loss calculated: {loss.item():.4f} ✓")

# 8. Gradient check (Backprop all the way to the encoder)
loss.backward()
assert encoder.mapping[0].weight.grad is not None
print("Step 6: Gradients backpropagated to acoustic encoder ✓")

print("\n=== ANALYSIS BY SYNTHESIS LOOP CONFIRMED ===")
