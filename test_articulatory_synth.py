"""Smoke test for Z³ Differentiable Articulatory Synthesizer Layer."""
import sys
sys.path.insert(0, ".")

import torch
from Z3_neural_dynamics import Z3NeuralDynamics, Z3Config
from z3_articulatory_synth import (
    ARTICULATORY_PARAMS, N_ARTICULATORY,
    ArticulatoryControlLayer,
    TrajectoryControlLayer,
    DifferentiableFormantSynth,
    SpectralLoss,
    Z3ArticulatoryVoice,
    get_articulatory_voice,
    differentiable_formant_resonator,
)

print("=== Z³ Articulatory Synthesizer Smoke Test ===\n")

# ── 1. Parameter space ──────────────────────────────────────────────────────
assert N_ARTICULATORY == 16
print(f"Articulatory parameters ({N_ARTICULATORY}):")
for i, p in enumerate(ARTICULATORY_PARAMS):
    print(f"  {i:2d}  {p['name']:22s}  [{p['min']:7.1f} – {p['max']:7.1f}]  default={p['default']}")

# ── 2. Frame-level ArticulatoryControlLayer ─────────────────────────────────
z3_pred_dim = 24 + 64 + 48  # evidence + state + context = 136
ctrl = ArticulatoryControlLayer(input_dim=z3_pred_dim, hidden_dim=128)
print(f"\nArticulatoryControlLayer params: {sum(p.numel() for p in ctrl.parameters()):,}")

fake_pred = torch.randn(2, z3_pred_dim)
art_params = ctrl(fake_pred)
assert art_params.shape == (2, 16)
assert (art_params >= 0).all() and (art_params <= 1).all()
print(f"Frame-level output shape: {art_params.shape} ✓")

# ── 3. TrajectoryControlLayer ───────────────────────────────────────────────
TRAJ_STEPS = 4
traj_ctrl = TrajectoryControlLayer(
    z3_pred_dim=z3_pred_dim,
    memory_dim=TrajectoryControlLayer.MEMORY_DIM,
    hidden_dim=256,
    trajectory_steps=TRAJ_STEPS,
)
print(f"\nTrajectoryControlLayer params: {sum(p.numel() for p in traj_ctrl.parameters()):,}")

# Without memory context (frame-level fallback)
traj_no_mem = traj_ctrl(fake_pred, memory_context=None)
assert traj_no_mem.shape == (2, TRAJ_STEPS, 16), f"wrong shape: {traj_no_mem.shape}"
assert (traj_no_mem >= 0).all() and (traj_no_mem <= 1).all()
print(f"Trajectory output (no memory): {traj_no_mem.shape} ✓")

# With memory context
fake_mem = torch.rand(2, TrajectoryControlLayer.MEMORY_DIM)
traj_with_mem = traj_ctrl(fake_pred, memory_context=fake_mem)
assert traj_with_mem.shape == (2, TRAJ_STEPS, 16)
print(f"Trajectory output (with memory): {traj_with_mem.shape} ✓")

# Verify trajectory steps are NOT identical — the GRU is producing different
# configurations at each step (coarticulation is happening)
step_diffs = []
for s in range(TRAJ_STEPS - 1):
    diff = (traj_with_mem[:, s+1, :] - traj_with_mem[:, s, :]).abs().mean().item()
    step_diffs.append(diff)
print(f"Mean parameter change between trajectory steps: {[f'{d:.4f}' for d in step_diffs]}")
assert all(d > 0 for d in step_diffs), "Trajectory steps are identical — GRU not working"
print("Coarticulation dynamics confirmed: steps differ ✓")

# Gradients flow through trajectory
pred_g = fake_pred.clone().requires_grad_(True)
mem_g = fake_mem.clone().requires_grad_(True)
traj_g = traj_ctrl(pred_g, memory_context=mem_g)
traj_g.sum().backward()
assert pred_g.grad is not None
assert mem_g.grad is not None
print(f"Gradients through trajectory (pred norm: {pred_g.grad.norm():.4f}, "
      f"mem norm: {mem_g.grad.norm():.4f}) ✓")

# ── 4. Memory context encoding ──────────────────────────────────────────────
mock_memory_output = {
    "reconstruction_confidence": 0.72,
    "salience": 0.65,
    "phase_position": 0.31,
    "top_matches": [
        {"resonance": 0.81, "phase_alignment": 0.74},
        {"resonance": 0.63, "phase_alignment": 0.58},
        {"resonance": 0.45, "phase_alignment": 0.41},
    ],
}
mem_ctx = TrajectoryControlLayer.encode_memory_context(mock_memory_output)
assert mem_ctx.shape == (1, TrajectoryControlLayer.MEMORY_DIM)
assert abs(mem_ctx[0, 0].item() - 0.72) < 1e-5  # reconstruction_confidence
assert abs(mem_ctx[0, 1].item() - 0.65) < 1e-5  # salience
assert abs(mem_ctx[0, 2].item() - 0.31) < 1e-5  # phase_position
assert abs(mem_ctx[0, 3].item() - 0.81) < 1e-5  # top resonance[0]
assert abs(mem_ctx[0, 8].item() - 0.74) < 1e-5  # phase_alignment[0]
assert mem_ctx[0, 6].item() == 0.0              # resonance[3] — zero-padded (only 3 matches)
assert mem_ctx[0, 11].item() == 0.0             # phase_alignment[3] — zero-padded
print(f"\nMemory context encoding shape: {mem_ctx.shape} ✓")
print(f"Memory context values: {mem_ctx[0].tolist()}")

# ── 5. Differentiable formant resonator ─────────────────────────────────────
source = torch.randn(2, 2205)
freq = torch.tensor([500.0, 800.0])
bw = torch.tensor([80.0, 100.0])
out = differentiable_formant_resonator(source, freq, bw, sample_rate=22050)
assert out.shape == (2, 2205)

freq_g = freq.clone().requires_grad_(True)
bw_g = bw.clone().requires_grad_(True)
out_g = differentiable_formant_resonator(source.clone(), freq_g, bw_g, 22050)
out_g.sum().backward()
assert freq_g.grad is not None and bw_g.grad is not None
print(f"\nFormant resonator gradients (freq: {freq_g.grad.norm():.4f}) ✓")

# ── 6. Full Z3ArticulatoryVoice — trajectory mode ───────────────────────────
voice = Z3ArticulatoryVoice(
    z3_prediction_dim=z3_pred_dim,
    sample_rate=22050,
    n_samples=2205,
    trajectory_steps=TRAJ_STEPS,
)
total_params = sum(p.numel() for p in voice.parameters())
print(f"\nZ3ArticulatoryVoice total params: {total_params:,}")

# Forward with trajectory + memory + loss
target = torch.randn(2, 2205 * TRAJ_STEPS) * 0.3
result = voice(fake_pred, target_waveform=target, memory_context=fake_mem, use_trajectory=True)
assert result["waveform"].shape == (2, 2205 * TRAJ_STEPS)
assert result["articulatory_params"].shape == (2, TRAJ_STEPS, 16)
assert "loss" in result
assert result["mode"] == "trajectory"
print(f"Trajectory forward: waveform {result['waveform'].shape}, "
      f"loss {result['loss'].item():.4f} ✓")

# Frame-level fallback
result_frame = voice(fake_pred, use_trajectory=False)
assert result_frame["waveform"].shape == (2, 2205)
assert result_frame["mode"] == "frame"
print(f"Frame-level fallback: waveform {result_frame['waveform'].shape} ✓")

# End-to-end gradient through trajectory
pred_e2e = fake_pred.clone().requires_grad_(True)
mem_e2e = fake_mem.clone().requires_grad_(True)
res_e2e = voice(pred_e2e, target_waveform=target, memory_context=mem_e2e)
res_e2e["loss"].backward()
assert pred_e2e.grad is not None
assert mem_e2e.grad is not None
print(f"E2E gradient norm (pred: {pred_e2e.grad.norm():.4f}, "
      f"mem: {mem_e2e.grad.norm():.4f}) ✓")

# ── 7. articulatory_state_broadcast with trajectory preview ─────────────────
broadcast = voice.articulatory_state_broadcast(
    fake_pred[:1], memory_context=fake_mem[:1]
)
assert broadcast["type"] == "articulatory_update"
assert len(broadcast["trajectory"]) == TRAJ_STEPS
assert len(broadcast["formant_hz"]) == 4
assert "memory_resonance" in broadcast
print(f"\nBroadcast payload keys: {list(broadcast.keys())} ✓")
print(f"Trajectory preview ({TRAJ_STEPS} steps):")
for i, step in enumerate(broadcast["trajectory"]):
    print(f"  Step {i}: F1={step[0]:.1f} F2={step[1]:.1f} F3={step[2]:.1f} F4={step[3]:.1f} Hz")

# ── 8. Integration with real Z³ model ───────────────────────────────────────
print("\n--- Integration with Z³ neural core ---")
cfg = Z3Config()
z3_model = Z3NeuralDynamics(cfg)
x = torch.randn(1, cfg.input_dim)
with torch.no_grad():
    z3_out = z3_model.forward(x, update_state=False, add_noise=False)

pred_input = torch.cat([
    z3_out["integrated_evidence"],
    z3_out["z3_after"],
    z3_out["context"],
], dim=-1)
assert pred_input.shape[-1] == z3_pred_dim

# Encode mock memory context
mem_ctx_z3 = TrajectoryControlLayer.encode_memory_context(mock_memory_output)
art_state = voice.articulatory_state(pred_input, memory_context=mem_ctx_z3)
print("Vocal tract geometry from Z³ state (trajectory step 0):")
for name, val in art_state.items():
    print(f"  {name:22s}: {val:.4f}")

# PCM output (full trajectory = 4 × 100ms = 400ms)
pcm_bytes = voice.synthesize(pred_input, memory_context=mem_ctx_z3, use_trajectory=True)
expected_bytes = 2205 * TRAJ_STEPS * 2
assert len(pcm_bytes) == expected_bytes, f"wrong PCM length: {len(pcm_bytes)}"
print(f"\nPCM output: {len(pcm_bytes)} bytes "
      f"({len(pcm_bytes)//2} samples = {TRAJ_STEPS}×100ms trajectory at 22050 Hz) ✓")

print("\n=== ALL TESTS PASSED ===")
