"""Offline smoke tests for the Z³ neural dynamics runtime.

Run with:
    python test_z3_neural_dynamics.py

The test suite intentionally avoids pytest so it can run in minimal environments.
It skips neural checks gracefully when PyTorch is not installed.
"""
from __future__ import annotations

import os
import tempfile

PASS = 0
FAIL = 0
SKIP = 0


def check(name: str, condition: bool, detail: object = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} -- {detail}")


def skip(name: str, detail: object = "") -> None:
    global SKIP
    SKIP += 1
    print(f"  SKIP {name} -- {detail}")


print("=" * 60)
print("TESTING Z3 NEURAL DYNAMICS")
print("=" * 60)

try:
    import torch
    from Z3_neural_dynamics import Z3Config, Z3NeuralDynamics, generate_regime_sequence, prepare_embedding_pairs
except ModuleNotFoundError as exc:
    skip("PyTorch neural checks", exc)
    print("=" * 60)
    print(f"PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
    raise SystemExit(0)


torch.manual_seed(11)
config = Z3Config(
    input_dim=8,
    context_dim=16,
    state_dim=24,
    local_dim=12,
    evidence_dim=10,
    hidden_dim=32,
    agent_count=4,
    agent_embed_dim=6,
    noise_scale=0.0,
)
model = Z3NeuralDynamics(config)
coherence_config = Z3Config.internal_coherence(input_dim=8)
balanced_config = Z3Config.balanced(input_dim=8)

check("Internal coherence preset down-weights prediction", coherence_config.beta_predictive < config.beta_predictive, coherence_config.beta_predictive)
check("Balanced preset keeps prediction and coherence active", balanced_config.beta_predictive > 0 and balanced_config.beta_coherence_band > 0, balanced_config)
check("Phi anti-monopoly regularizer is active", config.beta_phi_balance > 0.0 and config.phi_concentration_ratio_max >= 1.0, config)
check("Adaptive threshold regularizer is configured", config.adaptive_thresholds and config.beta_gate_rate > 0.0 and config.tau_min > 0.0, config)
check("Boot projection regularizers are configured", config.beta_boot_context > 0.0 and config.beta_boot_variance > 0.0, config)
check("Rare-expert credit is configured", config.rare_expert_decay > 0.0 and config.rare_expert_trust_bonus > 0.0, config)
check("Z-prime equation force terms are configured", config.entropy_force_strength > 0.0 and config.global_entropy_strength > 0.0 and config.inverse_square_repulsion_strength > 0.0 and config.cubic_self_strength > 0.0, config)
check("Initial phi starts near unit gain", bool(torch.allclose(model.phi, torch.ones_like(model.phi), atol=2e-4)), model.phi)
check("Clustering gamma is bounded", bool(0.0 <= float(model.clustering_gamma.detach()) <= 1.0), model.clustering_gamma)
check("Metric buffer size follows metric schema", tuple(model.last_metrics.shape) == (model.metric_count(),), model.last_metrics.shape)

x, y = generate_regime_sequence(8, config.input_dim, batch_size=2)
real_stream = torch.randn(2, 5, config.input_dim)
real_x, real_y = prepare_embedding_pairs(real_stream)
check("Embedding adapter flattens next-step pairs", tuple(real_x.shape) == (8, config.input_dim) and tuple(real_y.shape) == (8, config.input_dim), (real_x.shape, real_y.shape))

output = model.forward(x[:4], target=y[:4], update_state=False, add_noise=False)
check("Z3 before shape", tuple(output["z3_before"].shape) == (4, config.state_dim), output["z3_before"].shape)
check("Z3 after shape", tuple(output["z3_after"].shape) == (4, config.state_dim), output["z3_after"].shape)
check("Z-prime agent shape", tuple(output["agents_after"].shape) == (4, config.agent_count, config.local_dim), output["agents_after"].shape)
check("Evidence shape", tuple(output["evidence"].shape) == (4, config.agent_count, config.evidence_dim), output["evidence"].shape)
check("Soft gates bounded", bool(torch.all(output["gate"] >= 0.0) and torch.all(output["gate"] <= 1.0)), output["gate"])
check("Trust weights normalized", bool(torch.allclose(output["trust"].sum(dim=1), torch.ones(4), atol=1e-4)), output["trust"].sum(dim=1))
check("Trust weights remain positive under floor", bool(torch.all(output["trust"] > 0.0)), output["trust"])
check("Loss is finite", bool(torch.isfinite(output["losses"]["total"])), output["losses"]["total"])
check("Phi balance loss is exposed", "phi_balance" in output["losses"] and "phi_effective_agents" in output["losses"], output["losses"].keys())
check("Boundary calibration losses are exposed", all(k in output["losses"] for k in ("boot_context", "boot_variance", "gate_rate_band", "mean_theta_novelty", "mean_tau_coherence")), output["losses"].keys())
check("Adaptive thresholds have batch shape", tuple(output["theta_novelty_eff"].shape) == (4, 1) and tuple(output["tau_coherence_eff"].shape) == (4, 1), (output["theta_novelty_eff"].shape, output["tau_coherence_eff"].shape))
check("Adaptive temperatures respect tau floor", bool(torch.all(output["tau_novelty_eff"] >= config.tau_min) and torch.all(output["tau_coherence_eff"] >= config.tau_min)), (output["tau_novelty_eff"], output["tau_coherence_eff"]))
check("Uniform phi has no monopoly penalty", bool(output["losses"]["phi_balance"] <= 1e-8), output["losses"]["phi_balance"])
check("Metric vector matches schema", tuple(output["metrics"].shape) == (model.metric_count(),), output["metrics"].shape)
check("Z-prime equation diagnostics are exposed", all(k in output for k in ("agent_entropy", "global_entropy", "gamma", "phase_vectors", "phase_alignment", "entropy_force", "inverse_square_repulsion", "global_entropy_force", "cubic_self_drive", "physics_force")), output.keys())
check("Entropy scalar shapes are explicit", tuple(output["agent_entropy"].shape) == (4, config.agent_count) and tuple(output["global_entropy"].shape) == (4,), (output["agent_entropy"].shape, output["global_entropy"].shape))
check("Phase vectors match local agent shape", tuple(output["phase_vectors"].shape) == (4, config.agent_count, config.local_dim), output["phase_vectors"].shape)
check("Physics force terms are finite", all(bool(torch.isfinite(output[k]).all()) for k in ("entropy_force", "inverse_square_repulsion", "global_entropy_force", "cubic_self_drive", "physics_force")), {k: output[k] for k in ("entropy_force", "inverse_square_repulsion", "global_entropy_force", "cubic_self_drive", "physics_force")})
metric_dict = model.metrics_to_dict(output["metrics"], output["losses"])
check("New Z-prime equation metrics are exposed", all(k in metric_dict for k in ("mean_agent_entropy", "mean_global_entropy", "gamma", "phase_alignment", "entropy_force_norm", "inverse_square_repulsion_norm", "global_entropy_force_norm", "cubic_self_drive_norm", "physics_force_norm")), metric_dict.keys())
check("New Z-prime equation metrics are finite", all(metric_dict[k] == metric_dict[k] for k in ("mean_agent_entropy", "mean_global_entropy", "gamma", "phase_alignment", "entropy_force_norm", "inverse_square_repulsion_norm", "global_entropy_force_norm", "cubic_self_drive_norm", "physics_force_norm")), metric_dict)

zero_trust = torch.zeros(3, config.agent_count)
zero_weights = model.normalize_trust(zero_trust)
expected_uniform = torch.full_like(zero_weights, 1.0 / config.agent_count)
check("Zero-trust fallback is uniform", bool(torch.allclose(zero_weights, expected_uniform, atol=1e-6)), zero_weights)

with torch.no_grad():
    model.raw_phi[:] = torch.tensor([5.0, -3.0, -3.0, -3.0], dtype=model.raw_phi.dtype)
monopoly_output = model.forward(x[:2], target=y[:2], update_state=False, add_noise=False)
check("Concentrated phi triggers anti-monopoly penalty", bool(monopoly_output["losses"]["phi_balance"] > 0.0), monopoly_output["losses"]["phi_balance"])
model.reset_state(seed=11)
with torch.no_grad():
    model.raw_phi.fill_(torch.log(torch.expm1(torch.tensor(1.0))).item())

states = torch.randn(2, config.agent_count, config.local_dim)
distances = torch.cdist(states, states, p=2)
mask = torch.triu(torch.ones(config.agent_count, config.agent_count, dtype=torch.bool), diagonal=1)
manual_batch_local = distances[:, mask].mean()
helper_batch_local = model.mean_agent_pairwise_distance(states)
flattened_cross_batch = torch.pdist(states.reshape(-1, config.local_dim), p=2).mean()
check("Batch-local diversity helper matches manual calculation", bool(torch.allclose(helper_batch_local, manual_batch_local, atol=1e-6)), (helper_batch_local, manual_batch_local))
check("Batch-local diversity avoids flattened cross-batch calculation", bool(not torch.allclose(helper_batch_local, flattened_cross_batch, atol=1e-6)), (helper_batch_local, flattened_cross_batch))

clustered = torch.zeros(1, config.agent_count, config.local_dim)
clustered[0, 1, 0] = 0.01
clustered[0, 2, 0] = 0.02
clustered[0, 3, 0] = 0.03
repulsion = model.pairwise_repulsion_field(clustered)
check("Pairwise repulsion field is active for clustered agents", bool(torch.norm(repulsion) > 0), repulsion)
phase_clustered = torch.randn_like(clustered)
phase_clustered = torch.nn.functional.normalize(phase_clustered, dim=-1)
inverse_repulsion = model.inverse_square_repulsion_field(clustered, phase_clustered)
check("Inverse-square repulsion remains finite for near-overlap", bool(torch.isfinite(inverse_repulsion).all() and torch.norm(inverse_repulsion) > 0), inverse_repulsion)
entropy_probe = torch.randn(1, config.agent_count)
entropy_force = model.entropy_gradient_force(entropy_probe, phase_clustered)
check("Entropy-gradient force shape is local-agent shaped", tuple(entropy_force.shape) == (1, config.agent_count, config.local_dim) and torch.isfinite(entropy_force).all(), entropy_force)
cubic_probe = model.cubic_self_recursion(clustered)
check("Cubic self-recursion is finite and local-agent shaped", tuple(cubic_probe.shape) == (1, config.agent_count, config.local_dim) and torch.isfinite(cubic_probe).all(), cubic_probe)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
before = model.z3_state.detach().clone()
metrics = model.train_step(optimizer, x[:8], target=y[:8], update_recurrent_state=True)
after = model.z3_state.detach().clone()
check("Train step returns metrics", isinstance(metrics, dict) and "loss_total" in metrics, metrics)
check("Train step loss finite", metrics.get("loss_total", float("nan")) == metrics.get("loss_total", float("nan")), metrics)
check("Persistent Z3 state advances", bool(torch.norm(after - before) > 0), (before, after))
check("Rare-expert credit updates after recurrent commit", bool(torch.norm(model.rare_expert_credit) > 0), model.rare_expert_credit)
check("Train metrics expose boundary calibration", "boot_context" in metrics and "gate_rate_band" in metrics and "mean_theta_novelty" in metrics, metrics)

stream_model = Z3NeuralDynamics(config)
stream_optimizer = torch.optim.AdamW(stream_model.parameters(), lr=1e-3)
stream_before = stream_model.z3_state.detach().clone()
window_metrics = stream_model.train_sequence_window(stream_optimizer, real_stream, truncation_steps=2, commit_recurrent_state=True, add_noise=False)
stream_after = stream_model.z3_state.detach().clone()
check("Truncated BPTT window returns metrics", isinstance(window_metrics, dict) and "window_loss" in window_metrics, window_metrics)
check("Truncated BPTT chunk count is bounded", window_metrics.get("truncated_bptt_chunks") == 2.0, window_metrics)
check("Truncated BPTT commits detached recurrent state", bool(torch.norm(stream_after - stream_before) > 0 and not stream_model.z3_state.requires_grad), stream_model.z3_state)
check("Truncated BPTT updates rare-expert credit", bool(torch.norm(stream_model.rare_expert_credit) > 0), stream_model.rare_expert_credit)

projection = model.public_projection(output)
check("Public projection exposes z_cubed_state", "z_cubed_state" in projection, projection)
check("Public projection exposes phi", "phi" in projection and 0.0 <= projection["phi"] <= 1.0, projection)
check("Public projection exposes learning metrics", "learning" in projection, projection)

with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint = os.path.join(tmpdir, "z3.pt")
    model.save_checkpoint(checkpoint)
    restored = Z3NeuralDynamics.load_checkpoint(checkpoint)
    check("Checkpoint restores config", restored.config.input_dim == model.config.input_dim, restored.config)
    check("Checkpoint restores recurrent state", bool(torch.allclose(restored.z3_state, model.z3_state)), (restored.z3_state, model.z3_state))
    legacy_checkpoint = os.path.join(tmpdir, "legacy_z3.pt")
    payload = torch.load(checkpoint)
    payload["last_metrics"] = payload["last_metrics"][:8]
    payload["state_dict"] = dict(payload["state_dict"])
    payload["state_dict"]["last_metrics"] = payload["state_dict"]["last_metrics"][:8]
    torch.save(payload, legacy_checkpoint)
    restored_legacy = Z3NeuralDynamics.load_checkpoint(legacy_checkpoint)
    check("Legacy checkpoint metric buffer is padded", tuple(restored_legacy.last_metrics.shape) == (restored_legacy.metric_count(),), restored_legacy.last_metrics.shape)

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
raise SystemExit(1 if FAIL else 0)
