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
check("Initial phi starts near unit gain", bool(torch.allclose(model.phi, torch.ones_like(model.phi), atol=2e-4)), model.phi)
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

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL} SKIP={SKIP}")
raise SystemExit(1 if FAIL else 0)
