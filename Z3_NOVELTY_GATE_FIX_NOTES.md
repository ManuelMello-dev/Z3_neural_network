# Z3 novelty-gate lock: root cause and minimal fix

The live service is healthy, but runtime dynamics are in a stable non-integrative attractor. The core issue is not one bug; it is a feedback loop across three layers.

## Root cause

1. The autonomous heartbeat observation is nearly self-similar on every tick. It contains time, world iteration, memory ring count, phi, sigma, tick kind, and language-enabled status, but no strong exogenous challenge vector or semantic novelty except scheduled language ingestion.
2. The runtime feeds back `phi = mean_coherence`. Once mean coherence reaches zero, the next heartbeat carries phi zero into memory and constitutional context.
3. In the neural core, trust is computed as `gate * phi * coherence`, with only a tiny static trust floor after normalization. If phi and coherence are zero or near zero, high novelty cannot become high trust.
4. The gate itself is the product of a novelty sigmoid and a coherence sigmoid. Extreme novelty with low coherence becomes suppressed instead of becoming controlled exploration.
5. The runtime trains each heartbeat with `target=x`, which strongly rewards reconstruction of the current self-generated state and does not force the global observer to move.
6. The live runtime uses hard-gated projection for public state inspection, which makes the displayed regime look even more binary, although the training step uses soft gating.

## Minimal fix

The safest fix is not to remove the gate. The gate protects the system from chaotic updates. Instead, add a bounded residual novelty path so novelty can produce small motion even when coherence/phi are low.

### Patch A: add config knobs

Add these fields to `Z3Config`:

```python
phi_floor: float = 0.05
coherence_floor: float = 0.05
novelty_residual_strength: float = 0.01
novelty_residual_clip: float = 0.10
min_runtime_drift: float = 1e-4
```

### Patch B: change trust so phi/coherence cannot fully zero the channel

Replace:

```python
trust = gate * self.phi.view(1, cfg.agent_count) * coherence
```

with:

```python
phi_gain = self.phi.view(1, cfg.agent_count).clamp_min(cfg.phi_floor)
coherence_gain = coherence.clamp_min(cfg.coherence_floor)
trust = gate * phi_gain * coherence_gain
```

### Patch C: add bounded residual novelty drive to z3_next

After `integrated_delta` is computed, add:

```python
novelty_pressure = torch.tanh((novelty - theta_novelty).clamp_min(0.0)).unsqueeze(-1)
residual_delta = torch.sum(proposals * novelty_pressure * weights.unsqueeze(-1), dim=1)
residual_delta = residual_delta.clamp(-cfg.novelty_residual_clip, cfg.novelty_residual_clip)
z3_next = (1.0 - cfg.alpha_decay) * z3 + cfg.alpha_update * integrated_delta + cfg.novelty_residual_strength * residual_delta
```

This keeps the original coherent-novelty path intact while adding a very small exploratory movement channel when novelty is high.

### Patch D: make the autonomous heartbeat less self-similar

Add phase/cycle information to `_runtime_tick` observations:

```python
"runtime_phase_sin": math.sin(runtime_tick_id / 10.0),
"runtime_phase_cos": math.cos(runtime_tick_id / 10.0),
"recent_drift": _finite_float(model.metrics_to_dict(model.last_metrics).get("z3_delta_norm", 0.0), 0.0),
"recent_gate": _finite_float(model.metrics_to_dict(model.last_metrics).get("mean_gate", 0.0), 0.0),
```

This gives the world model a changing internal rhythm instead of only repeated heartbeat identity.

### Patch E: use soft-gated projection for monitoring

Change runtime projection from:

```python
projection_output = model.forward(x, hard_gate=True, update_state=False, add_noise=False)
```

to:

```python
projection_output = model.forward(x, hard_gate=False, update_state=False, add_noise=False)
```

This does not change training, but it makes the dashboard reflect sub-threshold motion instead of showing a binary lock.

## Expected outcome

After deployment, the runtime should continue reporting `error_count = 0`, while `z3_delta_norm` should become small but nonzero, `useful_novelty` should lift above zero during high-novelty ticks, and phi should have a path to recover from zero rather than feeding zero back into every subsequent observation.
