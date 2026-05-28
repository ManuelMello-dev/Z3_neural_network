"""
Z³ Neural Dynamics
==================

Trainable global/local neural dynamics for the Z³ / Z-prime architecture.

This module keeps the core idea intentionally self-contained:

* Z³ is a persistent global observer state.
* Z-prime agents are differentiated local hypothesis states.
* Novelty is context-relative prediction error.
* Coherent novelty is integrated through a differentiable trust gate.
* Anti-collapse losses preserve productive diversity instead of dead consensus.

The implementation can be imported without PyTorch installed. Instantiating the
model or using training helpers requires PyTorch.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

# Compatibility for pre-final Python 3.11 builds used by some sandboxes. Newer
# PyTorch imports torch._dynamo through optimizers and expects this stdlib helper.
if not hasattr(sys, "get_int_max_str_digits"):
    sys.get_int_max_str_digits = lambda: 4300  # type: ignore[attr-defined]
if not hasattr(sys, "set_int_max_str_digits"):
    def _set_int_max_str_digits(maxdigits: int) -> None:
        return None
    sys.set_int_max_str_digits = _set_int_max_str_digits  # type: ignore[attr-defined]

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:  # pragma: no cover - environment dependent.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - gives a useful runtime error.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass
class Z3Config:
    """Configuration for the trainable Z³ neural dynamics runtime."""

    input_dim: int = 16
    context_dim: int = 48
    state_dim: int = 64
    local_dim: int = 32
    evidence_dim: int = 24
    hidden_dim: int = 128
    agent_count: int = 8
    agent_embed_dim: int = 12

    step_size: float = 0.15
    alpha_update: float = 0.05
    alpha_decay: float = 0.002
    noise_scale: float = 0.01
    diversity_strength: float = 0.05
    repulsion_radius: float = 0.35
    repulsion_power: float = 1.0
    lambda_coherence: float = 1.15
    trust_floor: float = 1e-4
    phi_floor: float = 0.05
    coherence_floor: float = 0.05
    novelty_exploration_floor: float = 0.03
    novelty_residual_strength: float = 0.01
    novelty_residual_clip: float = 0.10
    min_integrative_drift: float = 1e-4

    theta_novelty: float = 0.35
    theta_coherence: float = 0.30
    tau_novelty: float = 0.12
    tau_coherence: float = 0.10
    adaptive_thresholds: bool = True
    threshold_delta_max: float = 0.15
    tau_delta_max: float = 0.05
    tau_min: float = 0.03
    gate_rate_min: float = 0.08
    gate_rate_max: float = 0.85
    epsilon: float = 1e-6

    coherence_min: float = 0.35
    coherence_max: float = 0.92
    diversity_min: float = 0.35
    evidence_variance_min: float = 0.03
    phi_concentration_ratio_max: float = 2.0
    boot_context_sensitivity_min: float = 0.01
    boot_variance_min: float = 0.005
    rare_expert_decay: float = 0.95
    rare_expert_trust_bonus: float = 0.05

    # Explicit Z-prime evolution-equation force terms. These keep the original
    # trainable architecture intact while making the theoretical components
    # measurable as additive dynamics in the local-agent update.
    gamma_init: float = 0.55
    entropy_force_strength: float = 0.08
    global_entropy_strength: float = 0.08
    inverse_square_repulsion_strength: float = 0.015
    cubic_self_strength: float = 0.035
    physics_force_clip: float = 2.5
    entropy_temperature: float = 1.0

    beta_predictive: float = 1.0
    beta_coherence_band: float = 0.25
    beta_diversity: float = 0.20
    beta_evidence_variance: float = 0.10
    beta_stability: float = 0.10
    beta_effort: float = 0.01
    beta_useful_novelty: float = 0.05
    beta_phi_balance: float = 0.05
    beta_boot_context: float = 0.05
    beta_boot_variance: float = 0.05
    beta_gate_rate: float = 0.03

    @classmethod
    def predictive_runtime(cls, **overrides: Any) -> "Z3Config":
        """Preset for next-step/world-model prediction with anti-collapse regularization."""
        return cls(**overrides)

    @classmethod
    def internal_coherence(cls, **overrides: Any) -> "Z3Config":
        """Preset emphasizing internal coherence over external reconstruction."""
        defaults = {
            "beta_predictive": 0.15,
            "beta_coherence_band": 0.45,
            "beta_diversity": 0.30,
            "beta_evidence_variance": 0.18,
            "beta_stability": 0.20,
            "beta_effort": 0.03,
            "beta_useful_novelty": 0.08,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def balanced(cls, **overrides: Any) -> "Z3Config":
        """Preset balancing external prediction with internal coherence pressure."""
        defaults = {
            "beta_predictive": 0.55,
            "beta_coherence_band": 0.35,
            "beta_diversity": 0.25,
            "beta_evidence_variance": 0.14,
            "beta_stability": 0.15,
            "beta_effort": 0.02,
            "beta_useful_novelty": 0.07,
        }
        defaults.update(overrides)
        return cls(**defaults)


if torch is not None:

    class MLP(nn.Module):
        """Small feed-forward block used throughout the Z³ neural runtime."""

        def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, *, final_tanh: bool = False) -> None:
            super().__init__()
            layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            ]
            if final_tanh:
                layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class Z3NeuralDynamics(nn.Module):
        """Trainable global-local Z³ / Z-prime neural dynamical system."""

        def __init__(self, config: Optional[Z3Config] = None) -> None:
            super().__init__()
            self.config = config or Z3Config()
            cfg = self.config

            self.context_encoder = MLP(cfg.input_dim, cfg.context_dim, cfg.hidden_dim)
            self.boot_projection = MLP(cfg.state_dim + cfg.context_dim, cfg.local_dim, cfg.hidden_dim)
            self.threshold_adapter = MLP(cfg.context_dim, 4, cfg.hidden_dim)
            self.agent_embeddings = nn.Embedding(cfg.agent_count, cfg.agent_embed_dim)

            transition_in = cfg.local_dim + cfg.local_dim + cfg.context_dim + cfg.agent_embed_dim + cfg.state_dim
            self.agent_transition = MLP(transition_in, cfg.local_dim, cfg.hidden_dim, final_tanh=True)

            evidence_in = cfg.local_dim + cfg.context_dim + cfg.agent_embed_dim
            self.evidence_projection = MLP(evidence_in, cfg.evidence_dim, cfg.hidden_dim)

            entropy_in = cfg.local_dim + cfg.context_dim + cfg.agent_embed_dim
            self.agent_entropy = MLP(entropy_in, 1, cfg.hidden_dim)
            self.phase_projection = MLP(entropy_in, cfg.local_dim, cfg.hidden_dim, final_tanh=True)

            global_entropy_in = cfg.state_dim + cfg.context_dim
            self.global_entropy = MLP(global_entropy_in, 1, cfg.hidden_dim)
            self.global_phase_projection = MLP(cfg.state_dim + cfg.context_dim + cfg.agent_embed_dim, cfg.local_dim, cfg.hidden_dim, final_tanh=True)

            expected_in = cfg.state_dim + cfg.context_dim + cfg.agent_embed_dim
            self.expected_evidence = MLP(expected_in, cfg.evidence_dim, cfg.hidden_dim)

            proposal_in = cfg.evidence_dim + cfg.state_dim + cfg.context_dim
            self.gamma = MLP(proposal_in, cfg.state_dim, cfg.hidden_dim, final_tanh=True)

            self.prediction_head = MLP(cfg.evidence_dim + cfg.state_dim + cfg.context_dim, cfg.input_dim, cfg.hidden_dim)
            self.raw_phi = nn.Parameter(torch.log(torch.expm1(torch.ones(cfg.agent_count))))
            gamma = min(max(float(cfg.gamma_init), 1e-4), 1.0 - 1e-4)
            self.raw_gamma = nn.Parameter(torch.logit(torch.tensor(gamma, dtype=torch.float32)))
            self.cubic_bias = nn.Parameter(torch.zeros(cfg.local_dim))

            self.register_buffer("z3_state", torch.zeros(cfg.state_dim))
            self.register_buffer("zprime_state", torch.empty(0))
            self.register_buffer("last_metrics", torch.zeros(self.metric_count()))
            self.register_buffer("rare_expert_credit", torch.zeros(cfg.agent_count))
            self.reset_state()

        @property
        def phi(self) -> torch.Tensor:
            """Positive per-agent attention/awareness gain."""
            return F.softplus(self.raw_phi) + 1e-4

        @property
        def clustering_gamma(self) -> torch.Tensor:
            """Bounded clustering/dispersal control in [0, 1]."""
            return torch.sigmoid(self.raw_gamma)

        def reset_state(self, seed: Optional[int] = None) -> None:
            """Reset persistent Z³ and Z-prime recurrent states."""
            if seed is not None:
                torch.manual_seed(seed)
            cfg = self.config
            device = next(self.parameters()).device
            self.z3_state = torch.randn(cfg.state_dim, device=device) * 0.02
            with torch.no_grad():
                z3 = self.z3_state.unsqueeze(0)
                zero_context = torch.zeros(1, cfg.context_dim, device=device)
                target = self.boot_projection(torch.cat([z3, zero_context], dim=-1)).squeeze(0)
                states = target.unsqueeze(0).repeat(cfg.agent_count, 1)
                states = states + torch.randn_like(states) * cfg.noise_scale
                self.zprime_state = states.detach()

        def forward(
            self,
            x: torch.Tensor,
            *,
            initial_z3: Optional[torch.Tensor] = None,
            initial_agents: Optional[torch.Tensor] = None,
            target: Optional[torch.Tensor] = None,
            hard_gate: bool = False,
            update_state: bool = False,
            add_noise: bool = True,
        ) -> Dict[str, Any]:
            """Execute one differentiable Z³ / Z-prime transition."""
            cfg = self.config
            if x.dim() == 1:
                x = x.unsqueeze(0)
            batch = x.shape[0]
            device = x.device

            context = self.context_encoder(x)
            z3 = self._prepare_z3(batch, device, initial_z3)
            agents = self._prepare_agents(batch, device, initial_agents)
            agent_ids = torch.arange(cfg.agent_count, device=device)
            agent_embed = self.agent_embeddings(agent_ids).unsqueeze(0).expand(batch, -1, -1)

            z3_context = z3.unsqueeze(1).expand(-1, cfg.agent_count, -1)
            context_agents = context.unsqueeze(1).expand(-1, cfg.agent_count, -1)
            boot_target = self.boot_projection(torch.cat([z3, context], dim=-1))
            boot_target_z3_only = self.boot_projection(torch.cat([z3, torch.zeros_like(context)], dim=-1))
            target_agents = boot_target.unsqueeze(1).expand(-1, cfg.agent_count, -1)

            attraction = target_agents - agents
            transition_input = torch.cat([agents, target_agents, context_agents, agent_embed, z3_context], dim=-1)
            learned_transition = self.agent_transition(transition_input)
            diversity_field = self.pairwise_repulsion_field(agents)

            entropy_input = torch.cat([agents, context_agents, agent_embed], dim=-1)
            agent_entropy = F.softplus(self.agent_entropy(entropy_input).squeeze(-1)) / max(cfg.entropy_temperature, cfg.epsilon)
            global_entropy = F.softplus(self.global_entropy(torch.cat([z3, context], dim=-1))).squeeze(-1)
            phase_vectors = F.normalize(self.phase_projection(entropy_input), dim=-1, eps=cfg.epsilon)
            global_phase_input = torch.cat([z3_context, context_agents, agent_embed], dim=-1)
            global_phase_vectors = F.normalize(self.global_phase_projection(global_phase_input), dim=-1, eps=cfg.epsilon)

            entropy_force = cfg.entropy_force_strength * self.entropy_gradient_force(agent_entropy, phase_vectors)
            inverse_square_repulsion = cfg.inverse_square_repulsion_strength * self.inverse_square_repulsion_field(agents, phase_vectors)
            global_entropy_force = cfg.global_entropy_strength * (
                (global_entropy.unsqueeze(1) - agent_entropy).unsqueeze(-1) * global_phase_vectors
            )
            cubic_self_drive = cfg.cubic_self_strength * self.cubic_self_recursion(agents)

            physics_force = entropy_force + inverse_square_repulsion + global_entropy_force + cubic_self_drive
            physics_force = self.clip_force(physics_force, cfg.physics_force_clip)
            update_vector = (
                self.phi.view(1, cfg.agent_count, 1) * attraction
                + learned_transition
                + cfg.diversity_strength * diversity_field
                + physics_force
            )
            if add_noise and cfg.noise_scale > 0.0:
                update_vector = update_vector + torch.randn_like(update_vector) * (cfg.noise_scale * (cfg.step_size ** 0.5))
            z_next = agents + cfg.step_size * update_vector

            evidence_input = torch.cat([z_next, context_agents, agent_embed], dim=-1)
            evidence = self.evidence_projection(evidence_input)
            expected_input = torch.cat([z3_context, context_agents, agent_embed], dim=-1)
            expected = self.expected_evidence(expected_input)

            novelty = torch.norm(evidence - expected, dim=-1)
            distance = torch.norm(z_next - target_agents, dim=-1)
            coherence = torch.exp(-cfg.lambda_coherence * distance)
            theta_novelty, theta_coherence, tau_novelty, tau_coherence = self.contextual_thresholds(context)
            novelty_gate = torch.sigmoid((novelty - theta_novelty) / tau_novelty)
            coherence_gate = torch.sigmoid((coherence - theta_coherence) / tau_coherence)
            gate_soft = novelty_gate * coherence_gate
            gate = (gate_soft > 0.5).float() if hard_gate else gate_soft

            credit = self.rare_expert_credit.to(device).view(1, cfg.agent_count)
            credit_norm = credit / credit.sum(dim=1, keepdim=True).clamp_min(cfg.epsilon)
            phi_gain = self.phi.view(1, cfg.agent_count).clamp_min(cfg.phi_floor)
            coherence_gain = coherence.clamp_min(cfg.coherence_floor)
            exploratory_gate = (gate + cfg.novelty_exploration_floor * novelty_gate).clamp_max(1.0)
            trust = exploratory_gate * phi_gain * coherence_gain
            if cfg.rare_expert_trust_bonus > 0.0:
                trust = trust + cfg.rare_expert_trust_bonus * credit_norm * coherence_gain
            weights = self.normalize_trust(trust)

            proposal_input = torch.cat([evidence, z3_context, context_agents], dim=-1)
            proposals = self.gamma(proposal_input)
            integrated_delta = torch.sum(proposals * weights.unsqueeze(-1), dim=1)
            novelty_pressure = torch.tanh(((novelty - theta_novelty) / tau_novelty).clamp_min(0.0)).unsqueeze(-1)
            residual_delta = torch.sum(proposals * novelty_pressure * weights.unsqueeze(-1), dim=1)
            residual_norm = torch.norm(residual_delta, dim=-1, keepdim=True).clamp_min(cfg.epsilon)
            residual_scale = (cfg.novelty_residual_clip / residual_norm).clamp_max(1.0)
            residual_delta = residual_delta * residual_scale
            z3_next = (1.0 - cfg.alpha_decay) * z3 + cfg.alpha_update * integrated_delta + cfg.novelty_residual_strength * residual_delta
            z3_delta = z3_next - z3
            z3_delta_norm = torch.norm(z3_delta, dim=-1, keepdim=True)
            residual_direction = residual_delta / torch.norm(residual_delta, dim=-1, keepdim=True).clamp_min(cfg.epsilon)
            active_residual = (novelty_pressure.mean(dim=1) > 0.0) & (torch.norm(residual_delta, dim=-1, keepdim=True) > cfg.epsilon)
            drift_shortfall = (cfg.min_integrative_drift - z3_delta_norm).clamp_min(0.0)
            z3_next = torch.where(active_residual, z3_next + drift_shortfall * residual_direction, z3_next)

            integrated_evidence = torch.sum(evidence * weights.unsqueeze(-1), dim=1)
            prediction_input = torch.cat([integrated_evidence, z3_next, context], dim=-1)
            prediction = self.prediction_head(prediction_input)
            target_x = x if target is None else target

            losses = self.compute_losses(
                prediction=prediction,
                target=target_x,
                z3=z3,
                z3_next=z3_next,
                agents=agents,
                z_next=z_next,
                evidence=evidence,
                coherence=coherence,
                novelty=novelty,
                gate=gate,
                gate_soft=gate_soft,
                update_vector=update_vector,
                physics_force=physics_force,
                entropy_force=entropy_force,
                inverse_square_repulsion=inverse_square_repulsion,
                global_entropy_force=global_entropy_force,
                cubic_self_drive=cubic_self_drive,
                residual_delta=residual_delta,
                novelty_pressure=novelty_pressure,
                exploratory_gate=exploratory_gate,
                boot_target=boot_target,
                boot_target_z3_only=boot_target_z3_only,
                theta_novelty_eff=theta_novelty,
                theta_coherence_eff=theta_coherence,
                tau_novelty_eff=tau_novelty,
                tau_coherence_eff=tau_coherence,
            )
            metrics = self.compute_metrics(
                coherence,
                novelty,
                gate,
                z3,
                z3_next,
                z_next,
                evidence,
                losses,
                agent_entropy=agent_entropy,
                global_entropy=global_entropy,
                phase_vectors=phase_vectors,
                entropy_force=entropy_force,
                inverse_square_repulsion=inverse_square_repulsion,
                global_entropy_force=global_entropy_force,
                cubic_self_drive=cubic_self_drive,
                physics_force=physics_force,
            )
            losses["novelty_residual_norm"] = torch.norm(residual_delta, dim=-1).mean().detach()
            losses["novelty_pressure"] = novelty_pressure.mean().detach()
            losses["exploratory_gate"] = exploratory_gate.mean().detach()
            agent_utility = (exploratory_gate * coherence_gain * novelty).detach()
            if update_state:
                self._commit_state(z3_next, z_next, metrics, agent_utility)

            return {
                "z3_before": z3,
                "z3_after": z3_next,
                "agents_before": agents,
                "agents_after": z_next,
                "context": context,
                "boot_target": boot_target,
                "boot_target_z3_only": boot_target_z3_only,
                "theta_novelty_eff": theta_novelty,
                "theta_coherence_eff": theta_coherence,
                "tau_novelty_eff": tau_novelty,
                "tau_coherence_eff": tau_coherence,
                "rare_expert_credit": self.rare_expert_credit.detach().clone(),
                "agent_utility": agent_utility,
                "agent_entropy": agent_entropy,
                "global_entropy": global_entropy,
                "gamma": self.clustering_gamma.detach().expand(batch),
                "phase_vectors": phase_vectors,
                "phase_alignment": self.mean_phase_alignment(phase_vectors),
                "entropy_force": entropy_force,
                "inverse_square_repulsion": inverse_square_repulsion,
                "global_entropy_force": global_entropy_force,
                "cubic_self_drive": cubic_self_drive,
                "physics_force": physics_force,
                "evidence": evidence,
                "expected": expected,
                "novelty": novelty,
                "coherence": coherence,
                "gate": gate,
                "trust": weights,
                "proposal": proposals,
                "integrated_delta": integrated_delta,
                "integrated_evidence": integrated_evidence,
                "prediction": prediction,
                "losses": losses,
                "metrics": metrics,
            }

        def compute_losses(
            self,
            *,
            prediction: torch.Tensor,
            target: torch.Tensor,
            z3: torch.Tensor,
            z3_next: torch.Tensor,
            agents: torch.Tensor,
            z_next: torch.Tensor,
            evidence: torch.Tensor,
            coherence: torch.Tensor,
            novelty: torch.Tensor,
            gate: torch.Tensor,
            gate_soft: torch.Tensor,
            update_vector: torch.Tensor,
            physics_force: torch.Tensor,
            entropy_force: torch.Tensor,
            inverse_square_repulsion: torch.Tensor,
            global_entropy_force: torch.Tensor,
            cubic_self_drive: torch.Tensor,
            residual_delta: torch.Tensor,
            novelty_pressure: torch.Tensor,
            exploratory_gate: torch.Tensor,
            boot_target: torch.Tensor,
            boot_target_z3_only: torch.Tensor,
            theta_novelty_eff: torch.Tensor,
            theta_coherence_eff: torch.Tensor,
            tau_novelty_eff: torch.Tensor,
            tau_coherence_eff: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """Compute the anti-collapse training objective."""
            cfg = self.config
            predictive = F.mse_loss(prediction, target)
            mean_coherence = coherence.mean()
            coherence_band = (
                F.relu(torch.tensor(cfg.coherence_min, device=coherence.device) - mean_coherence).pow(2)
                + F.relu(mean_coherence - torch.tensor(cfg.coherence_max, device=coherence.device)).pow(2)
            )
            pairwise_distance = self.mean_agent_pairwise_distance(z_next)
            diversity = F.relu(torch.tensor(cfg.diversity_min, device=z_next.device) - pairwise_distance).pow(2)
            evidence_variance_value = evidence.var(dim=1, unbiased=False).mean()
            evidence_variance = F.relu(
                torch.tensor(cfg.evidence_variance_min, device=evidence.device) - evidence_variance_value
            ).pow(2)
            stability = torch.mean((z3_next - z3).pow(2))
            effort = torch.mean((z_next - agents).pow(2)) + 0.1 * torch.mean(update_vector.pow(2)) + 0.05 * torch.mean(physics_force.pow(2))
            useful_novelty = torch.mean(exploratory_gate * coherence.clamp_min(cfg.coherence_floor) * novelty)
            residual_activity = torch.norm(residual_delta, dim=-1).mean() + novelty_pressure.mean()
            phi_weights = self.phi / self.phi.sum().clamp_min(cfg.epsilon)
            phi_concentration = torch.sum(phi_weights.pow(2))
            phi_concentration_limit = torch.tensor(
                cfg.phi_concentration_ratio_max / cfg.agent_count,
                device=phi_concentration.device,
                dtype=phi_concentration.dtype,
            )
            phi_balance = F.relu(phi_concentration - phi_concentration_limit).pow(2)
            phi_effective_agents = 1.0 / phi_concentration.clamp_min(cfg.epsilon)
            boot_context_sensitivity = torch.norm(boot_target - boot_target_z3_only, dim=-1).mean()
            boot_context = F.relu(
                torch.tensor(cfg.boot_context_sensitivity_min, device=boot_target.device, dtype=boot_target.dtype)
                - boot_context_sensitivity
            ).pow(2)
            if boot_target.shape[0] > 1:
                boot_variance_value = boot_target.var(dim=0, unbiased=False).mean()
            else:
                boot_variance_value = boot_target.new_tensor(cfg.boot_variance_min)
            boot_variance = F.relu(
                torch.tensor(cfg.boot_variance_min, device=boot_target.device, dtype=boot_target.dtype) - boot_variance_value
            ).pow(2)
            gate_rate = gate_soft.mean()
            gate_rate_band = (
                F.relu(torch.tensor(cfg.gate_rate_min, device=gate_soft.device, dtype=gate_soft.dtype) - gate_rate).pow(2)
                + F.relu(gate_rate - torch.tensor(cfg.gate_rate_max, device=gate_soft.device, dtype=gate_soft.dtype)).pow(2)
            )
            threshold_variance = theta_novelty_eff.var(unbiased=False) + theta_coherence_eff.var(unbiased=False)
            total = (
                cfg.beta_predictive * predictive
                + cfg.beta_coherence_band * coherence_band
                + cfg.beta_diversity * diversity
                + cfg.beta_evidence_variance * evidence_variance
                + cfg.beta_stability * stability
                + cfg.beta_effort * effort
                + cfg.beta_phi_balance * phi_balance
                + cfg.beta_boot_context * boot_context
                + cfg.beta_boot_variance * boot_variance
                + cfg.beta_gate_rate * gate_rate_band
                - cfg.beta_useful_novelty * useful_novelty
            )
            return {
                "total": total,
                "predictive": predictive,
                "coherence_band": coherence_band,
                "diversity": diversity,
                "evidence_variance": evidence_variance,
                "stability": stability,
                "effort": effort,
                "useful_novelty": useful_novelty,
                "residual_activity": residual_activity.detach(),
                "phi_balance": phi_balance,
                "phi_concentration": phi_concentration.detach(),
                "phi_effective_agents": phi_effective_agents.detach(),
                "boot_context": boot_context,
                "boot_context_sensitivity": boot_context_sensitivity.detach(),
                "boot_variance": boot_variance,
                "raw_boot_variance": boot_variance_value.detach(),
                "gate_rate_band": gate_rate_band,
                "gate_rate": gate_rate.detach(),
                "mean_theta_novelty": theta_novelty_eff.mean().detach(),
                "mean_theta_coherence": theta_coherence_eff.mean().detach(),
                "mean_tau_novelty": tau_novelty_eff.mean().detach(),
                "mean_tau_coherence": tau_coherence_eff.mean().detach(),
                "threshold_variance": threshold_variance.detach(),
                "mean_pairwise_distance": pairwise_distance.detach(),
                "raw_evidence_variance": evidence_variance_value.detach(),
            }

        def train_step(
            self,
            optimizer: torch.optim.Optimizer,
            x: torch.Tensor,
            *,
            target: Optional[torch.Tensor] = None,
            update_recurrent_state: bool = True,
            clip_grad_norm: Optional[float] = 1.0,
        ) -> Dict[str, float]:
            """Run one optimizer-backed training step and optionally advance persistent state."""
            self.train()
            optimizer.zero_grad(set_to_none=True)
            output = self.forward(x, target=target, update_state=False, add_noise=True)
            loss = output["losses"]["total"]
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
            optimizer.step()
            if update_recurrent_state:
                with torch.no_grad():
                    output = self.forward(x, target=target, update_state=True, add_noise=False)
            return self.metrics_to_dict(output["metrics"], output["losses"])

        def train_sequence_window(
            self,
            optimizer: torch.optim.Optimizer,
            embeddings: torch.Tensor,
            *,
            targets: Optional[torch.Tensor] = None,
            truncation_steps: int = 16,
            clip_grad_norm: Optional[float] = 1.0,
            commit_recurrent_state: bool = True,
            add_noise: bool = True,
        ) -> Dict[str, float]:
            """Train on a finite sequence window with explicit truncated BPTT."""
            if truncation_steps < 1:
                raise ValueError("truncation_steps must be >= 1")
            if embeddings.dim() == 2:
                sequence = embeddings.unsqueeze(0)
            elif embeddings.dim() == 3:
                sequence = embeddings
            else:
                raise ValueError(f"embeddings must have shape [steps, dim] or [batch, steps, dim], got {tuple(embeddings.shape)}")
            if sequence.shape[1] < 2:
                raise ValueError("sequence must contain at least two timesteps")
            if sequence.shape[-1] != self.config.input_dim:
                raise ValueError(f"embedding dimension must equal input_dim={self.config.input_dim}, got {sequence.shape[-1]}")

            if targets is None:
                target_sequence = sequence[:, 1:, :]
            else:
                if targets.dim() == 2:
                    target_sequence = targets.unsqueeze(0)
                elif targets.dim() == 3:
                    target_sequence = targets
                else:
                    raise ValueError(f"targets must have shape [steps, dim] or [batch, steps, dim], got {tuple(targets.shape)}")
                if target_sequence.shape[:2] != (sequence.shape[0], sequence.shape[1] - 1):
                    raise ValueError(
                        "targets must align with next-step pairs: expected "
                        f"[{sequence.shape[0]}, {sequence.shape[1] - 1}, {self.config.input_dim}], got {tuple(target_sequence.shape)}"
                    )

            self.train()
            batch = sequence.shape[0]
            device = sequence.device
            z3 = self._prepare_z3(batch, device, None).detach()
            agents = self._prepare_agents(batch, device, None).detach()
            chunk_metrics = []
            chunk_losses = []
            final_output: Optional[Dict[str, torch.Tensor]] = None

            for start in range(0, sequence.shape[1] - 1, truncation_steps):
                end = min(start + truncation_steps, sequence.shape[1] - 1)
                optimizer.zero_grad(set_to_none=True)
                total_loss = sequence.new_tensor(0.0)
                outputs = []
                for idx in range(start, end):
                    output = self.forward(
                        sequence[:, idx, :],
                        initial_z3=z3,
                        initial_agents=agents,
                        target=target_sequence[:, idx, :],
                        update_state=False,
                        add_noise=add_noise,
                    )
                    outputs.append(output)
                    total_loss = total_loss + output["losses"]["total"]
                    z3 = output["z3_after"]
                    agents = output["agents_after"]
                total_loss = total_loss / max(1, end - start)
                total_loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
                optimizer.step()

                final_output = outputs[-1]
                metrics = self.metrics_to_dict(final_output["metrics"], final_output["losses"])
                metrics["chunk_loss"] = float(total_loss.detach().cpu().item())
                metrics["chunk_start"] = float(start)
                metrics["chunk_end"] = float(end)
                chunk_metrics.append(metrics)
                chunk_losses.append(metrics["chunk_loss"])
                z3 = z3.detach()
                agents = agents.detach()

            if commit_recurrent_state and final_output is not None:
                with torch.no_grad():
                    self._commit_state(z3, agents, final_output["metrics"], final_output.get("agent_utility"))

            summary = dict(chunk_metrics[-1]) if chunk_metrics else {}
            if chunk_losses:
                summary["window_loss"] = float(sum(chunk_losses) / len(chunk_losses))
                summary["truncated_bptt_chunks"] = float(len(chunk_losses))
                summary["truncation_steps"] = float(truncation_steps)
            return summary

        @torch.no_grad()
        def step_runtime(self, x: torch.Tensor, *, hard_gate: bool = True) -> Dict[str, torch.Tensor]:
            """Advance the persistent recurrent state without optimizer updates."""
            self.eval()
            return self.forward(x, hard_gate=hard_gate, update_state=True, add_noise=False)

        def public_projection(self, output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
            """Project neural runtime metrics into compact public Z³ fields."""
            metrics = self.metrics_to_dict(output["metrics"], output["losses"])
            return {
                "z_cubed_state": {
                    "coherence": metrics["mean_coherence"],
                    "stability": max(0.0, min(1.0, 1.0 - metrics["z3_delta_norm"])),
                    "regime": self._regime(metrics["mean_coherence"], metrics["mean_novelty"]),
                    "neural_metrics": metrics,
                },
                "phi": metrics["mean_coherence"],
                "sigma": max(0.0, min(1.0, self.config.noise_scale + metrics["gate_entropy"])),
                "drift_vector": metrics["z3_delta_norm"],
                "learning": {
                    "z3_neural_loss": metrics["loss_total"],
                    "useful_novelty": metrics["useful_novelty"],
                    "agent_diversity": metrics["mean_pairwise_distance"],
                },
            }

        def save_checkpoint(self, path: str | Path) -> None:
            """Persist model parameters, config, recurrent state, and last metrics."""
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "config": asdict(self.config),
                    "state_dict": self.state_dict(),
                    "z3_state": self.z3_state.detach().cpu(),
                    "zprime_state": self.zprime_state.detach().cpu(),
                    "last_metrics": self.last_metrics.detach().cpu(),
                },
                path,
            )

        @classmethod
        def load_checkpoint(cls, path: str | Path, *, map_location: Optional[str] = None) -> "Z3NeuralDynamics":
            """Load a saved Z³ neural dynamics checkpoint."""
            payload = torch.load(path, map_location=map_location)
            model = cls(Z3Config(**payload["config"]))
            state_dict = dict(payload["state_dict"])
            if "last_metrics" in state_dict and state_dict["last_metrics"].numel() != model.metric_count():
                state_dict.pop("last_metrics")
            model.load_state_dict(state_dict, strict=False)
            device = next(model.parameters()).device
            model.z3_state = payload["z3_state"].to(device)
            model.zprime_state = payload["zprime_state"].to(device)
            if "last_metrics" in payload:
                loaded_metrics = payload["last_metrics"].to(device)
                target_count = model.metric_count()
                if loaded_metrics.numel() < target_count:
                    padded = torch.zeros(target_count, device=device, dtype=loaded_metrics.dtype)
                    padded[: loaded_metrics.numel()] = loaded_metrics.reshape(-1)
                    loaded_metrics = padded
                elif loaded_metrics.numel() > target_count:
                    loaded_metrics = loaded_metrics.reshape(-1)[:target_count]
                model.last_metrics = loaded_metrics
            return model

        def entropy_gradient_force(self, entropy: torch.Tensor, phase_vectors: torch.Tensor) -> torch.Tensor:
            """Approximate Σ_j (2γ - 1)(S_j - S_i)e^{iφ_ij} in real vector space."""
            cfg = self.config
            if entropy.numel() == 0 or entropy.shape[1] < 2:
                return torch.zeros_like(phase_vectors)
            entropy_delta = entropy.unsqueeze(1) - entropy.unsqueeze(2)  # S_j - S_i for each receiver i.
            phase_delta = phase_vectors.unsqueeze(2) - phase_vectors.unsqueeze(1)
            phase_direction = F.normalize(phase_delta, dim=-1, eps=cfg.epsilon)
            agent_count = entropy.shape[1]
            eye = torch.eye(agent_count, dtype=torch.bool, device=entropy.device).view(1, agent_count, agent_count)
            force = entropy_delta.unsqueeze(-1) * phase_direction
            force = force.masked_fill(eye.unsqueeze(-1), 0.0)
            return (2.0 * self.clustering_gamma - 1.0) * force.sum(dim=2) / max(agent_count - 1, 1)

        def inverse_square_repulsion_field(self, states: torch.Tensor, phase_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Short-range k/d² repulsion with safe clamping and optional phase direction."""
            cfg = self.config
            if states.numel() == 0 or states.shape[1] < 2:
                return torch.zeros_like(states)
            diff = states.unsqueeze(2) - states.unsqueeze(1)
            dist = torch.norm(diff, dim=-1).clamp_min(cfg.epsilon)
            direction = diff / dist.unsqueeze(-1)
            if phase_vectors is not None:
                phase_delta = phase_vectors.unsqueeze(2) - phase_vectors.unsqueeze(1)
                phase_direction = F.normalize(phase_delta, dim=-1, eps=cfg.epsilon)
                direction = F.normalize(direction + phase_direction, dim=-1, eps=cfg.epsilon)
            agent_count = states.shape[1]
            eye = torch.eye(agent_count, dtype=torch.bool, device=states.device).view(1, agent_count, agent_count)
            active = (dist < cfg.repulsion_radius) & (~eye)
            magnitude = active.to(states.dtype) / dist.pow(2).clamp_min(cfg.epsilon)
            magnitude = magnitude.clamp_max(1.0 / max(cfg.epsilon, 1e-3))
            repulsion = direction * magnitude.unsqueeze(-1)
            repulsion = repulsion.masked_fill(eye.unsqueeze(-1), 0.0)
            denom = active.sum(dim=2, keepdim=True).clamp_min(1).to(states.dtype)
            return repulsion.sum(dim=2) / denom

        def cubic_self_recursion(self, states: torch.Tensor) -> torch.Tensor:
            """Normalized cubic self-recursion term α((z/|z|)^3 + c)."""
            cfg = self.config
            unit = states / torch.norm(states, dim=-1, keepdim=True).clamp_min(cfg.epsilon)
            return unit.pow(3) + torch.tanh(self.cubic_bias).view(1, 1, -1)

        def clip_force(self, force: torch.Tensor, max_norm: float) -> torch.Tensor:
            """Clip force vectors by norm without altering direction."""
            if max_norm <= 0.0:
                return force
            norm = torch.norm(force, dim=-1, keepdim=True).clamp_min(self.config.epsilon)
            scale = (float(max_norm) / norm).clamp_max(1.0)
            return force * scale

        def mean_phase_alignment(self, phase_vectors: torch.Tensor) -> torch.Tensor:
            """Mean pairwise phase-direction cosine mapped to [0, 1]."""
            if phase_vectors.numel() == 0 or phase_vectors.shape[1] < 2:
                return phase_vectors.new_tensor(0.0)
            sim = torch.matmul(phase_vectors, phase_vectors.transpose(1, 2))
            agent_count = phase_vectors.shape[1]
            mask = torch.triu(torch.ones(agent_count, agent_count, dtype=torch.bool, device=phase_vectors.device), diagonal=1)
            return ((sim[:, mask].mean() + 1.0) * 0.5).detach()

        def pairwise_repulsion_field(self, states: torch.Tensor, top_k: Optional[int] = 3) -> torch.Tensor:
            """Push nearby Z-prime agents away from each other within each sample."""
            cfg = self.config
            if states.numel() == 0 or states.shape[1] < 2:
                return torch.zeros_like(states)
            diff = states.unsqueeze(2) - states.unsqueeze(1)
            dist = torch.norm(diff, dim=-1).clamp_min(cfg.epsilon)
            agent_count = states.shape[1]
            eye = torch.eye(agent_count, dtype=torch.bool, device=states.device).view(1, agent_count, agent_count)
            if top_k is not None and top_k < agent_count - 1:
                masked_dist = dist.masked_fill(eye, float("inf"))
                knn_idx = masked_dist.topk(k=top_k, largest=False, dim=-1).indices
                active = torch.zeros_like(dist, dtype=torch.bool)
                active.scatter_(-1, knn_idx, True)
                active = active & (~eye)
            else:
                active = (~eye) & (dist < cfg.repulsion_radius)
            close_pressure = F.relu(torch.tensor(cfg.repulsion_radius, device=states.device) - dist)
            direction = diff / dist.unsqueeze(-1)
            repulsion = direction * close_pressure.unsqueeze(-1).pow(cfg.repulsion_power)
            repulsion = repulsion.masked_fill(eye.unsqueeze(-1), 0.0)
            repulsion = repulsion.masked_fill((~active).unsqueeze(-1), 0.0)
            denom = active.sum(dim=2, keepdim=True).clamp_min(1).to(states.dtype)
            return repulsion.sum(dim=2) / denom

        def normalize_trust(self, trust: torch.Tensor) -> torch.Tensor:
            """Normalize agent trust with a small floor and uniform zero-mass fallback."""
            cfg = self.config
            floor = max(float(cfg.trust_floor), 0.0)
            if floor > 0.0:
                trust = trust + trust.new_full(trust.shape, floor)
            mass = trust.sum(dim=1, keepdim=True)
            normalized = trust / mass.clamp_min(cfg.epsilon)
            fallback = trust.new_full(trust.shape, 1.0 / cfg.agent_count)
            return torch.where(mass > cfg.epsilon, normalized, fallback)

        def mean_agent_pairwise_distance(self, states: torch.Tensor) -> torch.Tensor:
            """Return batch-local mean pairwise Z-prime diversity."""
            if states.numel() == 0 or states.shape[1] < 2:
                return states.new_tensor(0.0)
            distances = torch.cdist(states, states, p=2)
            agent_count = states.shape[1]
            mask = torch.triu(torch.ones(agent_count, agent_count, dtype=torch.bool, device=states.device), diagonal=1)
            return distances[:, mask].mean()

        def compute_metrics(
            self,
            coherence: torch.Tensor,
            novelty: torch.Tensor,
            gate: torch.Tensor,
            z3: torch.Tensor,
            z3_next: torch.Tensor,
            z_next: torch.Tensor,
            evidence: torch.Tensor,
            losses: Dict[str, torch.Tensor],
            *,
            agent_entropy: torch.Tensor,
            global_entropy: torch.Tensor,
            phase_vectors: torch.Tensor,
            entropy_force: torch.Tensor,
            inverse_square_repulsion: torch.Tensor,
            global_entropy_force: torch.Tensor,
            cubic_self_drive: torch.Tensor,
            physics_force: torch.Tensor,
        ) -> torch.Tensor:
            cfg = self.config
            gate_clamped = gate.clamp(cfg.epsilon, 1.0 - cfg.epsilon)
            gate_entropy = -(gate_clamped * gate_clamped.log() + (1.0 - gate_clamped) * (1.0 - gate_clamped).log()).mean()
            pairwise = self.mean_agent_pairwise_distance(z_next)
            return torch.stack(
                [
                    coherence.mean().detach(),
                    novelty.mean().detach(),
                    gate.mean().detach(),
                    torch.norm(z3_next - z3, dim=-1).mean().detach(),
                    pairwise.detach(),
                    evidence.var(dim=1, unbiased=False).mean().detach(),
                    gate_entropy.detach(),
                    losses["total"].detach(),
                    agent_entropy.mean().detach(),
                    global_entropy.mean().detach(),
                    self.clustering_gamma.detach(),
                    self.mean_phase_alignment(phase_vectors).detach(),
                    torch.norm(entropy_force, dim=-1).mean().detach(),
                    torch.norm(inverse_square_repulsion, dim=-1).mean().detach(),
                    torch.norm(global_entropy_force, dim=-1).mean().detach(),
                    torch.norm(cubic_self_drive, dim=-1).mean().detach(),
                    torch.norm(physics_force, dim=-1).mean().detach(),
                ]
            )

        def _prepare_z3(self, batch: int, device: torch.device, initial_z3: Optional[torch.Tensor]) -> torch.Tensor:
            cfg = self.config
            z3 = self.z3_state.to(device) if initial_z3 is None else initial_z3.to(device)
            if z3.dim() == 1:
                z3 = z3.unsqueeze(0).expand(batch, -1)
            if z3.shape != (batch, cfg.state_dim):
                raise ValueError(f"z3 must have shape [{batch}, {cfg.state_dim}], got {tuple(z3.shape)}")
            return z3

        def _prepare_agents(self, batch: int, device: torch.device, initial_agents: Optional[torch.Tensor]) -> torch.Tensor:
            cfg = self.config
            agents = self.zprime_state.to(device) if initial_agents is None else initial_agents.to(device)
            if agents.dim() == 2:
                agents = agents.unsqueeze(0).expand(batch, -1, -1)
            if agents.shape != (batch, cfg.agent_count, cfg.local_dim):
                raise ValueError(f"agents must have shape [{batch}, {cfg.agent_count}, {cfg.local_dim}], got {tuple(agents.shape)}")
            return agents

        def contextual_thresholds(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Return bounded context-adaptive novelty/coherence thresholds and temperatures."""
            cfg = self.config
            if not cfg.adaptive_thresholds:
                batch = context.shape[0]
                theta_novelty = context.new_full((batch, 1), cfg.theta_novelty)
                theta_coherence = context.new_full((batch, 1), cfg.theta_coherence)
                tau_novelty = context.new_full((batch, 1), cfg.tau_novelty)
                tau_coherence = context.new_full((batch, 1), cfg.tau_coherence)
                return theta_novelty, theta_coherence, tau_novelty, tau_coherence
            raw = torch.tanh(self.threshold_adapter(context))
            theta_novelty = (cfg.theta_novelty + cfg.threshold_delta_max * raw[:, 0:1]).clamp(0.0, 1.5)
            theta_coherence = (cfg.theta_coherence + cfg.threshold_delta_max * raw[:, 1:2]).clamp(0.0, 1.0)
            tau_novelty = (cfg.tau_novelty + cfg.tau_delta_max * raw[:, 2:3]).clamp_min(cfg.tau_min)
            tau_coherence = (cfg.tau_coherence + cfg.tau_delta_max * raw[:, 3:4]).clamp_min(cfg.tau_min)
            return theta_novelty, theta_coherence, tau_novelty, tau_coherence

        @torch.no_grad()
        def _commit_state(
            self,
            z3_next: torch.Tensor,
            z_next: torch.Tensor,
            metrics: torch.Tensor,
            agent_utility: Optional[torch.Tensor] = None,
        ) -> None:
            self.z3_state = z3_next.mean(dim=0).detach()
            self.zprime_state = z_next.mean(dim=0).detach()
            self.last_metrics = metrics.detach()
            if agent_utility is not None:
                cfg = self.config
                observed_credit = agent_utility.mean(dim=0).detach().to(self.rare_expert_credit.device)
                self.rare_expert_credit = (
                    cfg.rare_expert_decay * self.rare_expert_credit
                    + (1.0 - cfg.rare_expert_decay) * observed_credit
                )

        @staticmethod
        def metric_names() -> Tuple[str, ...]:
            return (
                "mean_coherence",
                "mean_novelty",
                "mean_gate",
                "z3_delta_norm",
                "mean_pairwise_distance",
                "evidence_variance",
                "gate_entropy",
                "loss_total",
                "mean_agent_entropy",
                "mean_global_entropy",
                "gamma",
                "phase_alignment",
                "entropy_force_norm",
                "inverse_square_repulsion_norm",
                "global_entropy_force_norm",
                "cubic_self_drive_norm",
                "physics_force_norm",
            )

        @classmethod
        def metric_count(cls) -> int:
            return len(cls.metric_names())

        @staticmethod
        def metrics_to_dict(metrics: torch.Tensor, losses: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
            names = Z3NeuralDynamics.metric_names()
            output = {name: float(value.detach().cpu().item()) for name, value in zip(names, metrics)}
            if losses:
                for key, value in losses.items():
                    if torch.is_tensor(value):
                        output[key if key != "total" else "loss_total"] = float(value.detach().cpu().item())
            return output

        @staticmethod
        def _regime(coherence: float, novelty: float) -> str:
            if coherence >= 0.70 and novelty >= 0.35:
                return "coherent_discovery"
            if coherence >= 0.70:
                return "stable_coherence"
            if novelty >= 0.55:
                return "volatile_novelty"
            return "watchful_recalibration"


    def prepare_embedding_pairs(embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert dense embedding streams into next-step training pairs."""
        if embeddings.dim() == 2:
            if embeddings.shape[0] < 2:
                raise ValueError("embedding stream must contain at least two steps")
            return embeddings[:-1], embeddings[1:]
        if embeddings.dim() == 3:
            if embeddings.shape[1] < 2:
                raise ValueError("batched embedding stream must contain at least two steps")
            input_dim = embeddings.shape[-1]
            x = embeddings[:, :-1, :].reshape(-1, input_dim)
            y = embeddings[:, 1:, :].reshape(-1, input_dim)
            return x, y
        raise ValueError(f"embeddings must have shape [steps, dim] or [batch, steps, dim], got {tuple(embeddings.shape)}")


    def generate_regime_sequence(
        steps: int,
        input_dim: int,
        *,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a deterministic toy sequence for smoke tests only."""
        device = device or torch.device("cpu")
        t = torch.linspace(0.0, 1.0, steps, device=device).unsqueeze(1)
        frequencies = torch.linspace(1.0, 3.5, input_dim, device=device).unsqueeze(0)
        base = torch.sin(2.0 * torch.pi * t * frequencies)
        modulation = torch.cos(2.0 * torch.pi * t * (frequencies + 0.5))
        sequence = 0.65 * base + 0.35 * modulation
        sequence = sequence.unsqueeze(0).repeat(batch_size, 1, 1)
        sequence = sequence + 0.02 * torch.randn_like(sequence)
        return prepare_embedding_pairs(sequence)


    def smoke_train(
        steps: int = 120,
        *,
        output_dir: str | Path = "outputs/z3_neural_dynamics",
        seed: int = 7,
    ) -> Dict[str, Any]:
        """Run a small trainability smoke test and save checkpoint/metrics."""
        torch.manual_seed(seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = Z3NeuralDynamics()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        x, y = generate_regime_sequence(max(steps + 1, 4), model.config.input_dim, batch_size=4)

        history = []
        for idx in range(steps):
            start = idx * 4
            end = start + 4
            xb = x[start:end]
            yb = y[start:end]
            metrics = model.train_step(optimizer, xb, target=yb, update_recurrent_state=True)
            history.append(metrics)

        model.save_checkpoint(output_dir / "z3_neural_dynamics.pt")
        with (output_dir / "z3_neural_dynamics_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        return {"model": model, "history": history, "output_dir": str(output_dir)}

else:

    class Z3NeuralDynamics:  # type: ignore[no-redef]
        """Placeholder explaining the missing PyTorch dependency."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise ModuleNotFoundError(
                "Z3NeuralDynamics requires PyTorch. Install an environment-appropriate torch package before use."
            ) from _TORCH_IMPORT_ERROR


    def prepare_embedding_pairs(*_: Any, **__: Any) -> Tuple[Any, Any]:
        raise ModuleNotFoundError("prepare_embedding_pairs requires PyTorch.") from _TORCH_IMPORT_ERROR


    def generate_regime_sequence(*_: Any, **__: Any) -> Tuple[Any, Any]:
        raise ModuleNotFoundError("generate_regime_sequence requires PyTorch.") from _TORCH_IMPORT_ERROR


    def smoke_train(*_: Any, **__: Any) -> Dict[str, Any]:
        raise ModuleNotFoundError("smoke_train requires PyTorch.") from _TORCH_IMPORT_ERROR
