From __future__ import annotations

Import json
From dataclasses import asdict, dataclass
From pathlib import Path
From typing import Any, Dict, Optional, Tuple

Try:
    Import torch
    Import torch.nn as nn
    Import torch.nn.functional as F
Except ModuleNotFoundError as exc:
    Torch = None
    Nn = None
    F = None
    TORCH_IMPORT_ERROR = exc
Else:
    TORCH_IMPORT_ERROR = None


@dataclass
Class Z3Config:
    Input_dim: int = 16
    Context_dim: int = 48
    State_dim: int = 64
    Local_dim: int = 32
    Evidence_dim: int = 24
    Hidden_dim: int = 128
    Agent_count: int = 8
    Agent_embed_dim: int = 12
    Step_size: float = 0.15
    Alpha_update: float = 0.05
    Alpha_decay: float = 0.002
    Noise_scale: float = 0.01
    Diversity_strength: float = 0.05
    Repulsion_radius: float = 0.35
    Repulsion_power: float = 1.0
    Lambda_coherence: float = 1.15
    Trust_floor: float = 1e-4
    Theta_novelty: float = 0.35
    Theta_coherence: float = 0.30
    Tau_novelty: float = 0.12
    Tau_coherence: float = 0.10
    Epsilon: float = 1e-6
    Coherence_min: float = 0.35
    Coherence_max: float = 0.92
    Diversity_min: float = 0.35
    Evidence_variance_min: float = 0.03
    Beta_predictive: float = 1.0
    Beta_coherence_band: float = 0.25
    Beta_diversity: float = 0.20
    Beta_evidence_variance: float = 0.10
    Beta_stability: float = 0.10
    Beta_effort: float = 0.01
    Beta_useful_novelty: float = 0.05


Class MLP(nn.Module):
    Def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, final_tanh: bool = False):
        Super().__init__()
        Layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        ]
        If final_tanh:
            Layers.append(nn.Tanh())
        Self.net = nn.Sequential(*layers)

    Def forward(self, x: torch.Tensor) -> torch.Tensor:
        Return self.net(x)


Class Z3NeuralDynamics(nn.Module):
    Def __init__(self, config: Optional[Z3Config] = None):
        Super().__init__()
        If torch is None:
            Raise ModuleNotFoundError(“Z3NeuralDynamics requires PyTorch.”) from TORCH_IMPORT_ERROR

        Self.config = config or Z3Config()
        C = self.config

        Self.context_encoder = MLP(c.input_dim, c.context_dim, c.hidden_dim)
        Self.boot_projection = MLP(c.state_dim + c.context_dim, c.local_dim, c.hidden_dim)
        Self.agent_embeddings = nn.Embedding(c.agent_count, c.agent_embed_dim)
        Self.agent_transition = MLP(
            c.local_dim + c.local_dim + c.context_dim + c.agent_embed_dim + c.state_dim,
            c.local_dim,
            c.hidden_dim,
            final_tanh=True,
        )
        Self.evidence_projection = MLP(
            c.local_dim + c.context_dim + c.agent_embed_dim,
            c.evidence_dim,
            c.hidden_dim,
        )
        Self.expected_evidence = MLP(
            c.state_dim + c.context_dim + c.agent_embed_dim,
            c.evidence_dim,
            c.hidden_dim,
        )
        Self.gamma = MLP(
            c.evidence_dim + c.state_dim + c.context_dim,
            c.state_dim,
            c.hidden_dim,
            final_tanh=True,
        )
        Self.prediction_head = MLP(
            c.evidence_dim + c.state_dim + c.context_dim,
            c.input_dim,
            c.hidden_dim,
        )

        Self.raw_phi = nn.Parameter(torch.log(torch.expm1(torch.ones(c.agent_count))))
        Self.register_buffer(“z3_state”, torch.zeros(c.state_dim))
        Self.register_buffer(“zprime_state”, torch.empty(0))
        Self.register_buffer(“last_metrics”, torch.zeros(self.metric_count()))
        Self.reset_state()

    @property
    Def phi(self) -> torch.Tensor:
        Return F.softplus(self.raw_phi) + 1e-4

    Def reset_state(self, seed: Optional[int] = None) -> None:
        If seed is not None:
            Torch.manual_seed(seed)

        C = self.config
        Device = next(self.parameters()).device
        Self.z3_state = torch.randn(c.state_dim, device=device) * 0.02

        With torch.no_grad():
            Z3 = self.z3_state.unsqueeze(0)
            Zero_context = torch.zeros(1, c.context_dim, device=device)
            Target = self.boot_projection(torch.cat([z3, zero_context], dim=-1)).squeeze(0)
            States = target.unsqueeze(0).repeat(c.agent_count, 1)
            States = states + torch.randn_like(states) * c.noise_scale
            Self.zprime_state = states.detach()

    Def _prepare_z3(self, batch: int, initial_z3: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        C = self.config
        Z3 = self.z3_state if initial_z3 is None else initial_z3.to(device)
        If z3.dim() == 1:
            Z3 = z3.unsqueeze(0).expand(batch, -1)
        If z3.shape != (batch, c.state_dim):
            Raise ValueError(f”z3 must have shape {(batch, c.state_dim)}, got {tuple(z3.shape)}”)
        Return z3

    Def _prepare_agents(self, batch: int, initial_agents: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        C = self.config
        Agents = self.zprime_state if initial_agents is None else initial_agents.to(device)
        If agents.dim() == 2:
            Agents = agents.unsqueeze(0).expand(batch, -1, -1)
        If agents.shape != (batch, c.agent_count, c.local_dim):
            Raise ValueError(f”agents must have shape {(batch, c.agent_count, c.local_dim)}, got {tuple(agents.shape)}”)
        Return agents

    Def pairwise_repulsion_field(self, states: torch.Tensor, top_k: int = 3) -> torch.Tensor:
        C = self.config
        If states.numel() == 0 or states.shape[1] < 2:
            Return torch.zeros_like(states)

        Diff = states.unsqueeze(2) – states.unsqueeze(1)
        Dist = torch.norm(diff, dim=-1).clamp_min(c.epsilon)
        Agent_count = states.shape[1]
        Eye = torch.eye(agent_count, dtype=torch.bool, device=states.device).view(1, agent_count, agent_count)

        If top_k is not None and top_k < agent_count – 1:
            Masked_dist = dist.masked_fill(eye, float(“inf”))
            Knn_idx = masked_dist.topk(k=top_k, largest=False, dim=-1).indices
            Active = torch.zeros_like(dist, dtype=torch.bool)
            Active.scatter_(-1, knn_idx, True)
            Active = active & (~eye)
        Else:
            Active = (~eye) & (dist < c.repulsion_radius)

        Close_pressure = F.relu(torch.tensor(c.repulsion_radius, device=states.device) – dist)
        Direction = diff / dist.unsqueeze(-1)
        Repulsion = direction * close_pressure.unsqueeze(-1).pow(c.repulsion_power)
        Repulsion = repulsion.masked_fill(eye.unsqueeze(-1), 0.0)
        Repulsion = repulsion.masked_fill((~active).unsqueeze(-1), 0.0)

        Denom = active.sum(dim=2, keepdim=True).clamp_min(1).to(states.dtype)
        Return repulsion.sum(dim=2) / denom

    Def normalize_trust(self, trust: torch.Tensor) -> torch.Tensor:
        C = self.config
        Floor = max(float(c.trust_floor), 0.0)
        If floor > 0.0:
            Trust = trust + trust.new_full(trust.shape, floor)
        Mass = trust.sum(dim=1, keepdim=True)
        Normalized = trust / mass.clamp_min(c.epsilon)
        Fallback = trust.new_full(trust.shape, 1.0 / c.agent_count)
        Return torch.where(mass > c.epsilon, normalized, fallback)

    Def mean_agent_pairwise_distance(self, states: torch.Tensor) -> torch.Tensor:
        If states.numel() == 0 or states.shape[1] < 2:
            Return states.new_tensor(0.0)
        Distances = torch.cdist(states, states, p=2)
        Agent_count = states.shape[1]
        Mask = torch.triu(torch.ones(agent_count, agent_count, dtype=torch.bool, device=states.device), diagonal=1)
        Return distances[:, mask].mean()

    Def compute_metrics(
        Self,
        Coherence: torch.Tensor,
        Novelty: torch.Tensor,
        Gate: torch.Tensor,
        Z3: torch.Tensor,
        Z3_next: torch.Tensor,
        Z_next: torch.Tensor,
        Evidence: torch.Tensor,
        Losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        Gate_clamped = gate.clamp(self.config.epsilon, 1.0 – self.config.epsilon)
        Gate_entropy = -(gate_clamped * gate_clamped.log() + (1.0 – gate_clamped) * (1.0 – gate_clamped).log()).mean()
        Pairwise = self.mean_agent_pairwise_distance(z_next)
        Return torch.stack([
            Coherence.mean().detach(),
            Novelty.mean().detach(),
            Gate.mean().detach(),
            Torch.norm(z3_next – z3, dim=-1).mean().detach(),
            Pairwise.detach(),
            Evidence.var(dim=1, unbiased=False).mean().detach(),
            Gate_entropy.detach(),
            Losses[“total”].detach(),
        ])

    Def forward(
        Self,
        X: torch.Tensor,
        Initial_z3: Optional[torch.Tensor] = None,
        Initial_agents: Optional[torch.Tensor] = None,
        Target: Optional[torch.Tensor] = None,
        Hard_gate: bool = False,
        Update_state: bool = False,
        Add_noise: bool = True,
    ) -> Dict[str, Any]:
        If x.dim() == 1:
            X = x.unsqueeze(0)

        C = self.config
        Device = x.device
        Batch = x.shape[0]

        Context = self.context_encoder(x)
        Z3 = self._prepare_z3(batch, initial_z3, device)
        Agents = self._prepare_agents(batch, initial_agents, device)

        Agent_ids = torch.arange(c.agent_count, device=device)
        Agent_embed = self.agent_embeddings(agent_ids).unsqueeze(0).expand(batch, -1, -1)

        Z3_context = z3.unsqueeze(1).expand(-1, c.agent_count, -1)
        Context_agents = context.unsqueeze(1).expand(-1, c.agent_count, -1)

        Boot_target = self.boot_projection(torch.cat([z3, context], dim=-1))
        Target_agents = boot_target.unsqueeze(1).expand(-1, c.agent_count, -1)

        Attraction = target_agents – agents
        Transition_input = torch.cat([agents, target_agents, context_agents, agent_embed, z3_context], dim=-1)
        Learned_transition = self.agent_transition(transition_input)
        Diversity_field = self.pairwise_repulsion_field(agents)

        Update_vector = (
            Self.phi.view(1, c.agent_count, 1) * attraction
            + learned_transition
            + c.diversity_strength * diversity_field
        )

        If add_noise and c.noise_scale > 0.0:
            Update_vector = update_vector + torch.randn_like(update_vector) * c.noise_scale

        Z_next = agents + c.step_size * update_vector

        Evidence_input = torch.cat([z_next, context_agents, agent_embed], dim=-1)
        Evidence = self.evidence_projection(evidence_input)

        Expected_input = torch.cat([z3_context, context_agents, agent_embed], dim=-1)
        Expected = self.expected_evidence(expected_input)

        Novelty = torch.norm(evidence – expected, dim=-1)
        Distance = torch.norm(z_next – target_agents, dim=-1)
        Coherence = torch.exp(-c.lambda_coherence * distance)

        Gate_soft = torch.sigmoid((novelty – c.theta_novelty) / c.tau_novelty) * torch.sigmoid(
            (coherence – c.theta_coherence) / c.tau_coherence
        )
        Gate = (gate_soft > 0.5).float() if hard_gate else gate_soft

        Trust = self.normalize_trust(gate * self.phi.view(1, -1) * coherence)

        Gamma_input = torch.cat([evidence, z3_context, context_agents], dim=-1)
        Proposals = self.gamma(gamma_input)
        Integrated_delta = torch.sum(proposals * trust.unsqueeze(-1), dim=1)
        Z3_next = (1.0 – c.alpha_decay) * z3 + c.alpha_update * integrated_delta

        Prediction_input = torch.cat(
            [
                Evidence * trust.unsqueeze(-1),
                Z3_next.unsqueeze(1).expand(-1, c.agent_count, -1),
                Context_agents,
            ],
            Dim=-1,
        )
        Prediction = self.prediction_head(prediction_input.mean(dim=1))

        Target_x = x if target is None else target
        Diversity_distance = self.mean_agent_pairwise_distance(z_next)

        Losses = {
            “predictive”: F.mse_loss(prediction, target_x),
            “coherence_band”: (
                F.relu(torch.tensor(c.coherence_min, device=device) – coherence.mean()).pow(2)
                + F.relu(coherence.mean() – torch.tensor(c.coherence_max, device=device)).pow(2)
            ),
            “diversity”: F.relu(torch.tensor(c.diversity_min, device=device) – diversity_distance).pow(2),
            “evidence_variance”: F.relu(
                Torch.tensor(c.evidence_variance_min, device=device) – evidence.var(dim=1, unbiased=False).mean()
            ).pow(2),
            “stability”: torch.mean((z3_next – z3).pow(2)),
            “effort”: torch.mean(update_vector.pow(2)) * 0.1,
            “useful_novelty”: -torch.mean(gate * coherence * novelty),
        }

        Losses[“total”] = (
            c.beta_predictive * losses[“predictive”]
            + c.beta_coherence_band * losses[“coherence_band”]
            + c.beta_diversity * losses[“diversity”]
            + c.beta_evidence_variance * losses[“evidence_variance”]
            + c.beta_stability * losses[“stability”]
            + c.beta_effort * losses[“effort”]
            + c.beta_useful_novelty * losses[“useful_novelty”]
        )

        Metrics = self.compute_metrics(coherence, novelty, gate, z3, z3_next, z_next, evidence, losses)

        If update_state:
            Self.z3_state = z3_next.mean(dim=0).detach()
            Self.zprime_state = z_next.mean(dim=0).detach()
            Self.last_metrics = metrics.detach()

        Return {
            “z3_before”: z3,
            “z3_after”: z3_next,
            “agents_before”: agents,
            “agents_after”: z_next,
            “context”: context,
            “boot_target”: boot_target,
            “evidence”: evidence,
            “expected”: expected,
            “novelty”: novelty,
            “coherence”: coherence,
            “gate”: gate,
            “trust”: trust,
            “proposal”: proposals,
            “integrated_delta”: integrated_delta,
            “prediction”: prediction,
            “losses”: losses,
            “metrics”: metrics,
        }

    Def train_step(
        Self,
        Optimizer: torch.optim.Optimizer,
        X: torch.Tensor,
        Target: Optional[torch.Tensor] = None,
        Update_recurrent_state: bool = True,
        Clip_grad_norm: Optional[float] = 1.0,
    ) -> Dict[str, float]:
        Self.train()
        Optimizer.zero_grad(set_to_none=True)
        Output = self.forward(x, target=target, update_state=False, add_noise=True)
        Loss = output[“losses”][“total”]
        Loss.backward()
        If clip_grad_norm is not None:
            Torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
        Optimizer.step()

        If update_recurrent_state:
            With torch.no_grad():
                Committed = self.forward(x, target=target, update_state=True, add_noise=False)
            Output = committed

        Return self.metrics_to_dict(output[“metrics”], output[“losses”])

    Def train_sequence_window(
        Self,
        Optimizer: torch.optim.Optimizer,
        Embeddings: torch.Tensor,
        *,
        Targets: Optional[torch.Tensor] = None,
        Truncation_steps: int = 16,
        Clip_grad_norm: Optional[float] = 1.0,
        Commit_recurrent_state: bool = True,
        Add_noise: bool = True,
    ) -> Dict[str, float]:
        If truncation_steps < 1:
            Raise ValueError(“truncation_steps must be >= 1”)

        If embeddings.dim() == 2:
            Sequence = embeddings.unsqueeze(0)
        Elif embeddings.dim() == 3:
            Sequence = embeddings
        Else:
            Raise ValueError(
                F”embeddings must have shape [steps, dim] or [batch, steps, dim], got {tuple(embeddings.shape)}”
            )

        If sequence.shape[1] < 2:
            Raise ValueError(“sequence must contain at least two timesteps”)
        If sequence.shape[-1] != self.config.input_dim:
            Raise ValueError(
                F”embedding dimension must equal input_dim={self.config.input_dim}, got {sequence.shape[-1]}”
            )

        If targets is None:
            Target_sequence = sequence[:, 1:, :]
        Else:
            If targets.dim() == 2:
                Target_sequence = targets.unsqueeze(0)
            Elif targets.dim() == 3:
                Target_sequence = targets
            Else:
                Raise ValueError(
                    F”targets must have shape [steps, dim] or [batch, steps, dim], got {tuple(targets.shape)}”
                )
            If target_sequence.shape[:2] != (sequence.shape[0], sequence.shape[1] – 1):
                Raise ValueError(
                    “targets must align with next-step pairs: expected “
                    F”[{sequence.shape[0]}, {sequence.shape[1] – 1}, {self.config.input_dim}], got {tuple(target_sequence.shape)}”
                )

        Self.train()
        Batch = sequence.shape[0]
        Device = sequence.device
        Z3 = self._prepare_z3(batch, device, None).detach()
        Agents = self._prepare_agents(batch, device, None).detach()

        Chunk_metrics = []
        Chunk_losses = []
        Final_output: Optional[Dict[str, torch.Tensor]] = None

        For start in range(0, sequence.shape[1] – 1, truncation_steps):
            End = min(start + truncation_steps, sequence.shape[1] – 1)
            Optimizer.zero_grad(set_to_none=True)
            Total_loss = sequence.new_tensor(0.0)
            Outputs = []

            For idx in range(start, end):
                Output = self.forward(
                    Sequence[:, idx, :],
                    Initial_z3=z3,
                    Initial_agents=agents,
                    Target=target_sequence[:, idx, :],
                    Update_state=False,
                    Add_noise=add_noise,
                )
                Outputs.append(output)
                Total_loss = total_loss + output[“losses”][“total”]
                Z3 = output[“z3_after”]
                Agents = output[“agents_after”]

            Total_loss = total_loss / max(1, end – start)
            Total_loss.backward()

            If clip_grad_norm is not None:
                Torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)

            Optimizer.step()

            Final_output = outputs[-1]
            Metrics_dict = self.metrics_to_dict(final_output[“metrics”], final_output[“losses”])
            Metrics_dict[“chunk_loss”] = float(total_loss.detach().cpu().item())
            Metrics_dict[“chunk_start”] = float(start)
            Metrics_dict[“chunk_end”] = float(end)
            Chunk_metrics.append(metrics_dict)
            Chunk_losses.append(metrics_dict[“chunk_loss”])

            Z3 = z3.detach()
            Agents = agents.detach()

        If commit_recurrent_state and final_output is not None:
            With torch.no_grad():
                Self._commit_state(z3, agents, final_output[“metrics”] if final_output is not None else self.last_metrics)

        Summary = dict(chunk_metrics[-1]) if chunk_metrics else {}
        If chunk_losses:
            Summary[“window_loss”] = float(sum(chunk_losses) / len(chunk_losses))
            Summary[“truncated_bptt_chunks”] = float(len(chunk_losses))
            Summary[“truncation_steps”] = float(truncation_steps)
        Return summary

    @torch.no_grad()
    Def step_runtime(self, x: torch.Tensor, *, hard_gate: bool = True) -> Dict[str, torch.Tensor]:
        Self.eval()
        Return self.forward(x, hard_gate=hard_gate, update_state=True, add_noise=False)

    Def public_projection(self, output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        Metrics = self.metrics_to_dict(output[“metrics”], output[“losses”])
        Return {
            “z_cubed_state”: {
                “coherence”: metrics[“mean_coherence”],
                “stability”: max(0.0, min(1.0, 1.0 – metrics[“z3_delta_norm”])),
                “regime”: self._regime(metrics[“mean_coherence”], metrics[“mean_novelty”]),
                “neural_metrics”: metrics,
            },
            “phi”: metrics[“mean_coherence”],
            “sigma”: max(0.0, min(1.0, self.config.noise_scale + metrics[“gate_entropy”])),
            “drift_vector”: metrics[“z3_delta_norm”],
            “learning”: {
                “z3_neural_loss”: metrics[“loss_total”],
                “useful_novelty”: metrics[“useful_novelty”],
                “agent_diversity”: metrics[“mean_pairwise_distance”],
            },
        }

    @staticmethod
    Def metric_names() -> Tuple[str, …]:
        Return (
            “mean_coherence”,
            “mean_novelty”,
            “mean_gate”,
            “z3_delta_norm”,
            “mean_pairwise_distance”,
            “evidence_variance”,
            “gate_entropy”,
            “loss_total”,
        )

    @classmethod
    Def metric_count(cls) -> int:
        Return len(cls.metric_names())

    @staticmethod
    Def metrics_to_dict(metrics: torch.Tensor, losses: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        Names = Z3NeuralDynamics.metric_names()
        Output = {name: float(value.detach().cpu().item()) for name, value in zip(names, metrics)}
        If losses:
            For key, value in losses.items():
                If torch.is_tensor(value):
                    Output[key if key != “total” else “loss_total”] = float(value.detach().cpu().item())
        Return output

    @staticmethod
    Def _regime(coherence: float, novelty: float) -> str:
        If coherence > 0.70 and novelty > 0.35:
            Return “coherent_discovery”
        If coherence > 0.70:
            Return “stable_coherence”
        If novelty > 0.55:
            Return “volatile_novelty”
        Return “watchful_recalibration”

    Def save_checkpoint(self, path: str | Path) -> None:
        Path = Path(path)
        Path.parent.mkdir(parents=True, exist_ok=True)
        Torch.save(
            {
                “config”: asdict(self.config),
                “state_dict”: self.state_dict(),
                “z3_state”: self.z3_state.detach().cpu(),
                “zprime_state”: self.zprime_state.detach().cpu(),
                “last_metrics”: self.last_metrics.detach().cpu(),
            },
            Path,
        )

    @classmethod
    Def load_checkpoint(cls, path: str | Path, *, map_location: Optional[str] = None) -> “Z3NeuralDynamics”:
        Payload = torch.load(path, map_location=map_location)
        Model = cls(Z3Config(**payload[“config”]))
        Model.load_state_dict(payload[“state_dict”])
        Model.z3_state = payload[“z3_state”].to(next(model.parameters()).device)
        Model.zprime_state = payload[“zprime_state”].to(next(model.parameters()).device)
        Model.last_metrics = payload[“last_metrics”].to(next(model.parameters()).device)
        Return model

