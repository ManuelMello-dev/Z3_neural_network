"""
Z³ Differentiable Articulatory Synthesizer Layer
==================================================

This module replaces the discrete token decoder with a continuous, physically
grounded vocal tract model. Instead of predicting the next character, Z³
predicts the next **geometry of the vocal tract** — and words emerge as a
byproduct of the changing physical configuration.

Biological Grounding
--------------------
A baby does not predict tokens. It:
  1. Entrains its neural oscillators to the phase/frequency of incoming speech.
  2. Learns to control the geometry of its vocal tract (a physical waveguide).
  3. Hears the result, computes the error against what it heard, and updates.

This module implements that loop in differentiable PyTorch:

  Z³ prediction_input [136-dim]
        │
        ▼
  ArticulatoryControlLayer  →  16 continuous geometric parameters
        │
        ▼
  DifferentiableFormantSynth  →  waveform [batch, samples]
        │
        ▼
  SpectralLoss  →  scalar loss (backpropagates through Z³)

The 16 Articulatory Parameters
-------------------------------
These map directly to Z³'s 16 `input_dim` nodes, making the connection
between Z³'s internal state and vocal geometry explicit:

  Node  0  — F0: Fundamental frequency / pitch (Hz, 80–400)
  Node  1  — Voicing: Voiced (1.0) vs. unvoiced (0.0) probability
  Node  2  — F1: First formant frequency (Hz, 200–1000) — jaw/tongue height
  Node  3  — F1_bw: F1 bandwidth (Hz, 40–200)
  Node  4  — F2: Second formant frequency (Hz, 700–2500) — tongue front/back
  Node  5  — F2_bw: F2 bandwidth (Hz, 40–250)
  Node  6  — F3: Third formant frequency (Hz, 1800–3500) — lip rounding
  Node  7  — F3_bw: F3 bandwidth (Hz, 60–300)
  Node  8  — F4: Fourth formant frequency (Hz, 2800–4500) — pharynx shape
  Node  9  — F4_bw: F4 bandwidth (Hz, 80–400)
  Node 10  — Nasality: Nasal coupling (0.0–1.0)
  Node 11  — Aspiration: Breathiness / glottal noise (0.0–1.0)
  Node 12  — Lip aperture: Mouth opening (0.0–1.0)
  Node 13  — Tongue body: Front-to-back position (0.0–1.0)
  Node 14  — Glottal pressure: Subglottal pressure / loudness (0.0–1.0)
  Node 15  — Articulation rate: Speed of movement (0.0–1.0)

Differentiable Formant Synthesis
---------------------------------
Each formant is modeled as a second-order resonator (digital bandpass filter)
in the z-domain. The transfer function is:

    H_k(z) = 1 / (1 - 2·r_k·cos(2π·F_k/fs)·z⁻¹ + r_k²·z⁻²)

where r_k = exp(-π·BW_k/fs) is the pole radius derived from the bandwidth.

Because we need gradients to flow through the filter parameters, we implement
the resonators as a time-domain recurrence in PyTorch rather than using
scipy.signal (which is not differentiable). The recurrence is:

    y[n] = x[n] + 2·r·cos(ω)·y[n-1] - r²·y[n-2]

This is fully differentiable with respect to F_k and BW_k.

Spectral Loss
-------------
The loss is the mean squared error between the log-magnitude spectra of the
synthesized waveform and a target waveform (or target spectral envelope).
Log-magnitude MSE is perceptually motivated — it corresponds to how the
auditory system processes sound on a logarithmic scale.

    L = MSE(log|STFT(y_synth)|, log|STFT(y_target)|)

This loss backpropagates through the STFT (via torch.stft), through the
formant resonators, and into Z³'s prediction_input — closing the acoustic
feedback loop.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ModuleNotFoundError as exc:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_OK = False
    _TORCH_ERROR = exc


# ---------------------------------------------------------------------------
# Articulatory parameter ranges (min, max, default)
# These define the physical bounds of each vocal tract parameter.
# ---------------------------------------------------------------------------

ARTICULATORY_PARAMS: List[Dict[str, Any]] = [
    {"name": "F0",              "min": 80.0,   "max": 400.0,  "default": 120.0},
    {"name": "voicing",         "min": 0.0,    "max": 1.0,    "default": 0.8},
    {"name": "F1",              "min": 200.0,  "max": 1000.0, "default": 500.0},
    {"name": "F1_bw",           "min": 40.0,   "max": 200.0,  "default": 80.0},
    {"name": "F2",              "min": 700.0,  "max": 2500.0, "default": 1500.0},
    {"name": "F2_bw",           "min": 40.0,   "max": 250.0,  "default": 100.0},
    {"name": "F3",              "min": 1800.0, "max": 3500.0, "default": 2500.0},
    {"name": "F3_bw",           "min": 60.0,   "max": 300.0,  "default": 120.0},
    {"name": "F4",              "min": 2800.0, "max": 4500.0, "default": 3500.0},
    {"name": "F4_bw",           "min": 80.0,   "max": 400.0,  "default": 160.0},
    {"name": "nasality",        "min": 0.0,    "max": 1.0,    "default": 0.1},
    {"name": "aspiration",      "min": 0.0,    "max": 1.0,    "default": 0.05},
    {"name": "lip_aperture",    "min": 0.0,    "max": 1.0,    "default": 0.5},
    {"name": "tongue_body",     "min": 0.0,    "max": 1.0,    "default": 0.5},
    {"name": "glottal_pressure","min": 0.0,    "max": 1.0,    "default": 0.7},
    {"name": "articulation_rate","min": 0.0,   "max": 1.0,    "default": 0.5},
]

N_ARTICULATORY = len(ARTICULATORY_PARAMS)  # 16


def _param_ranges() -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Return (min_vals, max_vals) tensors of shape [16]."""
    mins = torch.tensor([p["min"] for p in ARTICULATORY_PARAMS], dtype=torch.float32)
    maxs = torch.tensor([p["max"] for p in ARTICULATORY_PARAMS], dtype=torch.float32)
    return mins, maxs


def _param_defaults() -> "torch.Tensor":
    """Return default articulatory parameter tensor of shape [16]."""
    return torch.tensor([p["default"] for p in ARTICULATORY_PARAMS], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Differentiable second-order formant resonator
# ---------------------------------------------------------------------------

def differentiable_formant_resonator(
    source: "torch.Tensor",
    freq_hz: "torch.Tensor",
    bandwidth_hz: "torch.Tensor",
    sample_rate: int = 22050,
) -> "torch.Tensor":
    """Apply a differentiable second-order bandpass resonator to a source signal.

    Implements the recurrence:
        y[n] = x[n] + 2·r·cos(ω)·y[n-1] - r²·y[n-2]

    where:
        ω = 2π·F/fs
        r = exp(-π·BW/fs)

    This is fully differentiable with respect to freq_hz and bandwidth_hz,
    allowing gradients to flow back through the filter parameters into Z³.

    Parameters
    ----------
    source : Tensor [batch, n_samples]
        The glottal source signal (voiced + unvoiced mixture).
    freq_hz : Tensor [batch]
        Formant centre frequency in Hz.
    bandwidth_hz : Tensor [batch]
        Formant bandwidth in Hz.
    sample_rate : int
        Audio sample rate in Hz.

    Returns
    -------
    Tensor [batch, n_samples]
        The filtered (resonated) signal.
    """
    batch, n_samples = source.shape
    device = source.device
    fs = float(sample_rate)

    # Pole parameters — differentiable w.r.t. freq_hz and bandwidth_hz
    omega = 2.0 * math.pi * freq_hz / fs          # [batch]
    r = torch.exp(-math.pi * bandwidth_hz / fs)    # [batch]

    coeff_a1 = 2.0 * r * torch.cos(omega)          # [batch]
    coeff_a2 = -(r ** 2)                            # [batch]

    # Time-domain recurrence — loop over samples
    # We use a Python loop here. For long sequences this is slow, but it is
    # fully differentiable. For production, this can be replaced with a
    # parallel scan (e.g., associative scan) for efficiency.
    y = torch.zeros(batch, n_samples, device=device, dtype=source.dtype)
    y_prev1 = torch.zeros(batch, device=device, dtype=source.dtype)
    y_prev2 = torch.zeros(batch, device=device, dtype=source.dtype)

    for n in range(n_samples):
        y_n = source[:, n] + coeff_a1 * y_prev1 + coeff_a2 * y_prev2
        y[:, n] = y_n
        y_prev2 = y_prev1
        y_prev1 = y_n

    return y


# ---------------------------------------------------------------------------
# Differentiable Formant Synthesizer
# ---------------------------------------------------------------------------

if _TORCH_OK:

    class DifferentiableFormantSynth(nn.Module):
        """Differentiable vocal tract formant synthesizer.

        Takes 16 articulatory parameters and produces a waveform by:
          1. Generating a glottal source (voiced pulse train + unvoiced noise).
          2. Passing the source through 4 differentiable formant resonators.
          3. Mixing in aspiration noise modulated by the nasality parameter.
          4. Scaling by glottal pressure.

        The entire pipeline is differentiable with respect to the 16 input
        parameters, so gradients flow back through the synthesizer into Z³.

        Parameters
        ----------
        sample_rate : int
            Audio sample rate. Default 22050 (half of CD quality — sufficient
            for speech and keeps computation manageable).
        n_samples : int
            Number of audio samples per forward pass (one Z³ tick).
            Default 2205 = 100ms at 22050 Hz.
        """

        def __init__(
            self,
            sample_rate: int = 22050,
            n_samples: int = 2205,
        ) -> None:
            super().__init__()
            self.sample_rate = sample_rate
            self.n_samples = n_samples

            # Learnable phase offset for the glottal pulse train
            # Allows Z³ to learn the phase relationship between its internal
            # oscillators and the acoustic output
            self.glottal_phase_offset = nn.Parameter(torch.zeros(1))

            # Register parameter range tensors as buffers (not trainable)
            mins, maxs = _param_ranges()
            self.register_buffer("param_mins", mins)
            self.register_buffer("param_maxs", maxs)

        def _denormalize(self, params_norm: "torch.Tensor") -> "torch.Tensor":
            """Map sigmoid-normalized params [0,1] → physical units."""
            return self.param_mins + params_norm * (self.param_maxs - self.param_mins)

        def _glottal_source(
            self,
            f0: "torch.Tensor",
            voicing: "torch.Tensor",
            aspiration: "torch.Tensor",
        ) -> "torch.Tensor":
            """Generate the glottal source signal.

            Combines a voiced component (sawtooth-like pulse train at F0)
            with an unvoiced component (bandlimited noise), mixed by voicing.

            Parameters
            ----------
            f0 : Tensor [batch] — fundamental frequency in Hz
            voicing : Tensor [batch] — voiced probability [0, 1]
            aspiration : Tensor [batch] — breathiness [0, 1]

            Returns
            -------
            Tensor [batch, n_samples]
            """
            batch = f0.shape[0]
            device = f0.device
            t = torch.arange(self.n_samples, device=device, dtype=torch.float32)

            # Voiced component: sum of harmonics (Liljencrants-Fant approximation
            # using a simple sawtooth for differentiability)
            # Phase: 2π·F0·t/fs + learnable offset
            phase = (
                2.0 * math.pi
                * f0.unsqueeze(1)                          # [batch, 1]
                * t.unsqueeze(0)                           # [1, n_samples]
                / self.sample_rate
                + self.glottal_phase_offset
            )
            # Sawtooth via sum of first 6 harmonics (differentiable Fourier series)
            voiced = torch.zeros(batch, self.n_samples, device=device)
            for k in range(1, 7):
                voiced = voiced + ((-1) ** (k + 1)) / k * torch.sin(k * phase)
            voiced = voiced * (2.0 / math.pi)  # normalize to [-1, 1]

            # Unvoiced component: filtered noise
            noise = torch.randn(batch, self.n_samples, device=device) * 0.3

            # Mix voiced and unvoiced by voicing probability
            v = voicing.unsqueeze(1)          # [batch, 1]
            asp = aspiration.unsqueeze(1)     # [batch, 1]
            source = v * voiced + (1.0 - v) * noise + asp * noise * 0.5
            return source

        def forward(
            self,
            articulatory_params: "torch.Tensor",
        ) -> Dict[str, "torch.Tensor"]:
            """Synthesize a waveform from articulatory parameters.

            Parameters
            ----------
            articulatory_params : Tensor [batch, 16]
                Normalized articulatory parameters in [0, 1].
                These are the raw outputs of the ArticulatoryControlLayer
                after sigmoid activation.

            Returns
            -------
            dict with:
              waveform : Tensor [batch, n_samples] — synthesized audio
              params_physical : Tensor [batch, 16] — parameters in physical units
              formant_outputs : list of Tensor [batch, n_samples] — per-formant signals
            """
            batch = articulatory_params.shape[0]
            device = articulatory_params.device

            # Denormalize to physical units
            p = self._denormalize(articulatory_params)  # [batch, 16]

            # Extract parameters
            f0       = p[:, 0]   # fundamental frequency
            voicing  = p[:, 1]   # voicing probability (already in [0,1] after denorm)
            f1       = p[:, 2]   # F1 frequency
            f1_bw    = p[:, 3]   # F1 bandwidth
            f2       = p[:, 4]   # F2 frequency
            f2_bw    = p[:, 5]   # F2 bandwidth
            f3       = p[:, 6]   # F3 frequency
            f3_bw    = p[:, 7]   # F3 bandwidth
            f4       = p[:, 8]   # F4 frequency
            f4_bw    = p[:, 9]   # F4 bandwidth
            nasality = p[:, 10]  # nasal coupling
            aspiration = p[:, 11]  # breathiness
            # lip_aperture = p[:, 12]  # modulates F1 (higher aperture → lower F1)
            # tongue_body  = p[:, 13]  # modulates F2
            glottal_pressure = p[:, 14]  # overall amplitude
            # articulation_rate = p[:, 15]  # future: modulate transition speed

            # Lip aperture modulates F1: more open mouth → lower F1
            lip_mod = p[:, 12]  # [0,1]
            f1_modulated = f1 * (1.0 - 0.15 * lip_mod)

            # Tongue body modulates F2: tongue front → higher F2
            tongue_mod = p[:, 13]  # [0,1]
            f2_modulated = f2 * (0.85 + 0.30 * tongue_mod)

            # Generate glottal source
            source = self._glottal_source(f0, voicing, aspiration)

            # Pass through 4 formant resonators in series
            out1 = differentiable_formant_resonator(source,      f1_modulated, f1_bw, self.sample_rate)
            out2 = differentiable_formant_resonator(out1,         f2_modulated, f2_bw, self.sample_rate)
            out3 = differentiable_formant_resonator(out2,         f3,           f3_bw, self.sample_rate)
            out4 = differentiable_formant_resonator(out3,         f4,           f4_bw, self.sample_rate)

            # Add nasal resonance (simplified: blend in a low-frequency copy)
            nasal_source = differentiable_formant_resonator(
                source,
                torch.full_like(f0, 250.0),   # nasal formant ~250 Hz
                torch.full_like(f1_bw, 100.0),
                self.sample_rate,
            )
            waveform = out4 + nasality.unsqueeze(1) * nasal_source * 0.3

            # Scale by glottal pressure (amplitude envelope)
            waveform = waveform * glottal_pressure.unsqueeze(1)

            # Normalize to prevent clipping
            max_val = waveform.abs().max(dim=1, keepdim=True).values.clamp_min(1e-6)
            waveform = waveform / max_val * 0.9

            return {
                "waveform": waveform,
                "params_physical": p,
                "formant_outputs": [out1, out2, out3, out4],
            }


    # ---------------------------------------------------------------------------
    # Articulatory Control Layer (frame-level, single snapshot)
    # ---------------------------------------------------------------------------

    class ArticulatoryControlLayer(nn.Module):
        """Maps Z³'s prediction_input to 16 articulatory control parameters.

        This is the bridge between Z³'s internal state and the vocal tract
        geometry. It replaces the token decoder head.

        Input:  prediction_input [batch, evidence_dim + state_dim + context_dim]
                = [batch, 24 + 64 + 48] = [batch, 136]
        Output: articulatory_params [batch, 16] in [0, 1] (sigmoid-normalized)

        The output is then fed into DifferentiableFormantSynth to produce audio.

        Parameters
        ----------
        input_dim : int
            Dimension of Z³'s prediction_input (default 136).
        hidden_dim : int
            Hidden layer dimension (default 128).
        """

        def __init__(
            self,
            input_dim: int = 136,
            hidden_dim: int = 128,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.n_params = N_ARTICULATORY

            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.n_params),
                nn.Sigmoid(),  # output in [0, 1] — maps to physical ranges in synth
            )

            # Initialize to produce near-default articulatory parameters
            # so the system starts with a reasonable vocal configuration
            defaults = _param_defaults()
            mins, maxs = _param_ranges()
            default_norm = (defaults - mins) / (maxs - mins).clamp_min(1e-6)
            # Bias the final layer toward defaults
            with torch.no_grad():
                self.net[-2].bias.data = torch.log(
                    default_norm.clamp(0.01, 0.99) /
                    (1.0 - default_norm.clamp(0.01, 0.99))
                )

        def forward(self, prediction_input: "torch.Tensor") -> "torch.Tensor":
            """Map Z³ state → articulatory parameters [batch, 16] in [0,1]."""
            return self.net(prediction_input)


    # ---------------------------------------------------------------------------
    # Trajectory Control Layer (coarticulation-aware, multi-step)
    # ---------------------------------------------------------------------------

    class TrajectoryControlLayer(nn.Module):
        """Maps Z³ state + resonant memory context to a trajectory of articulatory
        parameters across multiple future ticks.

        This is the coarticulation-aware upgrade to ArticulatoryControlLayer.
        Instead of producing a single 16-parameter snapshot, it outputs a short
        sequence of parameter vectors — one per future tick — so the vocal tract
        is already moving toward the next configuration before the current one
        is complete.

        The memory context encodes the coarticulatory pull: which prior vocal
        configurations are resonating with the current state, and how strongly
        they are attracting the trajectory forward. This maps directly to Z³'s
        resonant memory geometry — the memory rings carry the coarticulatory
        trajectory context via phase alignment and reconstruction confidence.

        Input
        -----
        prediction_input : Tensor [batch, z3_pred_dim]
            Z³'s concatenated [integrated_evidence, z3_next, context].
        memory_context : Tensor [batch, memory_dim], optional
            Compact encoding of the resonant memory state. Constructed from:
              - reconstruction_confidence (scalar)
              - salience (scalar)
              - phase_position (scalar)
              - top_match resonance scores (up to 5 scalars)
              - top_match phase_alignment scores (up to 5 scalars)
            Total: 13 scalars, zero-padded if fewer matches available.
            If None, the layer operates in frame-level mode (same as
            ArticulatoryControlLayer).

        Output
        ------
        Tensor [batch, trajectory_steps, 16]
            A sequence of articulatory parameter vectors in [0, 1], one per
            future tick. The first step (index 0) is the current frame.
            Subsequent steps represent the predicted trajectory of the vocal
            tract geometry over the next (trajectory_steps - 1) ticks.

        Parameters
        ----------
        z3_pred_dim : int
            Dimension of Z³'s prediction_input (default 136).
        memory_dim : int
            Dimension of the memory context vector (default 13).
        hidden_dim : int
            Hidden layer dimension (default 256).
        trajectory_steps : int
            Number of future ticks to predict (default 4 = 400ms at 100ms/tick).
        """

        MEMORY_DIM: int = 13  # reconstruction_confidence + salience + phase_position + 5 resonances + 5 phase_alignments

        def __init__(
            self,
            z3_pred_dim: int = 136,
            memory_dim: int = 13,
            hidden_dim: int = 256,
            trajectory_steps: int = 4,
        ) -> None:
            super().__init__()
            self.z3_pred_dim = z3_pred_dim
            self.memory_dim = memory_dim
            self.hidden_dim = hidden_dim
            self.trajectory_steps = trajectory_steps
            self.n_params = N_ARTICULATORY

            # Shared encoder: fuses Z³ state + memory context
            fused_dim = z3_pred_dim + memory_dim
            self.encoder = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )

            # Trajectory decoder: unrolls the encoded state into a sequence
            # Each step attends to the shared encoding via a small GRU cell
            # so earlier steps in the trajectory influence later ones
            # (the tongue's current position constrains where it can be next)
            self.trajectory_gru = nn.GRUCell(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
            )

            # Per-step output projection: hidden → 16 articulatory params
            self.output_proj = nn.Linear(hidden_dim, self.n_params)

            # Initialize output bias toward default articulatory configuration
            defaults = _param_defaults()
            mins, maxs = _param_ranges()
            default_norm = (defaults - mins) / (maxs - mins).clamp_min(1e-6)
            with torch.no_grad():
                self.output_proj.bias.data = torch.log(
                    default_norm.clamp(0.01, 0.99) /
                    (1.0 - default_norm.clamp(0.01, 0.99))
                )

        @staticmethod
        def encode_memory_context(
            memory_output: Dict[str, Any],
            device: Optional["torch.device"] = None,
        ) -> "torch.Tensor":
            """Convert a resonant memory observe() output dict to a context tensor.

            Extracts the 13 scalars that encode the coarticulatory pull:
              [reconstruction_confidence, salience, phase_position,
               resonance_0..4, phase_alignment_0..4]

            Parameters
            ----------
            memory_output : dict
                The dict returned by ResonantMemoryGeometry.observe().
            device : torch.device, optional
                Target device for the output tensor.

            Returns
            -------
            Tensor [1, 13] — memory context vector.
            """
            vec = [
                float(memory_output.get("reconstruction_confidence", 0.0)),
                float(memory_output.get("salience", 0.0)),
                float(memory_output.get("phase_position", 0.0)),
            ]
            top_matches = memory_output.get("top_matches") or []
            for i in range(5):
                if i < len(top_matches):
                    vec.append(float(top_matches[i].get("resonance", 0.0)))
                else:
                    vec.append(0.0)
            for i in range(5):
                if i < len(top_matches):
                    vec.append(float(top_matches[i].get("phase_alignment", 0.0)))
                else:
                    vec.append(0.0)
            t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)  # [1, 13]
            if device is not None:
                t = t.to(device)
            return t

        def forward(
            self,
            prediction_input: "torch.Tensor",
            memory_context: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            """Produce a trajectory of articulatory parameters.

            Parameters
            ----------
            prediction_input : Tensor [batch, z3_pred_dim]
            memory_context : Tensor [batch, memory_dim], optional
                If None, uses a zero context vector (frame-level fallback).

            Returns
            -------
            Tensor [batch, trajectory_steps, 16] in [0, 1]
            """
            batch = prediction_input.shape[0]
            device = prediction_input.device

            if memory_context is None:
                memory_context = torch.zeros(batch, self.memory_dim, device=device)
            elif memory_context.shape[0] == 1 and batch > 1:
                memory_context = memory_context.expand(batch, -1)

            # Fuse Z³ state with memory context
            fused = torch.cat([prediction_input, memory_context], dim=-1)  # [batch, fused_dim]
            encoded = self.encoder(fused)  # [batch, hidden_dim]

            # Unroll trajectory via GRU
            # The GRU cell carries the biomechanical constraint: the vocal tract
            # cannot teleport — each step is conditioned on the previous one
            h = encoded  # initial hidden state = current Z³ encoding
            trajectory = []
            for _ in range(self.trajectory_steps):
                h = self.trajectory_gru(encoded, h)  # [batch, hidden_dim]
                params = torch.sigmoid(self.output_proj(h))  # [batch, 16]
                trajectory.append(params)

            return torch.stack(trajectory, dim=1)  # [batch, trajectory_steps, 16]


    # ---------------------------------------------------------------------------
    # Spectral Loss
    # ---------------------------------------------------------------------------

    class SpectralLoss(nn.Module):
        """Log-magnitude spectral MSE loss for differentiable acoustic training.

        Computes the mean squared error between the log-magnitude STFT of the
        synthesized waveform and a target waveform (or target spectrum).

        This is the perceptually motivated loss that closes the acoustic
        feedback loop: Z³ hears the error between what it produced and what
        it was trying to produce.

        Parameters
        ----------
        n_fft : int
            FFT size. Default 512.
        hop_length : int
            STFT hop length. Default 128.
        win_length : int
            STFT window length. Default 512.
        """

        def __init__(
            self,
            n_fft: int = 512,
            hop_length: int = 128,
            win_length: int = 512,
        ) -> None:
            super().__init__()
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.win_length = win_length
            window = torch.hann_window(win_length)
            self.register_buffer("window", window)

        def _log_stft_magnitude(self, waveform: "torch.Tensor") -> "torch.Tensor":
            """Compute fully vectorized log-magnitude STFT of a waveform [batch, n_samples].

            torch.stft natively supports 2D [batch, n_samples] input, so we
            eliminate the sequential Python loop entirely. This allows the GPU
            to process the entire batch matrix concurrently and lets spectral
            gradients flow backward at full GPU acceleration.
            """
            # Ensure window is on the correct device and dtype (buffer may be
            # on CPU if the model was moved to GPU after init)
            window = self.window
            if window.device != waveform.device or window.dtype != waveform.dtype:
                window = window.to(device=waveform.device, dtype=waveform.dtype)

            spec = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=True,       # perfect time-alignment across frames
                normalized=False,
                onesided=True,     # drop redundant symmetric negative frequencies
                return_complex=True,
            )  # → [batch, freq_bins, time_frames]

            magnitude = spec.abs()
            return torch.log(magnitude.clamp_min(1e-5))

        def forward(
            self,
            synthesized: "torch.Tensor",
            target: "torch.Tensor",
        ) -> "torch.Tensor":
            """Compute spectral MSE loss.

            Parameters
            ----------
            synthesized : Tensor [batch, n_samples]
                The waveform produced by DifferentiableFormantSynth.
            target : Tensor [batch, n_samples]
                The target waveform (e.g., from the audio input encoder,
                or from the existing FM synth output as a bootstrap target).

            Returns
            -------
            Tensor scalar — the spectral loss.
            """
            # Pad or trim to same length
            min_len = min(synthesized.shape[1], target.shape[1])
            s = synthesized[:, :min_len]
            t = target[:, :min_len]

            log_spec_s = self._log_stft_magnitude(s)
            log_spec_t = self._log_stft_magnitude(t)

            return F.mse_loss(log_spec_s, log_spec_t)


    # ---------------------------------------------------------------------------
    # Full Articulatory Voice Module (combines all three components)
    # ---------------------------------------------------------------------------

    class Z3ArticulatoryVoice(nn.Module):
        """Complete differentiable vocal tract module for Z³.

        Combines:
          - TrajectoryControlLayer: Z³ state + memory context → trajectory of
            16 vocal tract parameter vectors (coarticulation-aware)
          - DifferentiableFormantSynth: parameters → waveform per step
          - SpectralLoss: waveform vs. target → scalar loss for backprop

        The TrajectoryControlLayer uses a GRU to unroll the Z³ state across
        ``trajectory_steps`` future ticks, conditioned on the resonant memory
        context. This models coarticulation: the vocal tract is already moving
        toward the next configuration before the current one is complete.

        Parameters
        ----------
        z3_prediction_dim : int
            Dimension of Z³'s prediction_input tensor (default 136).
        sample_rate : int
            Audio sample rate. Default 22050 Hz.
        n_samples : int
            Samples per Z³ tick. Default 2205 (100ms at 22050 Hz).
        trajectory_steps : int
            Number of future ticks to predict. Default 4 (400ms lookahead).
        """

        def __init__(
            self,
            z3_prediction_dim: int = 136,
            sample_rate: int = 22050,
            n_samples: int = 2205,
            trajectory_steps: int = 4,
        ) -> None:
            super().__init__()
            self.trajectory_steps = trajectory_steps
            self.trajectory_ctrl = TrajectoryControlLayer(
                z3_pred_dim=z3_prediction_dim,
                memory_dim=TrajectoryControlLayer.MEMORY_DIM,
                hidden_dim=256,
                trajectory_steps=trajectory_steps,
            )
            # Keep frame-level control for backward compatibility / inference
            self.control = ArticulatoryControlLayer(
                input_dim=z3_prediction_dim,
                hidden_dim=128,
            )
            self.synth = DifferentiableFormantSynth(
                sample_rate=sample_rate,
                n_samples=n_samples,
            )
            self.loss_fn = SpectralLoss()

        def forward(
            self,
            prediction_input: "torch.Tensor",
            target_waveform: Optional["torch.Tensor"] = None,
            memory_context: Optional["torch.Tensor"] = None,
            use_trajectory: bool = True,
        ) -> Dict[str, Any]:
            """Full forward pass: Z³ state → articulatory trajectory → waveform → loss.

            Parameters
            ----------
            prediction_input : Tensor [batch, 136]
                Z³'s concatenated [integrated_evidence, z3_next, context].
            target_waveform : Tensor [batch, n_samples * trajectory_steps], optional
                Target audio for computing the spectral loss over the full
                trajectory window. If None, no loss is computed.
            memory_context : Tensor [batch, 13], optional
                Resonant memory context from ResonantMemoryGeometry.observe().
                Encodes the coarticulatory pull from prior vocal configurations.
                If None, operates in frame-level mode.
            use_trajectory : bool
                If True (default), uses TrajectoryControlLayer for multi-step
                output. If False, uses the frame-level ArticulatoryControlLayer.

            Returns
            -------
            dict with:
              articulatory_params : Tensor [batch, trajectory_steps, 16] or [batch, 16]
              params_physical : Tensor [batch, 16] — first step in physical units
              waveform : Tensor [batch, n_samples * trajectory_steps] — full trajectory audio
              loss : Tensor scalar (only if target_waveform provided)
              param_names : list[str]
              trajectory_steps : int
            """
            if use_trajectory:
                # Trajectory mode: output [batch, trajectory_steps, 16]
                art_traj = self.trajectory_ctrl(prediction_input, memory_context)
                # [batch, trajectory_steps, 16]

                # Synthesize each step and concatenate waveforms
                waveforms = []
                for step in range(self.trajectory_steps):
                    step_params = art_traj[:, step, :]  # [batch, 16]
                    step_out = self.synth(step_params)
                    waveforms.append(step_out["waveform"])

                full_waveform = torch.cat(waveforms, dim=1)  # [batch, n_samples * trajectory_steps]
                params_physical = self.synth._denormalize(art_traj[:, 0, :])  # first step

                result: Dict[str, Any] = {
                    "articulatory_params": art_traj,
                    "params_physical": params_physical,
                    "waveform": full_waveform,
                    "param_names": [p["name"] for p in ARTICULATORY_PARAMS],
                    "trajectory_steps": self.trajectory_steps,
                    "mode": "trajectory",
                }
            else:
                # Frame-level fallback
                art_params = self.control(prediction_input)
                synth_out = self.synth(art_params)
                result = {
                    "articulatory_params": art_params,
                    "params_physical": synth_out["params_physical"],
                    "waveform": synth_out["waveform"],
                    "param_names": [p["name"] for p in ARTICULATORY_PARAMS],
                    "trajectory_steps": 1,
                    "mode": "frame",
                }

            # Spectral loss against target (closes the acoustic feedback loop)
            if target_waveform is not None:
                loss = self.loss_fn(result["waveform"], target_waveform)
                result["loss"] = loss

            return result

        def synthesize(
            self,
            prediction_input: "torch.Tensor",
            memory_context: Optional["torch.Tensor"] = None,
            use_trajectory: bool = True,
        ) -> bytes:
            """Inference-only: return waveform as raw 16-bit PCM bytes.

            Parameters
            ----------
            prediction_input : Tensor [batch, 136]
            memory_context : Tensor [batch, 13], optional
            use_trajectory : bool
                If True, synthesizes the full trajectory window.

            Returns
            -------
            bytes — raw 16-bit PCM at self.synth.sample_rate
            """
            import numpy as np
            self.eval()
            with torch.no_grad():
                out = self.forward(
                    prediction_input,
                    memory_context=memory_context,
                    use_trajectory=use_trajectory,
                )
            waveform = out["waveform"].squeeze(0).cpu().numpy()
            pcm = (waveform * 32767.0).astype(np.int16)
            return pcm.tobytes()

        def articulatory_state(
            self,
            prediction_input: "torch.Tensor",
            memory_context: Optional["torch.Tensor"] = None,
        ) -> Dict[str, float]:
            """Return the current vocal tract geometry as a human-readable dict.

            Returns the first step of the trajectory (current frame) in
            physical units. Subsequent trajectory steps represent the predicted
            coarticulatory movement toward the next configuration.
            """
            self.eval()
            with torch.no_grad():
                art_traj = self.trajectory_ctrl(prediction_input, memory_context)
                # Use first step of trajectory as current state
                p_phys = self.synth._denormalize(art_traj[:, 0, :]).squeeze(0)
            return {
                param["name"]: round(float(p_phys[i].item()), 4)
                for i, param in enumerate(ARTICULATORY_PARAMS)
            }

        def articulatory_state_broadcast(
            self,
            prediction_input: "torch.Tensor",
            memory_context: Optional["torch.Tensor"] = None,
        ) -> Dict[str, Any]:
            """Return a WebSocket-ready payload for the dashboard vocal tract canvas.

            Broadcasts the current vocal tract geometry and coarticulatory
            trajectory every 100ms tick. The front-end can:
              - Draw a live cross-section of the throat (current frame)
              - Animate the predicted trajectory of tongue/lip movement
              - Visualise the spectral envelope from formant frequencies

            The payload includes:
              - ``type``: ``"articulatory_update"`` for the WS message router
              - ``params``: dict of physical parameter values (current frame)
              - ``trajectory``: list of formant_hz arrays for each future step
              - ``formant_hz``: [F1, F2, F3, F4] current frame
              - ``voicing``, ``pitch_hz``, ``lip_aperture``, ``tongue_body``,
                ``nasality``, ``glottal_pressure``: current frame scalars
              - ``memory_resonance``: reconstruction confidence from memory rings
            """
            self.eval()
            with torch.no_grad():
                art_traj = self.trajectory_ctrl(prediction_input, memory_context)
                # [1, trajectory_steps, 16]
                traj_physical = self.synth._denormalize(
                    art_traj.view(-1, N_ARTICULATORY)
                ).view(art_traj.shape[0], self.trajectory_steps, N_ARTICULATORY)

            current = {}
            for i, param in enumerate(ARTICULATORY_PARAMS):
                current[param["name"]] = round(float(traj_physical[0, 0, i].item()), 4)

            # Trajectory preview: formant positions for each future step
            trajectory_preview = []
            for step in range(self.trajectory_steps):
                step_formants = [
                    round(float(traj_physical[0, step, 2].item()), 2),  # F1
                    round(float(traj_physical[0, step, 4].item()), 2),  # F2
                    round(float(traj_physical[0, step, 6].item()), 2),  # F3
                    round(float(traj_physical[0, step, 8].item()), 2),  # F4
                ]
                trajectory_preview.append(step_formants)

            memory_resonance = 0.0
            if memory_context is not None:
                memory_resonance = round(float(memory_context[0, 0].item()), 4)  # reconstruction_confidence

            return {
                "type": "articulatory_update",
                "params": current,
                "trajectory": trajectory_preview,
                "formant_hz": trajectory_preview[0],
                "voicing": current["voicing"],
                "pitch_hz": current["F0"],
                "lip_aperture": current["lip_aperture"],
                "tongue_body": current["tongue_body"],
                "nasality": current["nasality"],
                "glottal_pressure": current["glottal_pressure"],
                "memory_resonance": memory_resonance,
                "trajectory_steps": self.trajectory_steps,
            }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_VOICE: Optional["Z3ArticulatoryVoice"] = None


def get_articulatory_voice(
    z3_prediction_dim: int = 136,
    sample_rate: int = 22050,
    n_samples: int = 2205,
) -> "Z3ArticulatoryVoice":
    """Return the module-level Z3ArticulatoryVoice singleton."""
    global _VOICE
    if not _TORCH_OK:
        raise ModuleNotFoundError("PyTorch is required for Z3ArticulatoryVoice")
    if _VOICE is None:
        _VOICE = Z3ArticulatoryVoice(
            z3_prediction_dim=z3_prediction_dim,
            sample_rate=sample_rate,
            n_samples=n_samples,
        )
    return _VOICE
