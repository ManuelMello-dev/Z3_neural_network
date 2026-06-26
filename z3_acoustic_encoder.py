"""Z³ Acoustic Input Encoder: Analysis by Synthesis Feedback Loop.

This module provides the differentiable acoustic encoder that maps raw
audio waveforms into Z³'s 16-dimensional input space, enabling the
network to entrain its internal phase oscillators to external sound.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class Z3AcousticEncoder(nn.Module):
    """Maps raw audio waveforms to Z³'s 16D input space."""

    def __init__(
        self,
        input_dim: int = 16,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Mel-scale filterbank (not learnable, but differentiable)
        from torchaudio.transforms import MelScale
        self.mel_scale = MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=80.0,
            f_max=8000.0,
            n_stft=n_fft // 2 + 1,
        )

        # Neural mapping from spectral features to Z³ input space
        # 16 mel-magnitudes + 16 mean phase angles = 32D input
        self.mapping = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, input_dim),
            nn.Tanh(), # Bound to [-1, 1] for Z³ stability
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode raw audio into Z³ input vector.

        Parameters
        ----------
        waveform : Tensor [batch, n_samples]
            Raw audio samples.

        Returns
        -------
        Tensor [batch, input_dim]
            Z³-compatible input vector.
        """
        # 1. STFT
        # [batch, freq_bins, time_frames, complex]
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(waveform.device),
            return_complex=True,
            center=True,
        )

        # 2. Magnitude and Phase
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)

        # 3. Mel-scale magnitudes
        # [batch, n_mels, time_frames]
        mel_spec = self.mel_scale(magnitude)
        
        # Mean across time frames to get a single vector for the 100ms window
        mel_vec = torch.mean(mel_spec, dim=-1) # [batch, n_mels]
        
        # Log-scale for better dynamic range
        mel_vec = torch.log1p(mel_vec)

        # 4. Mean phase angles
        # [batch, freq_bins, time_frames] -> [batch, freq_bins]
        # We simplify by taking the mean phase of the most energetic bins
        # or mapping the phase spectrum to mel-bins as well.
        # For now, we take the mean phase across time frames.
        phase_vec = torch.mean(phase, dim=-1) # [batch, freq_bins]
        
        # Map high-res phase spectrum to 16 mel-like bins
        # (Using the same mel-scale logic for simplicity)
        phase_mel = self.mel_scale(torch.abs(stft).detach() * phase).div(
            mel_spec.detach() + 1e-8
        )
        phase_mel_vec = torch.mean(phase_mel, dim=-1) # [batch, n_mels]

        # 5. Combine and Map
        features = torch.cat([mel_vec, phase_mel_vec], dim=-1) # [batch, 32]
        z3_input = self.mapping(features) # [batch, 16]

        return z3_input

# Singleton instance
_ENCODER = None

def get_acoustic_encoder(input_dim: int = 16) -> Z3AcousticEncoder:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = Z3AcousticEncoder(input_dim=input_dim)
    return _ENCODER
