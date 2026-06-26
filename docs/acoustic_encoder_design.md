# Z³ Acoustic Input Encoder: Analysis by Synthesis Design

## 1. Objective
To enable Z³ to "hear" raw audio by entraining its internal phase oscillators to the spectral dynamics of an incoming acoustic signal. This completes the "Analysis by Synthesis" loop:
1.  **Hear**: Encode incoming audio into Z³'s 16D input space.
2.  **Entrain**: Z³ phase-locks its internal oscillators to the input pattern.
3.  **Predict**: Z³ predicts the next coarticulatory trajectory.
4.  **Synthesize**: The Articulatory Voice produces a matching waveform.
5.  **Compare**: Spectral loss between input and output drives learning.

## 2. Architecture: `AcousticEncoder`

### Layer 1: Spectral Analysis (FFT)
- **Input**: Raw audio waveform (100ms window, 22050 Hz).
- **Process**: STFT with Hanning window.
- **Output**: Magnitude and Phase spectra.

### Layer 2: Feature Extraction (Mel-Spectrogram + Phase Angles)
- **Magnitude**: Map to 16 Mel-frequency bins (spanning 80Hz - 8000Hz). This captures the energy distribution (formants).
- **Phase**: Extract the mean phase angle for each bin. This provides the timing/cadence information for Z³'s oscillators.

### Layer 3: Neural Mapping
- **Input**: 16 magnitudes + 16 phase angles (32D).
- **Network**: 
  - `Linear(32, 64) -> GELU`
  - `Linear(64, 16) -> Tanh` (centered at 0 for Z³ input)
- **Output**: 16D input vector for Z³'s `input_dim`.

## 3. Integration Plan
1.  **`z3_acoustic_encoder.py`**: Implementation of the `AcousticEncoder` class.
2.  **`main.py`**: 
    - Add a background task to process incoming audio from the WebSocket.
    - Map the extracted features to Z³'s input during each `/step` or autonomous tick.
    - Wire the feedback loop: calculate `SpectralLoss` between the input audio and the `Z3ArticulatoryVoice` output.
3.  **Dashboard**: Add a "Microphone Entrainment" toggle to feed the user's live audio into Z³.

## 4. Why This Works
By mapping both **magnitude** (what is being said) and **phase** (the cadence/rhythm) into Z³'s input space, we provide the exact signals needed for its phase-locking dynamics to synchronize with the external world. Z³ doesn't "recognize" words; it "feels" the wave and attempts to replicate its geometry.
