"""Mel spectrogram extraction for MLX."""

import math
from typing import Optional

import mlx.core as mx
import numpy as np


def create_mel_filterbank(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> mx.array:
    """Create mel filterbank matrix.

    Args:
        n_fft: FFT size
        n_mels: Number of mel bands
        sample_rate: Audio sample rate
        f_min: Minimum frequency
        f_max: Maximum frequency (default: sample_rate / 2)

    Returns:
        Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    # Mel scale conversion functions
    def hz_to_mel(freq):
        return 2595.0 * math.log10(1.0 + freq / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    # Create mel points
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    # Convert to FFT bin indices
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    # Create filterbank
    filterbank = np.zeros((n_mels, n_freqs))

    for i in range(n_mels):
        f_left = hz_points[i]
        f_center = hz_points[i + 1]
        f_right = hz_points[i + 2]

        # Rising slope
        for j, f in enumerate(fft_freqs):
            if f_left <= f <= f_center:
                filterbank[i, j] = (f - f_left) / (f_center - f_left + 1e-10)
            elif f_center < f <= f_right:
                filterbank[i, j] = (f_right - f) / (f_right - f_center + 1e-10)

    return mx.array(filterbank, dtype=mx.float32)


def hann_window(window_length: int) -> mx.array:
    """Create a Hann window."""
    n = mx.arange(window_length, dtype=mx.float32)
    return 0.5 - 0.5 * mx.cos(2 * math.pi * n / window_length)


def stft(
    audio: mx.array,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: Optional[int] = None,
) -> mx.array:
    """Compute Short-Time Fourier Transform.

    Args:
        audio: Input audio of shape (samples,) or (batch, samples)
        n_fft: FFT size
        hop_length: Hop length between frames
        win_length: Window length (default: n_fft)

    Returns:
        Complex STFT output of shape (..., n_freqs, n_frames)
    """
    if win_length is None:
        win_length = n_fft

    # Handle batch dimension
    if audio.ndim == 1:
        audio = audio[None, :]
        squeeze_batch = True
    else:
        squeeze_batch = False

    batch_size, audio_length = audio.shape

    # Pad audio (MLX doesn't support reflect, use edge padding via numpy then convert back)
    pad_length = n_fft // 2
    # Use numpy for reflect padding
    audio_np = np.array(audio)
    audio_np = np.pad(audio_np, [(0, 0), (pad_length, pad_length)], mode="reflect")
    audio = mx.array(audio_np)

    # Create window
    window = hann_window(win_length)
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = mx.pad(window, [(pad_left, pad_right)])

    # Calculate number of frames
    n_frames = (audio.shape[1] - n_fft) // hop_length + 1

    # Extract frames using stride tricks
    frames = []
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[:, start:start + n_fft]
        frames.append(frame)

    # Stack frames: (batch, n_frames, n_fft)
    frames = mx.stack(frames, axis=1)

    # Apply window
    frames = frames * window

    # Compute FFT
    spectrum = mx.fft.rfft(frames, axis=-1)

    # Transpose to (batch, n_freqs, n_frames)
    spectrum = mx.transpose(spectrum, (0, 2, 1))

    if squeeze_batch:
        spectrum = spectrum.squeeze(0)

    return spectrum


def log_mel_spectrogram(
    audio: mx.array,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 100,
    sample_rate: int = 24000,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    normalize: bool = False,
    clip_val: float = 1e-7,  # Match PyTorch safe_log default
) -> mx.array:
    """Compute log mel spectrogram.

    Args:
        audio: Input audio of shape (samples,) or (batch, samples)
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_mels: Number of mel bands
        sample_rate: Audio sample rate
        f_min: Minimum frequency
        f_max: Maximum frequency
        normalize: Whether to normalize
        clip_val: Minimum value for log

    Returns:
        Log mel spectrogram of shape (batch, n_mels, n_frames)
    """
    # Handle dimensions
    if audio.ndim == 1:
        audio = audio[None, :]
        squeeze_batch = True
    else:
        squeeze_batch = False

    # Compute STFT
    spec = stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Get magnitude
    magnitude = mx.abs(spec)

    # Create mel filterbank
    mel_basis = create_mel_filterbank(n_fft, n_mels, sample_rate, f_min, f_max)

    # Apply mel filterbank: (batch, n_mels, n_frames)
    mel_spec = mx.matmul(mel_basis, magnitude)

    # Log scale
    log_mel = mx.log(mx.maximum(mel_spec, clip_val))

    if normalize:
        log_mel = (log_mel - mx.mean(log_mel)) / (mx.std(log_mel) + 1e-5)

    if squeeze_batch:
        log_mel = log_mel.squeeze(0)

    return log_mel


class MelSpectrogramExtractor:
    """Mel spectrogram extractor compatible with IndexTTS."""

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 100,
        sample_rate: int = 24000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        normalize: bool = False,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.normalize = normalize

        # Pre-compute mel filterbank
        self._mel_basis = None

    @property
    def mel_basis(self) -> mx.array:
        if self._mel_basis is None:
            self._mel_basis = create_mel_filterbank(
                self.n_fft, self.n_mels, self.sample_rate, self.f_min, self.f_max
            )
        return self._mel_basis

    def __call__(self, audio: mx.array) -> mx.array:
        """Extract log mel spectrogram.

        Args:
            audio: Input audio of shape (samples,) or (batch, samples)

        Returns:
            Log mel spectrogram of shape (batch, n_mels, n_frames)
        """
        return log_mel_spectrogram(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
            normalize=self.normalize,
        )
