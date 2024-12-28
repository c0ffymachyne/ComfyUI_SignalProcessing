#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    utility functions operating in frequency domain
"""

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from scipy.signal import butter
from typing import Any


def tilt(
    audio: torch.Tensor,
    sample_rate: int,
    tilt_amount: float = -1.0,
    pivot_freq: float = 300,
    n_fft: int = 4096,
) -> torch.Tensor:
    """
    Apply a spectral tilt to audio.

    Parameters:
        audio (Tensor): [channels, samples] audio signal.
        sr (int): Sample rate.
        tilt_amount (float): Positive for brighter, negative for darker (dB/octave).
        pivot_freq (float): Pivot frequency in Hz.

    Returns:
        Tensor: Tilted audio signal.
    """
    hop_length = n_fft // 4

    # Perform STFT
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate).to(audio.device)

    # Calculate tilt filter
    slope = tilt_amount / (freqs.max() - pivot_freq)
    tilt_filter = 1 + slope * (freqs - pivot_freq)
    tilt_filter = tilt_filter.clamp(min=0.1)  # Avoid negative or zero gain

    # Apply filter to magnitude
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    tilted_mag = mag * tilt_filter.unsqueeze(0).unsqueeze(-1)

    # Reconstruct the tilted signal
    tilted_stft = tilted_mag * torch.exp(1j * phase)
    tilted_audio = torch.istft(
        tilted_stft, n_fft=n_fft, hop_length=hop_length, length=audio.size(-1)
    )

    return tilted_audio


def perceptual_weighted_central_frequency(
    audio: torch.Tensor, sample_rate: int, n_fft: int = 4096, hop_length: int = 1024
) -> torch.Tensor:
    """
    Calculate a perceptually weighted central frequency.

    Parameters:
        audio (Tensor): [channels, samples] audio signal.
        sr (int): Sample rate.
        n_fft (int): FFT size.
        hop_length (int): Hop length for STFT. Defaults to n_fft // 4.

    Returns:
        float: Perceptually weighted central frequency in Hz.
    """

    window = torch.hann_window(n_fft, device=audio.device)  # Hann window for STFT
    # Perform STFT
    stft = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window
    )
    mag = torch.abs(stft)  # Magnitude spectrum
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate).to(
        audio.device
    )  # Frequency bins

    # Apply A-weighting (approximation for perceptual loudness)
    a_weighting = (
        1.2589
        * ((12200**2) * (freqs**4))
        / (
            ((freqs**2) + (20.6**2))
            * ((freqs**2) + (12200**2))
            * torch.sqrt((freqs**2) + (107.7**2))
        )
    )

    # Ensure dimensions match for broadcasting
    a_weighting = a_weighting.unsqueeze(0).unsqueeze(-1)  # [1, freq_bins, 1]
    weighted_mag = mag * a_weighting  # Apply A-weighting

    # Calculate weighted central frequency
    spectral_sum = weighted_mag.sum(dim=1)  # Sum across frequency axis
    weighted_sum = (weighted_mag * freqs.unsqueeze(0).unsqueeze(-1)).sum(
        dim=1
    )  # Weighted sum of frequencies
    central_frequency = (weighted_sum / (spectral_sum + 1e-8)).mean()

    return central_frequency.item()


def copy_frequency_amplitudes_with_balance(
    source: torch.Tensor,
    target: torch.Tensor,
    sampler_rate: int,
    n_fft: int = 4096,
    hop_length: int = 1024,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Copy frequency amplitudes from the source to the target song with
    frequency balance adjustment for stereo signals.

    Parameters:
        source (Tensor): Source audio signal [channels, samples].
        target (Tensor): Target audio signal [channels, samples].
        sr (int): Sample rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT. Defaults to n_fft // 4.
        alpha (float): Blending factor for magnitude adjustment (0 = target-only, 1 = source-only).

    Returns:
        Tensor: Modified target audio signal with balanced frequency power.
    """

    # Ensure the source and target have the same number of channels
    if source.size(0) != target.size(0):
        raise ValueError("Source and target must have the same number of channels.")

    # Define frequency bands (in Hz) for low, mid, and high
    bands = [(0, 200), (200, 2000), (2000, sampler_rate // 2)]

    # Process each channel independently
    modified_channels = []
    for ch in range(source.size(0)):

        source_window = torch.hann_window(
            n_fft, device=source.device
        )  # Hann window for STFT
        target_window = torch.hann_window(
            n_fft, device=target.device
        )  # Hann window for STFT
        # Perform STFT on both source and target for the current channel
        source_stft = torch.stft(
            source[ch],
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
            window=source_window,
        )
        target_stft = torch.stft(
            target[ch],
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
            window=target_window,
        )

        # Extract magnitudes and phases
        source_mag = torch.abs(source_stft)  # Source magnitudes
        target_mag = torch.abs(target_stft)  # Target magnitudes
        target_phase = torch.angle(target_stft)  # Target phases

        # Frequency bins for the FFT
        freqs = torch.fft.rfftfreq(n_fft, 1 / sampler_rate).to(source.device)

        # Adjust frequency bands
        for low, high in bands:
            # Find indices for the frequency range
            band_indices = (freqs >= low) & (freqs < high)

            # Compute power in the band for source and target
            source_power = source_mag[band_indices, :].pow(2).mean()
            target_power = target_mag[band_indices, :].pow(2).mean()

            # Compute scaling factor
            if target_power > 0:
                scaling_factor = (source_power / target_power).sqrt()
            else:
                scaling_factor = 1.0  # Avoid division by zero

            # Apply scaling to target magnitudes in the band
            target_mag[band_indices, :] *= scaling_factor

        # Blend magnitudes: alpha * source_mag + (1 - alpha) * target_mag
        blended_mag = alpha * source_mag + (1 - alpha) * target_mag

        # Reconstruct STFT with blended magnitudes and original target phase
        modified_stft = blended_mag * torch.exp(1j * target_phase)

        # Perform ISTFT to reconstruct the modified target signal
        modified_audio = torch.istft(
            modified_stft,
            n_fft=n_fft,
            hop_length=hop_length,
            length=target.size(-1),
        )

        # Store the modified channel
        modified_channels.append(modified_audio)

    # Stack the modified channels back into a single tensor
    modified_audio = torch.stack(modified_channels, dim=0)

    return modified_audio


def get_emphasis_alpha(
    audio: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512
) -> torch.Tensor:
    """
    Analyze the audio signal and compute high and low-frequency energy ratios.

    Parameters:
        audio (Tensor): [channels, samples] input audio signal.
        sr (int): Sample rate of the audio.

    Returns:
        float: Ratio of high-frequency energy to total energy.
    """
    # Compute the Short-Time Fourier Transform (STFT)
    # Compute the Short-Time Fourier Transform (STFT)
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    magnitudes = torch.abs(
        stft
    )  # Get magnitude spectrum [channels, freq_bins, time_frames]

    # Define frequency ranges
    freqs = torch.fft.rfftfreq(n=2048, d=1 / sample_rate)  # Frequency bins

    # Create masks for low and high frequencies
    low_freqs = freqs < 500  # Low frequencies < 500 Hz
    high_freqs = freqs > 2000  # High frequencies > 2 kHz

    # Apply masks along the frequency dimension
    low_energy = magnitudes[low_freqs, :].pow(2).sum()  # Energy in low frequencies
    high_energy = magnitudes[high_freqs, :].pow(2).sum()  # Energy in high frequencies

    # Avoid division by zero
    if low_energy > 0:
        ratio = high_energy / low_energy
    else:
        ratio = float("inf")  # Dominated by high frequencies

    return ratio.item()


def get_emphasis_aplha_clipped(
    audio: torch.Tensor,
    sample_rate: int,
    min_alpha: float = 0.0,
    max_alpha: float = 0.99,
) -> Any:

    ratio = get_emphasis_alpha(audio.mean(dim=0), sample_rate)

    # Map the ratio to an alpha value
    # Higher ratio -> lower alpha (more high-frequency content, less emphasis)
    alpha = max_alpha - ratio / (1 + ratio) * (max_alpha - min_alpha)
    alpha = torch.clamp(
        torch.tensor(alpha), min_alpha, max_alpha
    ).item()  # Ensure alpha is within range

    return alpha


def pre_emphasis(audio: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """
    Apply pre-emphasis to boost high frequencies before saturation.

    Parameters:
        audio (Tensor): [channels, samples] input audio signal.
        alpha (float): Pre-emphasis factor (0 < alpha < 1).

    Returns:
        Tensor: Pre-emphasized audio signal.
    """
    x_shifted = F.pad(audio.unsqueeze(1), (1, 0))[:, :, :-1].squeeze(1)
    emphasized = audio - alpha * x_shifted
    return emphasized


def de_emphasis(audio: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """
    Apply de-emphasis to restore original frequency balance.

    Parameters:
        emphasized (Tensor): [channels, samples] input audio signal.
        alpha (float): De-emphasis factor (0 < alpha < 1).

    Returns:
        Tensor: De-emphasized audio signal.
    """
    # IIR filter coefficients
    b = torch.tensor([1.0, 0.0], dtype=audio.dtype, device=audio.device)
    a = torch.tensor([1.0, -alpha], dtype=audio.dtype, device=audio.device)

    # Ensure [channels, samples] format
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Apply IIR filter for de-emphasis
    x = AF.lfilter(audio, a_coeffs=a, b_coeffs=b, clamp=False)

    # Return to original dimensions
    if x.size(0) == 1:
        x = x.squeeze(0)
    return x


def olive_coloring(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Apply 'olive' sound coloring to an audio signal.

    Parameters:
        audio (Tensor): [channels, samples] audio signal.
        sr (int): Sample rate.

    Returns:
        Tensor: Colored audio signal.
    """
    n_fft = 4096
    hop_length = n_fft // 4
    window = torch.hann_window(n_fft, device=audio.device)  # Hann window for STFT

    # Apply spectral tilt
    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    mag = torch.abs(stft)  # Magnitude spectrum
    phase = torch.angle(stft)  # Phase spectrum
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate).to(
        audio.device
    )  # Frequency bins

    # Create tilt filter
    tilt_filter = (freqs + torch.max(freqs)) ** -0.7  # Gentle tilt down the highs
    tilt_filter = tilt_filter.clamp(
        0.9, 1.2
    )  # Prevent excessive attenuation or boosting
    tilt_filter = tilt_filter.unsqueeze(0).unsqueeze(
        -1
    )  # Broadcast across time dimension

    # Blend original magnitude with tilted magnitude
    tilted_mag = mag * tilt_filter.expand_as(mag)
    blended_mag = 0.75 * mag + 0.25 * tilted_mag  # Retain some original characteristics

    # Reconstruct audio
    colored_stft = blended_mag * torch.exp(1j * phase)
    audio_tilted = torch.istft(
        colored_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        length=audio.size(-1),
        window=window,
    )

    # Apply a gentle high-frequency attenuation (above 15 kHz)
    nyquist = sample_rate / 2
    cutoff = 15000 / nyquist

    b, a = butter(1, cutoff, btype="low", output="ba")  # 6 dB/octave slope
    b = torch.tensor(b, dtype=audio.dtype, device=audio.device)
    a = torch.tensor(a, dtype=audio.dtype, device=audio.device)
    audio_lowpassed = AF.lfilter(audio_tilted, a_coeffs=a, b_coeffs=b)

    # Add dynamic saturation (soft clipping)
    saturated_audio = torch.tanh(1.1 * audio_lowpassed)

    # saturated_audio = torch.atan(0.9 * saturated_audio)

    # Normalize output
    max_val = torch.max(torch.abs(saturated_audio))
    if max_val > 0:
        saturated_audio = saturated_audio / max_val

    return saturated_audio
