#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    various normalization methods
"""

import torch
import pyloudnorm as pyln  # pip install pyloudnorm
import numpy as np


def rms_normalization(audio: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(audio**2))
    scaling_factor = target_rms / rms
    normalized_audio = audio * scaling_factor
    return normalized_audio


def lufs_normalization(
    audio: torch.Tensor, sample_rate: int, target_lufs: float = -14.0
) -> torch.Tensor:
    meter = pyln.Meter(sample_rate)  # Create a loudness meter

    __audio = audio.T.cpu().numpy()
    loudness = meter.integrated_loudness(__audio)  # Current LUFS

    loudness_offset = target_lufs - loudness
    normalized_audio = __audio * (10 ** (loudness_offset / 20.0))

    result = torch.from_numpy(normalized_audio)
    result = result.T

    result = result.to(device=audio.device, dtype=result.dtype)

    return result


def peak_normalization(audio: torch.Tensor, target_peak: float = 0.9) -> torch.Tensor:
    peak = torch.max(torch.abs(audio))
    scaling_factor = target_peak / peak
    normalized_audio = audio * scaling_factor
    return normalized_audio


def get_loudness(audio: torch.Tensor, sample_rate: int) -> float:
    meter = pyln.Meter(sample_rate)  # Create a loudness meter
    audio = audio.T
    audio = audio.cpu()
    loudness: float = float(meter.integrated_loudness(audio.numpy()))  # Current LUFS
    return loudness


def set_loudness2(
    audio_signal: torch.Tensor, sample_rate: int, target_loudness_db: float = -20.0
) -> torch.Tensor:
    """
    Adjusts the loudness of the audio signal to a target level in decibels.

    Args:
        audio_signal (torch.Tensor): Input audio signal (channels, samples).
        sample_rate (int): Sample rate of the audio signal.
        target_loudness_db (float): Desired loudness in dB (e.g., -20.0 dB).

    Returns:
        torch.Tensor: Audio signal adjusted to the target loudness.
    """
    # Convert PyTorch tensor to NumPy array for loudness calculation
    audio_np = audio_signal.cpu().numpy().T  # Convert to [samples, channels]

    # Use pyloudnorm Meter to calculate and normalize loudness
    meter = pyln.Meter(sample_rate)  # Create loudness meter
    current_loudness = meter.integrated_loudness(audio_np)  # Measure LUFS

    # Compute loudness adjustment gain
    loudness_offset = target_loudness_db - current_loudness
    gain_factor = 10 ** (loudness_offset / 20.0)

    # Apply gain to adjust loudness
    adjusted_audio_np = audio_np * gain_factor

    # Convert back to PyTorch tensor
    adjusted_audio = torch.from_numpy(adjusted_audio_np.T).to(
        audio_signal.device, dtype=torch.float32
    )

    return adjusted_audio


def set_loudness(
    audio_signal: torch.Tensor, sample_rate: int, target_loudness_db: float = -20.0
) -> torch.Tensor:
    """
    Adjusts the loudness of the audio signal to a target level in decibels,
    ensuring no clipping occurs.

    Args:
        audio_signal (torch.Tensor): Input audio signal (channels, samples).
        sample_rate (int): Sample rate of the audio signal.
        target_loudness_db (float): Desired loudness in dB (e.g., -20.0 dB).

    Returns:
        torch.Tensor: Audio signal adjusted to the target loudness.
    """
    # Convert PyTorch tensor to NumPy array for loudness calculation
    audio_np = audio_signal.cpu().numpy().T  # Convert to [samples, channels]

    # Use pyloudnorm Meter to calculate and normalize loudness
    meter = pyln.Meter(sample_rate)  # Create loudness meter
    current_loudness = meter.integrated_loudness(audio_np)  # Measure LUFS

    # Compute loudness adjustment gain
    loudness_offset = target_loudness_db - current_loudness
    gain_factor = 10 ** (loudness_offset / 20.0)

    # Apply gain to adjust loudness
    adjusted_audio_np = audio_np * gain_factor

    # Prevent clipping by normalizing the peak
    peak_amplitude = np.max(np.abs(adjusted_audio_np))
    if peak_amplitude > 1.0:
        adjusted_audio_np = adjusted_audio_np / peak_amplitude

    # Convert back to PyTorch tensor
    adjusted_audio = torch.from_numpy(adjusted_audio_np.T).to(
        audio_signal.device, dtype=torch.float32
    )

    return adjusted_audio


def automatic_gain_control(
    audio: torch.Tensor, target_level: float = 0.7, alpha: float = 0.1
) -> torch.Tensor:
    current_level = torch.mean(torch.abs(audio))
    gain = target_level / (current_level + 1e-6)
    smoothed_gain = alpha * gain + (1 - alpha) * 1.0
    agc_audio = audio * smoothed_gain
    return agc_audio
