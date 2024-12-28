#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    utility functions for various filtering tasks
"""

import torch
from scipy.signal import butter
import torchaudio.functional as AF


def anti_aliasing_iir_filter(
    audio: torch.Tensor, sample_rate: int, cutoff: float = 0.0, order: int = 4
) -> torch.Tensor:
    """
    Apply an anti-aliasing IIR low-pass filter to the audio.

    Parameters:
        audio (Tensor): [channels, samples] audio signal.
        sr (int): Sample rate.
        cutoff (float): Cutoff frequency for the filter. Defaults to Nyquist limit (sr / 2).
        order (int): Order of the IIR filter.

    Returns:
        Tensor: Filtered audio signal.
    """
    if cutoff == 0:
        cutoff = sample_rate / 2  # Default to Nyquist frequency
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff / nyquist - 0.01

    # Design the Butterworth filter
    b, a = butter(order, normalized_cutoff, btype="low", output="ba")
    b = torch.tensor(b, dtype=audio.dtype, device=audio.device)
    a = torch.tensor(a, dtype=audio.dtype, device=audio.device)

    # Apply the filter
    filtered_audio = AF.lfilter(audio, b_coeffs=b, a_coeffs=a)

    return filtered_audio


def band_stop_filter(
    audio: torch.Tensor,
    sample_rate: int,
    low_cut: float,
    high_cut: float,
    filter_order: int = 2,
) -> torch.Tensor:
    """
    Apply a band-stop filter to attenuate lower mid frequencies.

    Parameters:
        audio (Tensor): [channels, samples] input audio signal.
        low_cut (float): Lower cutoff frequency of the band in Hz.
        high_cut (float): Upper cutoff frequency of the band in Hz.
        sr (int): Sample rate in Hz.
        filter_order (int): Order of the Butterworth filter.

    Returns:
        Tensor: Audio signal after band-stop filtering.
    """
    nyquist = sample_rate / 2
    normalized_band = [low_cut / nyquist, high_cut / nyquist]

    # Design band-stop Butterworth filter
    b, a = butter(filter_order, normalized_band, btype="bandstop", analog=False)

    # Convert coefficients to Torch tensors
    b = torch.tensor(b, dtype=audio.dtype, device=audio.device)
    a = torch.tensor(a, dtype=audio.dtype, device=audio.device)

    # Ensure [channels, samples] format
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Apply the filter using lfilter
    filtered_audio = AF.lfilter(audio, a_coeffs=a, b_coeffs=b, clamp=False)

    return filtered_audio.squeeze(0) if filtered_audio.size(0) == 1 else filtered_audio


def low_pass_filter(
    audio: torch.Tensor, sampler_rate: int, cutoff_freq: float, filter_order: int = 4
) -> torch.Tensor:
    """
    Apply a low-pass filter using a Butterworth filter.
    """
    # Design Butterworth filter using SciPy
    nyquist = sampler_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(filter_order, normalized_cutoff, btype="low", analog=False)

    # Convert coefficients to Torch tensors
    b = torch.tensor(b, dtype=audio.dtype, device=audio.device)
    a = torch.tensor(a, dtype=audio.dtype, device=audio.device)

    # Ensure [channels, samples] format
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Apply the filter using torchaudio
    filtered_audio = AF.lfilter(audio, a_coeffs=a, b_coeffs=b, clamp=False)

    return filtered_audio.squeeze(0) if filtered_audio.size(0) == 1 else filtered_audio


def butter_filter(
    audio: torch.Tensor,
    sample_rate: int,
    cutoff_freq: float,
    filter_type: str = "low",
    order: int = 4,
) -> torch.Tensor:
    """
    Create and apply a Butterworth filter (low-pass or high-pass).

    Parameters:
        audio (Tensor): [channels, samples] input audio signal.
        cutoff_freq (float): Cutoff frequency in Hz.
        sr (int): Sample rate in Hz.
        filter_type (str): "low" for low-pass, "high" for high-pass.
        order (int): Filter order.

    Returns:
        Tensor: Filtered audio signal.
    """
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)

    b = torch.tensor(b, dtype=audio.dtype, device=audio.device)
    a = torch.tensor(a, dtype=audio.dtype, device=audio.device)

    # Ensure [channels, samples] format
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    filtered_audio = AF.lfilter(audio, a_coeffs=a, b_coeffs=b, clamp=False)
    return filtered_audio.squeeze(0) if filtered_audio.size(0) == 1 else filtered_audio


def butter_low_pass(
    audio: torch.Tensor, sample_rate: int, cutoff_freq: float, order: int = 4
) -> torch.Tensor:
    return butter_filter(
        audio, sample_rate, cutoff_freq, filter_type="low", order=order
    )


def butter_high_pass(
    audio: torch.Tensor, sample_rate: int, cutoff_freq: int, order: int = 4
) -> torch.Tensor:
    return butter_filter(
        audio, sample_rate, cutoff_freq, filter_type="high", order=order
    )
