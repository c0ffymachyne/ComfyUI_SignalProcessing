#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: C0ffymachyne
License: GPLv3
Version: 1.0.0

Description:
    various plotting methods for debugging and visualization
"""


import torch
import torchaudio
import torchaudio.functional as F
from typing import List


def enhance_harmonics(
    audio: torch.Tensor,
    sample_rate: int,
    harmonics: List[int] = [1, 3, 5, 7, 9, 11],
    gain_db: float = 5,
    base_frequency: float = 0,
    Q: float = 0.707,
) -> torch.Tensor:

    pitch = F.detect_pitch_frequency(audio, sample_rate)
    base_frequency = pitch.mean().item()
    if base_frequency <= 0:  # Fallback if pitch detection fails
        base_frequency = 440  # Use a default base frequency

    # Apply EQ boosts to specific harmonic frequencies
    for harmonic in harmonics:
        freq = base_frequency * harmonic
        if freq < sample_rate / 2:  # Ensure it's within the Nyquist frequency
            audio = F.equalizer_biquad(
                audio, sample_rate, center_freq=freq, gain=gain_db, Q=Q
            )

    return audio


def enhance_harmonics2(
    audio: torch.Tensor,
    sample_rate: int,
    harmonics: list[int] = [1, 2, 3, 4, 5],
    gain_db: float = 5,
    base_frequency: float = 0,
    Q: float = 0.707,
) -> torch.Tensor:
    """
    Enhance specified harmonics in an audio signal, emulating Distressor-like harmonic enhancement.
    Parameters:
        audio (Tensor): Input audio signal (1D or 2D [channels, samples]).
        sample_rate (int): Sampling rate of the audio.
        harmonics (list): List of harmonic multipliers to enhance.
        gain_db (float): Gain to apply to each harmonic.
        base_frequency (float, optional): Fundamental frequency. If None, it will be estimated.
        Q (float): Quality factor for the EQ bands.
    Returns:
        Tensor: Audio signal with enhanced harmonics.
    """

    if base_frequency == 0:
        # Detect the pitch frequency using torchaudio's pitch detection
        pitch = F.detect_pitch_frequency(audio, sample_rate)
        base_frequency = pitch.mean().item()
        if base_frequency <= 0:  # Fallback if pitch detection fails
            base_frequency = 440  # Default base frequency (A4)

    # Create a copy of the input signal for processing
    processed_audio: torch.Tensor = audio.clone()

    # Enhance harmonics using biquad EQ for precision
    for harmonic in harmonics:
        freq = base_frequency * harmonic
        if freq < sample_rate / 2:  # Ensure frequency is within Nyquist limit
            processed_audio = F.equalizer_biquad(
                processed_audio, sample_rate, center_freq=freq, gain=gain_db, Q=Q
            )

    # Apply a non-linear saturation for warmth and further harmonic enhancement
    def non_linear_saturation(audio: torch.Tensor, drive: float = 1.0) -> torch.Tensor:
        k = torch.tensor(1.0 + drive, dtype=audio.dtype, device=audio.device)
        return torch.tanh(k * audio) / torch.tanh(k)

    processed_audio = non_linear_saturation(processed_audio, drive=gain_db / 10.0)

    # Blend processed harmonics with the original signal
    output_audio = audio + processed_audio * (gain_db / 20.0)  # Scale blend by gain
    return output_audio / torch.max(torch.abs(output_audio))  # Normalize output


def batch_equalizer_biquad(
    audio: torch.Tensor, sample_rate: int, freqs: torch.Tensor, gain_db: float, Q: float
) -> torch.Tensor:
    """
    Apply biquad filters to enhance multiple harmonics in a batch.
    Parameters:
        audio (Tensor): Input audio signal (1D or 2D [channels, samples]).
        sample_rate (int): Sampling rate of the audio.
        freqs (Tensor): Frequencies for biquad filters.
        gain_db (float): Gain to apply to each harmonic.
        Q (float): Quality factor for all filters.
    Returns:
        Tensor: Audio signal with harmonics enhanced.
    """
    audio = audio.unsqueeze(0) if audio.dim() == 1 else audio

    # Precompute filter coefficients for all frequencies
    coeffs = [
        torchaudio.functional.equalizer_biquad(
            audio, sample_rate, center_freq=f, gain=gain_db, Q=Q
        )
        for f in freqs
    ]

    # Sum the filtered outputs for all harmonics
    filtered_audio = sum(coeffs)

    return filtered_audio


def enhance_harmonics3(
    audio: torch.Tensor,
    sample_rate: int,
    harmonics: List[int] = [1, 2, 3, 4, 5],
    gain_db: float = 5.0,
    base_frequency: float = 0,
    Q: float = 0.707,
) -> torch.Tensor:
    """
    Enhance specified harmonics in an audio signal, efficiently processing harmonics in a batch.
    Parameters:
        audio (Tensor): Input audio signal (1D or 2D [channels, samples]).
        sample_rate (int): Sampling rate of the audio.
        harmonics (list): List of harmonic multipliers to enhance.
        gain_db (float): Gain to apply to each harmonic.
        base_frequency (float, optional): Fundamental frequency. If None, it will be estimated.
        Q (float): Quality factor for the EQ bands.
    Returns:
        Tensor: Audio signal with enhanced harmonics.
    """
    if base_frequency is None:
        # Detect the pitch frequency using torchaudio's pitch detection
        pitch = torchaudio.functional.detect_pitch_frequency(audio, sample_rate)
        base_frequency = pitch.mean().item()
        if base_frequency <= 0:  # Fallback if pitch detection fails
            base_frequency = 440.0  # Default base frequency (A4)

    # Calculate harmonic frequencies
    harmonic_freqs = torch.tensor(
        [base_frequency * h for h in harmonics], device=audio.device
    )

    # Ensure frequencies are within Nyquist limit
    harmonic_freqs = harmonic_freqs[harmonic_freqs < sample_rate / 2]

    # Apply batched harmonic enhancement
    processed_audio = batch_equalizer_biquad(
        audio, sample_rate, harmonic_freqs, gain_db, Q
    )

    # Normalize the output
    return processed_audio / torch.max(torch.abs(processed_audio))


def enahnce_harmonics_23(
    audio: torch.Tensor, sample_rate: int, gain_db_base: int = 0, Q: float = 0.707
) -> torch.Tensor:

    audio = enhance_harmonics3(
        audio, sample_rate, harmonics=[2], gain_db=gain_db_base + 1, Q=0.303
    )
    audio = enhance_harmonics3(
        audio, sample_rate, harmonics=[3], gain_db=gain_db_base + 3, Q=0.303
    )

    return audio
